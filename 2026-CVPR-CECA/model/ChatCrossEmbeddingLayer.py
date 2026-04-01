import torch
import torch.nn as nn
import torch.nn.functional as F


def l2norm(X, dim, eps=1e-6):          
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    return X / norm

def maxk(x, dim, k):
    idx = x.topk(k, dim=dim)[1]
    return x.gather(dim, idx)

def maxk_pool1d_var(x, dim, k, lengths):
    results = []
    lengths = [int(v) for v in list(lengths.detach().cpu().numpy())]
    for i, L in enumerate(lengths):
        kk = max(1, min(k, L))
        results.append(maxk(x[i, :L, :], dim - 1, kk).mean(dim - 1))
    return torch.stack(results, dim=0)

def maybe_half(x):                      
    return x.half() if x.is_cuda else x

@torch.no_grad()
def text_topk_from_attn(attn, text_ids, ratio: float, min_k: int = 1):

    B, Nq, Nt = attn.shape
    dev = attn.device
    valid = text_ids.ne(0)
    sos = torch.zeros(B, dtype=torch.long, device=dev)
    eos = text_ids.argmax(dim=-1)
    row = attn[torch.arange(B, device=dev), eos, :]     # [B,Nt]
    row = row.masked_fill(~valid, float('-inf'))
    row[torch.arange(B), sos] = float('-inf')
    row[torch.arange(B), eos] = float('-inf')
    K = max(min_k, int((Nt - 2) * float(ratio)))
    max_avail = int(valid.sum(1).min().item()) - 2
    if max_avail < 1:
        return torch.ones(B, 1, dtype=torch.long, device=dev)
    K = min(K, max_avail)
    _, idx = row.topk(K, dim=-1)
    return idx  # [B,K]

@torch.no_grad()
def image_topk_from_attn(attn, ratio: float, min_k: int = 1):

    if attn.dim() == 4:                 # [B,H,Ni,Ni] 
        attn = attn.mean(dim=1)
    if attn.size(1) != attn.size(2):    # [B,Nq,Ni] 
        col = attn.mean(dim=1)          # [B,Ni]
    else:
        col = attn.mean(dim=1)          # [B,Ni]
    B, Ni = col.size()
    col[:, 0] = float('-inf')
    K = max(min_k, int((Ni - 1) * float(ratio)))
    K = min(K, Ni - 1)
    _, idx = col.topk(K, dim=-1)
    return idx  # [B,K]

def gather_by_index(x, idx):            # x:[B,N,D], idx:[B,K] -> [B,K,D]
    return x.gather(1, idx.unsqueeze(-1).expand(-1, -1, x.size(-1)))

# ----------------- fp16-safe LN & MLP -----------------
class Fp32LayerNorm(nn.LayerNorm):
    def forward(self, x):
        y = F.layer_norm(
            x.float(),
            self.normalized_shape,
            self.weight.float() if self.weight is not None else None,
            self.bias.float()   if self.bias   is not None else None,
            self.eps,
        )
        return y.to(x.dtype)

class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.bns    = nn.ModuleList(nn.BatchNorm1d(k) for k in h + [output_dim])

    def forward(self, x):               # x: [B,N,D]
        B, N, D = x.size()
        y = x.reshape(B * N, D)
        orig = y.dtype
        for i, (bn, layer) in enumerate(zip(self.bns, self.layers)):
            y = layer(y)
            if i < self.num_layers - 1:
                y = bn(y.float()).to(orig)
                y = F.relu(y)
        return y.view(B, N, self.output_dim)


class CondRefiner(nn.Module):
    def __init__(self, d=512, nhead=8, mlp_ratio=2):
        super().__init__()
        self.ln_q  = Fp32LayerNorm(d)
        self.sa    = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln_c  = Fp32LayerNorm(d)
        self.ca    = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.mlp   = nn.Sequential(
            nn.Linear(d, d*mlp_ratio), nn.GELU(), nn.Linear(d*mlp_ratio, d)
        )
        self.g     = nn.Parameter(torch.tensor(-3.0))

    def forward(self, q_tokens, cond_tokens=None, return_attn=False):
        q = self.ln_q(q_tokens)
        sa_out, _ = self.sa(q, q, q, need_weights=False)
        y = sa_out
        attn_weights = None
        if cond_tokens is not None:
            c = self.ln_c(q_tokens)
            ca_out, attn_weights = self.ca(c, cond_tokens, cond_tokens, need_weights=True)
            y = y + ca_out
        y = self.mlp(y)
        gate = torch.sigmoid(self.g)
        out = q_tokens + gate * y
        if return_attn:
            return out, attn_weights  
        else:
            return out


class TexualEmbeddingLayer(nn.Module):

    def __init__(self, input_dim=512, embed_dim=1024, ratio=0.3, nhead=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.ratio = ratio
        self.refiner = CondRefiner(d=input_dim, nhead=nhead, mlp_ratio=2)

    def forward(self, features, text, atten,
                cond_features=None, cond_atten=None, use_cond: bool=False, cond_ratio: float=None):

        B, Nt, D = features.size()
        device = features.device
        mask = (text != 0).long()
        lengths = mask.sum(1).view(-1) - 2                
        k = max(1, int((atten.size(1) - 2) * self.ratio))


        att = atten.clone()
        eos = text.argmax(dim=-1)
        att[torch.arange(B), :, eos] = -1
        att[torch.arange(B), :, 0]   = -1
        att = att[torch.arange(B), eos, :] * mask         # [B,Nt]
        idx_topk = att.topk(dim=-1, k=k)[1]               # [B,k]
        tok_k = gather_by_index(features, idx_topk)       # [B,k,D]


        cond_tok = None
        if use_cond and (cond_features is not None):
            cr = self.ratio if cond_ratio is None else cond_ratio
            idx_c = image_topk_from_attn(cond_atten, cr)  # [B,Kc]
            cond_tok = gather_by_index(cond_features, idx_c)  # [B,Kc,D]

        tok_k = l2norm(tok_k, dim=-1)
        tok_k = self.refiner(tok_k, cond_tok)             # [B,k,D]


        cap_emb = self.linear(maybe_half(tok_k))
        feat = self.mlp(tok_k) + cap_emb

        lengths = torch.tensor([min(int(lengths[i].item()), k) for i in range(B)], device=device)
        out = maxk_pool1d_var(feat, 1, 1, lengths.half())
        return out.float()

class VisualEmbeddingLayer(nn.Module):

    def __init__(self, input_dim=512, embed_dim=1024, ratio=0.3, nhead=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.ratio = ratio
        self.fc  = nn.Linear(input_dim, embed_dim)
        self.mlp = MLP(input_dim, embed_dim // 2, embed_dim, 2)
        self.refiner = CondRefiner(d=input_dim, nhead=nhead, mlp_ratio=2)

    def forward(self, base_features, atten,
                cond_features=None, cond_atten=None, text_ids=None,
                use_cond: bool=False, cond_ratio: float=None):
  
        B, Ni, D = base_features.size()
        device = base_features.device
        k = max(1, int((atten.size(1) - 1) * self.ratio))


        att = atten.clone()
        att[torch.arange(B), :, 0] = -1
        idx_topk = att[:, 0].topk(dim=-1, k=k)[1]         
        tok_k = gather_by_index(base_features, idx_topk)  # [B,k,D]


        cond_tok = None
        if use_cond and (cond_features is not None):
            cr = self.ratio if cond_ratio is None else cond_ratio
            if text_ids is not None:
                
                idx_c = text_topk_from_attn(cond_atten, text_ids, cr)  # [B,Kc]
            else:
               
                idx_c = image_topk_from_attn(cond_atten, cr)
            cond_tok = gather_by_index(cond_features, idx_c)           # [B,Kc,D]


        tok_k = l2norm(tok_k, dim=-1)
        tok_k = self.refiner(tok_k, cond_tok)             # [B,k,D]


        tok_k = tok_k.half()
        feat_lengths = torch.full((B,), tok_k.size(1), device=device).half()
        feat = self.fc(tok_k)
        feat = self.mlp(tok_k) + feat
        out = maxk_pool1d_var(feat, 1, 1, feat_lengths)
        return out.float()
