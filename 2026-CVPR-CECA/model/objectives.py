import math
import torch
import torch.nn as nn
import torch.nn.functional as F

 
def compute_sdm_per(scores, pid, logit_scale, epsilon=1e-8):
    """
    Similarity Distribution Matching
    """
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float()

    t2i_cosine_theta = scores
    i2t_cosine_theta = t2i_cosine_theta.t()

    text_proj_image = logit_scale * t2i_cosine_theta
    image_proj_text = logit_scale * i2t_cosine_theta

    # normalize the true matching distribution
    labels_distribute = labels / labels.sum(dim=1)

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * (F.log_softmax(image_proj_text, dim=1) - torch.log(labels_distribute + epsilon))
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(labels_distribute + epsilon))

    loss = torch.sum(i2t_loss, dim=1) + torch.sum(t2i_loss, dim=1)

    return loss

def compute_TRL_per(scores, pid, margin = 0.2, tau=0.02):       
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_1 =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_2 = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    pos_1 = (alpha_1 * scores).sum(1)
    pos_2 = (alpha_2 * scores.t()).sum(1)

    neg_1 = (mask*scores).max(1)[0]
    neg_2 = (mask*scores.t()).max(1)[0]

    cost_1 = (margin + neg_1 - pos_1).clamp(min=0)
    cost_2 = (margin + neg_2 - pos_2).clamp(min=0)
    return cost_1 + cost_2

 
def compute_InfoNCE_per(scores, logit_scale):
    
    # cosine similarity as logits
    logits_per_image = logit_scale * scores
    logits_per_text = logits_per_image.t()

    p1 = F.softmax(logits_per_image, dim=1)
    p2 = F.softmax(logits_per_text, dim=1)

    loss = (- p1.diag().log() - p2.diag().log())/2    
    return loss

def compute_TAL_per(scores, pid, tau, margin):
    batch_size = scores.shape[0]
    pid = pid.reshape((batch_size, 1)) # make sure pid size is [batch_size, 1]
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).float().cuda()
    mask = 1 - labels

    alpha_i2t =((scores/tau).exp()* labels / ((scores/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()
    alpha_t2i = ((scores.t()/tau).exp()* labels / ((scores.t()/tau).exp()* labels).sum(dim=1, keepdim=True)).detach()

    loss = (-  (alpha_i2t*scores).sum(1) + tau * ((scores / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)  \
        +  (-  (alpha_t2i*scores.t()).sum(1) + tau * ((scores.t() / tau).exp() * mask).sum(1).clamp(max=10e35).log() + margin).clamp(min=0)
    
    return loss 

def compute_rbs(i_feats, t_feats, i_tse_f, t_tse_f, pid, label_hat=None, tau=0.02, margin=0.1, loss_type='TAL', logit_scale=50):

    loss_bge, _ = compute_per_loss(i_feats, t_feats, pid, tau, margin, loss_type, logit_scale)
    loss_tse, _ = compute_per_loss(i_tse_f, t_tse_f, pid, tau, margin, loss_type, logit_scale)
    #loss_merge = compute_gate_infonce_per_from_feats()
    loss_bge = loss_bge.sum()
    loss_tse = loss_tse.sum()
    #loss_merge = 
    if loss_type in ['TAL','TRL']:
        return loss_bge, loss_tse
        #return loss_bge
    #else:
        #return  loss_tse.sum() # mean

def compute_per_loss(image_features, text_features, pid, tau=0.02, margin=0.2, loss_type='TAL', logit_scale=50):
    
    # # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)
    scores = text_norm @ image_norm.t()

    if 'TAL' in loss_type:
        per_loss = compute_TAL_per(scores, pid, tau, margin=margin)
    elif 'TRL' in loss_type:
        per_loss = compute_TRL_per(scores, pid, tau=tau, margin=margin)
    elif 'InfoNCE' in loss_type:
        per_loss = compute_InfoNCE_per(scores, logit_scale)
    elif 'SDM' in loss_type:
        per_loss = compute_sdm_per(scores, pid, logit_scale)
    else:
        exit()

    return per_loss, scores.diag()



import torch
import torch.nn.functional as F

# ========= helpers =========

def _normalize(x): 
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)

def _build_pos_mask(pid, device):

    B = pid.shape[0] if pid is not None else None
    if pid is None:
      
        raise ValueError("pid is None, please pass B to make identity mask outside or provide pid.")
    pid = pid.reshape((B, 1))
    pid_dist = pid - pid.t()
    labels = (pid_dist == 0).to(device)
    return labels  # [B,B] bool/byte

def _row_multi_pos_ce(logits, pos_mask):

    m = logits.max(dim=1, keepdim=True).values           # [B,1]
    exp_all = torch.exp(logits - m)                      # [B,B]
    denom = exp_all.sum(dim=1) + 1e-12                   # [B]
    exp_pos = (exp_all * pos_mask.float()).sum(dim=1) + 1e-12
    return -(torch.log(exp_pos) - torch.log(denom))      # [B]

def _row_pos_prob(logits, pos_mask):

    m = logits.max(dim=1, keepdim=True).values
    exp_all = torch.exp(logits - m)
    denom = exp_all.sum(dim=1) + 1e-12
    exp_pos = (exp_all * pos_mask.float()).sum(dim=1) + 1e-12
    return (exp_pos / denom).clamp(1e-8, 1.0)



def compute_gate_infonce_per_from_feats(
    c_feats, t_feats, i_feats, pid=None,
    logit_scale_pivot=50.0, logit_scale_ti=50.0,
    gamma=1.0, symmetric_ti=True, clip_w=(None, None)
):
    
    device = c_feats.device
    C = _normalize(c_feats)
    T = _normalize(t_feats)
    I = _normalize(i_feats)

    B = C.size(0)
    if pid is None:
        pos_mask = torch.eye(B, device=device, dtype=torch.bool)
    else:
        pos_mask = _build_pos_mask(pid, device)


    scores_ct = T @ C.t()      
    scores_ci = I @ C.t()      
    scores_ti = I @ T.t()      

    return compute_gate_infonce_per_from_scores(
        scores_ti=scores_ti, scores_ct=scores_ct.t(), scores_ci=scores_ci.t(),
        pid=pid, logit_scale_pivot=logit_scale_pivot, logit_scale_ti=logit_scale_ti,
        gamma=gamma, symmetric_ti=symmetric_ti, clip_w=clip_w
    )



def compute_gate_infonce_per_from_scores(
    scores_ti, scores_ct, scores_ci, pid=None,
    logit_scale_pivot=50.0, logit_scale_ti=50.0,
    gamma=1.0, symmetric_ti=True, clip_w=(None, None)
):
   
    device = scores_ti.device
    B = scores_ti.size(0)

    if pid is None:
        pos_mask = torch.eye(B, device=device, dtype=torch.bool)
    else:
        pos_mask = _build_pos_mask(pid, device)


    logits_ct = logit_scale_pivot * scores_ct         # c->t
    logits_ci = logit_scale_pivot * scores_ci         # c->i
    L_ct_row = _row_multi_pos_ce(logits_ct, pos_mask) # [B]
    L_ci_row = _row_multi_pos_ce(logits_ci, pos_mask) # [B]
    p_ct = _row_pos_prob(logits_ct, pos_mask)         # [B]
    p_ci = _row_pos_prob(logits_ci, pos_mask)         # [B]


    with torch.no_grad():
        w = (p_ct * p_ci).pow(gamma).clamp_min(1e-8)  # 
        w = w / (w.mean().clamp_min(1e-8))            #
        #w = w / (w.sum().clamp_min(1e-8))  
        lo, hi = clip_w
        if lo is not None:
            w = torch.maximum(w, torch.tensor(lo, device=device, dtype=w.dtype))
        if hi is not None:
            w = torch.minimum(w, torch.tensor(hi, device=device, dtype=w.dtype))


    logits_ti = logit_scale_ti * scores_ti           # t->i
    L_ti_row = _row_multi_pos_ce(logits_ti, pos_mask)

    if symmetric_ti:
        logits_it = logit_scale_ti * scores_ti.t()   # i->t
        L_it_row = _row_multi_pos_ce(logits_it, pos_mask)
        L_ti_mix = 0.5 * (L_ti_row + L_it_row)
    else:
        L_ti_mix = L_ti_row


    #per_sample = (1.0 - w) * (L_ct_row + L_ci_row) + w * L_ti_mix
    per_sample = w * (L_ct_row + L_ci_row) + (1.0-w) * L_ti_mix

    diag_ti = scores_ti.diag()

    return per_sample, diag_ti
