from model import objectives
#from .ChatCrossEmbeddingLayer import TexualEmbeddingLayer,VisualEmbeddingLayer
from .CrossEmbeddingLayer_tse import TexualEmbeddingLayer, VisualEmbeddingLayer
from .clip_model import build_CLIP_from_openai_pretrained, convert_weights
import torch
import torch.nn as nn 
import torch.nn.functional as F

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X

class RDE(nn.Module):
    def __init__(self, args, num_classes=11003):
        super().__init__()
        self.args = args
        self.num_classes = num_classes
        self._set_task()

        self.base_model, base_cfg = build_CLIP_from_openai_pretrained(args.pretrain_choice, args.img_size, args.stride_size)
        self.embed_dim = base_cfg['embed_dim']

        self.logit_scale = torch.ones([]) * (1 / args.temperature) 
        #self.QFormer = TwoBranchFeatureExtractor(ratio_img=0.3, ratio_txt=0.3)
        self.visul_emb_layer = VisualEmbeddingLayer(ratio=1.0)
        self.texual_emb_layer = TexualEmbeddingLayer(ratio=1.0)
 
        if 'TAL' in self.current_task:
            loss_type = 'TAL'
        elif 'TRL' in self.current_task:
            loss_type = 'TRL'
        elif 'InfoNCE' in self.current_task:
            loss_type = 'InfoNCE'
        elif 'SDM' in self.current_task:
            loss_type = 'SDM'
        else:
            exit()
        self.loss_type = loss_type
 
    def _set_task(self):
        loss_names = self.args.loss_names
        self.current_task = [l.strip() for l in loss_names.split('+')]
        print(f'Training Model with {self.current_task} tasks')
    
    def encode_chat(self,chat):
        x,attn = self.base_model.encode_text(chat)
        return x,attn

    def encode_image(self, image):
        x, _ = self.base_model.encode_image(image)
        return x[:, 0, :].float()
      
    def encode_text(self, text):
        x, _ = self.base_model.encode_text(text.long())
        return x[torch.arange(x.shape[0]), text.argmax(dim=-1)].float()

    def encode_image_tse(self, image):
        x,atten_i = self.base_model.encode_image(image)
        i_tse_f = self.visul_emb_layer(x, atten_i)   
        return i_tse_f.float()
 
    def encode_text_tse(self, text):
        x,atten_t = self.base_model.encode_text(text.long())
        t_tse_f = self.texual_emb_layer(x, text, atten_t)
        return t_tse_f.float()

    def encode_chat_tse(self,image,chat):
        x,attn_i,y,attn_c = self.base_model(image,chat)
        i_tse_f = self.visul_emb_layer( x, attn_i,cond_features=y, cond_atten=attn_c, text_ids=chat,use_cond=True, cond_ratio=1.0)
        c_tse_f = self.texual_emb_layer(y, chat, attn_c,cond_features=x, cond_atten=attn_i,use_cond=True, cond_ratio=1.0 )
        return i_tse_f.float(),c_tse_f.float()

    

    def forward(self, batch):
        ret = dict()
        ret.update({'temperature': 1 / self.logit_scale})

        images = batch['images']
        caption_ids = batch['caption_ids']
        chat_ids = batch['chat_ids']
        image_feats, atten_i, text_feats, atten_t = self.base_model(images, caption_ids)
        chat_feats,atten_c = self.encode_chat(chat_ids)
        i_feats = image_feats[:, 0, :].float()
        # i_feats = image_feats.float() # for CLIP ResNet visual model
        t_feats = text_feats[torch.arange(text_feats.shape[0]), caption_ids.argmax(dim=-1)].float()
        c_feats = chat_feats[torch.arange(chat_feats.shape[0]), chat_ids.argmax(dim=-1)].float()
        merge_feats = 0.3*c_feats+0.7*i_feats
        #merge_feats = i_feats
        i_tse_f = self.visul_emb_layer(image_feats, atten_i,cond_features=chat_feats, cond_atten=atten_c, text_ids=chat_ids,use_cond=True, cond_ratio=0.3)
        c_tse_f = self.texual_emb_layer(chat_feats, chat_ids, atten_c,cond_features=image_feats, cond_atten=atten_i,use_cond=True, cond_ratio=0.3 )
        t_tse_f = self.texual_emb_layer(text_feats, caption_ids, atten_t)
        merge_tse_f = 0.3*c_tse_f+0.7*i_tse_f
        #label_hat = batch['label_hat'].to(i_feats.device) 
     
        loss1,loss2 = objectives.compute_rbs(merge_feats, t_feats, merge_tse_f, t_tse_f, batch['pids'], \
                                               margin=self.args.margin,tau=self.args.tau,\
                                                loss_type=self.loss_type,logit_scale=self.logit_scale)
        loss3,_ = objectives.compute_gate_infonce_per_from_feats(c_feats,t_feats,i_feats,batch['pids'],logit_scale_pivot=50.0, logit_scale_ti=50.0,
    gamma=1.0, symmetric_ti=True, clip_w=(0.2, 5.0))

        loss3 = loss3.mean()

        ret.update({'bge_loss':loss1})
        ret.update({'tse_loss':loss2})
        ret.update({'merge_loss':loss3})

  
        return ret


def build_model(args, num_classes=11003):
    model = RDE(args, num_classes)
    # covert model to fp16
    convert_weights(model)
    return model
