from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

EPS=1e-7
class Normalize(nn.Module):
    def forward(self,x): return x / x.pow(2).sum(1,keepdim=True).pow(.5).clamp_min(EPS)
class Embed(nn.Module):
    def __init__(self,dim_in,dim_out): super().__init__(); self.linear=nn.Linear(dim_in,dim_out); self.l2norm=Normalize()
    def forward(self,x): return self.l2norm(self.linear(x.reshape(x.shape[0],-1)))
class ContrastLoss(nn.Module):
    def __init__(self,n_data): super().__init__(); self.n_data=n_data
    def forward(self,x,mask):
        b=x.shape[0]; m=x.size(1)-1; pn=1/float(self.n_data)
        pos=x.select(1,0); lp=(pos/(pos+m*pn+EPS)).log()*mask.reshape(-1,1)
        neg=x.narrow(1,1,m); l0=((neg.clone().fill_(m*pn))/(neg+m*pn+EPS)).log()*mask.reshape(-1,1,1)
        return -(lp.sum(0)+l0.reshape(-1,1).sum(0))/b
class ContrastMemory(nn.Module):
    def __init__(self,dim,n_data,k,temp=.07,momentum=.05):
        super().__init__(); self.k=k; self.register_buffer("params",torch.tensor([k,temp,-1.,-1.,momentum]))
        std=1/math.sqrt(dim/3); self.register_buffer("memory_v1",torch.rand(n_data,dim).mul_(2*std).add_(-std)); self.register_buffer("memory_v2",torch.rand(n_data,dim).mul_(2*std).add_(-std))
    def forward(self,v1,v2,index,sample_idx):
        k=int(self.params[0]); temp=float(self.params[1]); b,dim=v1.shape; n=self.memory_v1.shape[0]
        idx=sample_idx.long(); w1=self.memory_v1.index_select(0,idx.reshape(-1)).detach().reshape(b,k+1,dim); out2=torch.exp(torch.bmm(w1,v2.reshape(b,dim,1))/temp)
        w2=self.memory_v2.index_select(0,idx.reshape(-1)).detach().reshape(b,k+1,dim); out1=torch.exp(torch.bmm(w2,v1.reshape(b,dim,1))/temp)
        if float(self.params[2])<0: self.params[2]=out1.mean().detach()*n
        if float(self.params[3])<0: self.params[3]=out2.mean().detach()*n
        z1=float(self.params[2]); z2=float(self.params[3]); z1=z2 if math.isnan(z1) else z1
        out1=(out1/z1).contiguous(); out2=(out2/z2).contiguous(); mom=float(self.params[4])
        with torch.no_grad():
            a=self.memory_v1.index_select(0,index)*mom+v1*(1-mom); a=a/a.pow(2).sum(1,keepdim=True).sqrt().clamp_min(EPS); self.memory_v1.index_copy_(0,index,a)
            a=self.memory_v2.index_select(0,index)*mom+v2*(1-mom); a=a/a.pow(2).sum(1,keepdim=True).sqrt().clamp_min(EPS); self.memory_v2.index_copy_(0,index,a)
        return out1,out2
class CanonicalUnisCRDLoss(nn.Module):
    """Mathematically faithful port of root unis_crdloss.py; only device handling is corrected."""
    def __init__(self,s_dim,t_dim,feat_dim,n_data,nce_k,nce_t=.07,nce_m=.05):
        super().__init__(); self.embed_s=Embed(s_dim,feat_dim); self.embed_t=Embed(t_dim,feat_dim); self.contrast=ContrastMemory(feat_dim,n_data,nce_k,nce_t,nce_m); self.cs=ContrastLoss(n_data); self.ct=ContrastLoss(n_data)
    def forward(self,fs,ft,ps,pt,index,sample_idx,labels):
        fs=self.embed_s(fs); ft=self.embed_t(ft)
        entropy=-(F.softmax(pt,1)*torch.log(F.softmax(pt,1)+1e-5)).sum(1); weight=1+torch.exp(-entropy); weight=32*weight/weight.sum(); con=weight.reshape(-1,1,1)
        spa=F.softmax(ps,1).argmax(1); tpa=F.softmax(pt,1).argmax(1)
        ac=spa.eq(labels); bc=tpa.eq(labels); either=ac|bc; both=ac&bc; aa=(~ac)|both; bb=(~bc)|both
        if not bool(either.any()): either=~either; aa=~aa; bb=~bb
        mask=bc.float(); fsa=fs.detach(); fta=ft.detach(); fs=torch.stack([fsa[i] if not bool(aa[i]) else fs[i] for i in range(len(fs))]); ft=torch.stack([fta[i] if not bool(bb[i]) else ft[i] for i in range(len(ft))])
        mask=torch.where(mask==0,torch.full_like(mask,-1),mask); os_,ot=self.contrast(fs,ft,index,sample_idx)
        return self.cs(os_*con,mask)+self.ct(ot*con,mask)
