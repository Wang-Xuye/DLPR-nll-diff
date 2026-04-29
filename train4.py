import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # 指定只有第二张GPU对PyTorch可见
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.backends.cudnn as cudnn

from utils.data.datasets import ImageDataset
from utils.data.transform import build_transforms
import random
import numpy as np
import math
from tqdm import tqdm

from nll_model import NearLosslessCompressor, RateDistortion
from nll_model_p import NearLosslessCompressor_I, NearLosslessCompressor_P, RateDistortion, unpatchify_from_batch, patchify_to_batch
from global_nii_max import nii_max, nii_max_f
def replicate_pad(x, p):
    height=x.shape[-2]
    width=x.shape[-1]
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    padding_right = new_w - width
    padding_bottom = new_h - height
    if padding_right == 0 and padding_bottom == 0:
        return x
    return np.pad(x, [(0, 0), (0, padding_bottom), (0, padding_right)], mode='edge')



def configure_optimizers(model, lr, aux_lr):

    parameters = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles") and not n.startswith("residual_compressor_cond.") and not n.startswith("fusion_cond.") and p.requires_grad
    }
    parameters_nll = {
        n
        for n, p in model.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad and (n.startswith("residual_compressor_cond.") or n.startswith("fusion_cond."))
    }
    aux_parameters = {
        n
        for n, p in model.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }
    # Make sure we don't have an intersection of parameters
    params_dict = dict(model.named_parameters())
    inter_params1 = parameters & aux_parameters
    inter_params2 = parameters_nll & aux_parameters
    inter_params3 = parameters & parameters_nll
    union_params = parameters | parameters_nll | aux_parameters

    assert len(inter_params1) == 0
    assert len(inter_params2) == 0
    assert len(inter_params3) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=lr,
    )
    optimizer_nll = optim.Adam(
        (params_dict[n] for n in sorted(parameters_nll)),
        lr=lr,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=aux_lr,
    )

    return optimizer, aux_optimizer, optimizer_nll

class Dataset_seq():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.crop = 64
        self.clips = []
        for root, _, files in os.walk(self.root_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                clip = np.load(full_path).transpose(2, 0, 1)
                self.clips.append(clip)
            
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, index):
        clip0 = self.clips[index]              # (T, H, W)
        clip1 = np.expand_dims(clip0, axis=1) # (T, 1, H, W)

        T, C, H, W = clip1.shape
        start_idx = np.random.randint(0, T - 8) 
        clip1 = clip1[start_idx:start_idx+8, :, :, :]
        # ===== 随机翻转（整段同步）=====
        if random.random() < 0.5:
            clip1 = clip1[:, :, :, ::-1] 
        if random.random() < 0.5:
            clip1 = clip1[:, :, ::-1, :] 
        clip1 = np.ascontiguousarray(clip1, dtype=np.float32)
        return clip1
        
class Dataset_test():
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.crop = 64
        self.clips = []
        for root, _, files in os.walk(self.root_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                clip = np.load(full_path)      
                clip=replicate_pad(clip,128)
                self.clips.append(clip)
            
    def __len__(self):
        return len(self.clips)
    
    def __getitem__(self, index):
        clip0 = self.clips[index]              # (T, H, W)
        clip1 = np.expand_dims(clip0, axis=1) # (T, 1, H, W)
        clip1 = np.ascontiguousarray(clip1, dtype=np.float32)
        return clip1
        
def train_one_epoch(model_I, model_P, criterion, train_dataloader, optimizer_P, aux_optimizer_P, optimizer_nll_P, train_step, tb_writer=None, clip_max_norm=None):
    
    model_I.eval()
    model_P.train()
    device = next(model_I.parameters()).device
    
    train_size = 0
    for x in tqdm(train_dataloader):

        x = x.to(device).contiguous()
        tau=random.randint(1,5)
        lastframe= None
        for i in range(1, 8):
            optimizer_P.zero_grad()
            aux_optimizer_P.zero_grad()
            optimizer_nll_P.zero_grad()
            if i==1:
                inp=patchify_to_batch(x[:,0],64)
                out = model_I(inp, tau)
                lastframe=unpatchify_from_batch(out["rec_img"].detach(), x[:,i].shape[0], x[:,i].shape[2],x[:,i].shape[3])
            
            out = model_P(x[:,i], lastframe, tau)
            
            lastframe=unpatchify_from_batch(out["rec_img"].detach(), x[:,i].shape[0], x[:,i].shape[2],x[:,i].shape[3])

            out_criterion = criterion(out, patchify_to_batch(x[:,i],64))
            out_criterion["loss"].backward()
            if clip_max_norm:
                nn.utils.clip_grad_norm_(model_P.parameters_wo_cond(), clip_max_norm)
            optimizer_P.step()
    
            out_criterion["res_bpp_cond"].backward()
            if clip_max_norm:
                nn.utils.clip_grad_norm_(model_P.parameters_w_cond(), clip_max_norm)       
            optimizer_nll_P.step()
            
            aux_loss = model_P.lossy_compressor.aux_loss()
            aux_loss.backward()
            aux_optimizer_P.step()

            
        
def eval_epoch(model_I, model_P, criterion, eval_dataloader, epoch, tb_writer=None):
    
    model_I.eval()
    model_P.eval()
    
    device = next(model_I.parameters()).device
    
    loss = 0
    img_bpp_i = 0
    res_bpp_i = 0
    res_bpp_cond_i = 0
    img_bpp_p = 0
    res_bpp_p = 0
    res_bpp_cond_p = 0
    mse_loss = 0
    aux_loss = []

    
    eval_size = 0
    with torch.no_grad():
        for x in eval_dataloader:
            x = x.to(device).contiguous()
            tau=random.randint(1,5)
            eval_size+=1
            Lastframe=None
            for i in range(1, 8):
                if i==1:
                    inp=patchify_to_batch(x[:,0],64)
                    out = model_I(inp, tau)
                    lastframe=unpatchify_from_batch(out["rec_img"].detach(), x[:,i].shape[0], x[:,i].shape[2],x[:,i].shape[3])
                
                out = model_P(x[:,i], lastframe, tau)
                lastframe=unpatchify_from_batch(out["rec_img"].detach(), x[:,i].shape[0], x[:,i].shape[2],x[:,i].shape[3])
                
                out_criterion = criterion(out,  patchify_to_batch(x[:,i],64))
                img_bpp_p += out_criterion["img_bpp"] 
                res_bpp_p += out_criterion["res_bpp"]  
                res_bpp_cond_p += out_criterion["res_bpp_cond"]                    
           
    img_bpp_p /=eval_size*7
    res_bpp_p /=eval_size*7
    res_bpp_cond_p /=eval_size*7

    torch.save(model_P.state_dict(), 'log4/p-Ep{}-{:.3f}-{:.3f}-{:.3f}.pth'.format(epoch + 1, img_bpp_p, img_bpp_p+res_bpp_p, img_bpp_p+res_bpp_cond_p))   
        


    
    
def train(train_dataloader, eval_dataloader, epochs, ckp_dir, log_dir, resume=False):
    nll_module_I = NearLosslessCompressor_I(192, 5, "3P").cuda()
    nll_module_P = NearLosslessCompressor_P(192, 5, "3P").cuda()
    
    rd_criterion = RateDistortion(lmbda=0.05)

    ckp = torch.load(os.path.join(ckp_dir, "ckp_nll/0001/ckp.tar"))
    nll_module_I.load_state_dict(ckp['model_state_dict'])

    optimizer_P, aux_optimizer_P, optimizer_nll_P = configure_optimizers(nll_module_P, lr=1e-4, aux_lr=1e-3)


    scheduler_P = optim.lr_scheduler.MultiStepLR(optimizer_P, milestones=[350, 390, 430, 470, 510, 550, 590], gamma=0.9, verbose=True)
    scheduler_nll_P = optim.lr_scheduler.MultiStepLR(optimizer_nll_P, milestones=[350, 390, 430, 470, 510, 550, 590], gamma=0.9, verbose=True)
    tb_writer = SummaryWriter(log_dir)

    start_epoch = 0
    train_step = 0

    for epoch in range(start_epoch, epochs):
        # train
        train_one_epoch(nll_module_I, nll_module_P, rd_criterion, train_dataloader, optimizer_P, aux_optimizer_P, optimizer_nll_P, train_step, tb_writer, clip_max_norm=1.0)
        # eval
        if epoch%30==0:
            eval_epoch(nll_module_I, nll_module_P, rd_criterion, eval_dataloader, epoch, tb_writer)

        scheduler_P.step()
        scheduler_nll_P.step()

    tb_writer.close()
    
    
if __name__ == "__main__":
    

    epochs = 10000
    resume = False

    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)

    ckp_dir=""
    log_dir=""

    train_dataset = Dataset_seq("/root/autodl-tmp/Mosmed/train-full")

    test_dataset = Dataset_test("/root/autodl-tmp/Mosmed/test-slice")

    train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, prefetch_factor=2, pin_memory=True)

    eval_dataloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=8, prefetch_factor=2, pin_memory=True)
    
    train(train_dataloader, eval_dataloader, epochs, ckp_dir, log_dir, resume)
    
    