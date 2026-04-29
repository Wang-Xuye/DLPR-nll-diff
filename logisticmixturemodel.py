import torch
import torch.nn as nn
import torch.nn.functional as F

from global_nii_max import nii_max, nii_max_f
class LogisticMixtureModel(nn.Module):
    def __init__(self, mean, log_sigma, mixture_weights, autoregression_coefficients=None, *args, **kwargs):
        super().__init__(*args, **kwargs)

        N,C,H,W = mean.shape

        self.mean = mean
        self.log_sigma = torch.clamp(log_sigma, min=-8.)

        self.mixture_weights = mixture_weights
        if autoregression_coefficients is not None:
            self.coeffs = torch.tanh(autoregression_coefficients)

    
    def forward(self, input, tau=0):
        N,C,H,W = input.shape

        half = float(0.5 + tau)/nii_max_f*2
        x=input
        centered_x = x - self.mean

        inv_sigma = torch.exp(-self.log_sigma)
        plus_in = inv_sigma * (centered_x + half)
        cdf_plus = torch.sigmoid(plus_in)
        min_in = inv_sigma * (centered_x - half)
        cdf_min = torch.sigmoid(min_in)
        log_one_minus_cdf_min = -F.softplus(min_in) # 255
        log_cdf_plus = plus_in - F.softplus(plus_in) # -255
        cdf_delta = cdf_plus - cdf_min
        
        log_probs = torch.where(x - half < -1.999, log_cdf_plus,
                               torch.where(x + half > 1.999, log_one_minus_cdf_min,
                                          torch.log(torch.clamp(cdf_delta, min=1e-9))))
        
        log_probs = log_probs + self.log_prob_from_logits(self.mixture_weights)
 
        return self.log_sum_exp(log_probs)

    
    def log_prob_from_logits(self, x): # normalize
        axis = 1
        m = torch.amax(x, axis, keepdim=True)
        return x - m - torch.log(torch.sum(torch.exp(x-m), axis, keepdim=True))
    
    
    def log_sum_exp(self, x):
        axis = 1
        m = torch.amax(x, axis)
        m2 = torch.amax(x, axis, keepdim=True)
        
        return m + torch.log(torch.sum(torch.exp(x-m2), axis))
        
        
        
