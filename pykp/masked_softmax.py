import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedSoftmax(nn.Module):
    def __init__(self, dim):
        super(MaskedSoftmax, self).__init__()
        self.dim = dim

    def forward(self, logit, mask=None):
        if mask is None:
            dist = F.softmax(logit - torch.max(logit, dim=self.dim, keepdim=True)[0], dim=self.dim)
        else:
            dist_ = F.softmax(logit - torch.max(logit, dim=self.dim, keepdim=True)[0], dim=self.dim) * mask
            normalization_factor = dist_.sum(self.dim, keepdim=True)
            dist = dist_ / normalization_factor
        return dist

'''
def masked_softmax(logit, dim, mask=None):
    if mask is not None:
        dist_ = F.softmax( logit - torch.max(logit, dim=dim, keepdim=True)[0] , dim=dim) * mask
        # attn_dist_ = self.softmax(scores) * src_mask  # [batch_size, max_input_seq_len]
        normalization_factor = dist_.sum(dim, keepdim=True)
        dist = dist_ / normalization_factor
    else:
        dist = F.softmax( logit - torch.max(logit, dim=dim, keepdim=True)[0] , dim=dim)
    return dist
'''
