###
# Author: Kai Li
# Date: 2021-06-09 16:43:09
# LastEditors: Please set LastEditors
# LastEditTime: 2024-01-24 00:00:52
###

import torch
from torch.nn.modules.loss import _Loss

def freq_MAE(output, target):
    loss = 0.
    eps = torch.finfo(torch.float32).eps
    all_win = [32, 64, 128, 256, 512, 1024, 2048]
    for win in all_win:
        est_spec = torch.stft(output.view(-1, output.shape[-1]), n_fft=win, hop_length=win//2, 
                            window=torch.hann_window(win).to(output.device).float(),
                            return_complex=True)
        target_spec = torch.stft(target.view(-1, target.shape[-1]), n_fft=win, hop_length=win//2, 
                                window=torch.hann_window(win).to(target.device).float(),
                                return_complex=True)
        
        loss = loss + (est_spec.abs() - target_spec.abs()).abs().mean() / (target_spec.abs().mean() + eps)
    
    return loss / len(all_win)

class MultiFrequencyDisLoss(_Loss):
    def __init__(self, eps=1e-8):
        super(MultiFrequencyDisLoss, self).__init__()

    def forward(self, target_outputs, est_outputs, return_stats=False):
        D_real = 0
        D_fake = 0
        for i in range(len(target_outputs)):
            D_real = D_real + (target_outputs[i] - 1).pow(2).mean() / len(target_outputs)
            D_fake = D_fake + (est_outputs[i]).pow(2).mean() / len(est_outputs)
        total = D_real + D_fake
        if return_stats:
            stats = {
                "d_real": sum(o.mean() for o in target_outputs) / len(target_outputs),
                "d_fake": sum(o.mean() for o in est_outputs) / len(est_outputs),
            }
            return total, stats
        return total
    
class MultiFrequencyGenLoss(_Loss):
    def __init__(self, eps=1e-8):
        super(MultiFrequencyGenLoss, self).__init__()
        self.eps = eps

    def forward(self, est_outputs, est_feature_maps, targets_feature_maps, output, ori_data,
                return_stats=False):
        G_fake = 0
        feature_matching = 0
        eps = self.eps

        for i in range(len(est_outputs)):
            G_fake = G_fake + (est_outputs[i] - 1).pow(2).mean() / len(est_outputs)
            for j in range(len(est_feature_maps[i])):
                feature_matching = feature_matching + (est_feature_maps[i][j] - targets_feature_maps[i][j].detach()).abs().mean() / (targets_feature_maps[i][j].detach().abs().mean() + eps)
        
        feature_matching = feature_matching / (len(est_outputs) * len(est_feature_maps[0]))
        freq_loss = freq_MAE(output, ori_data.unsqueeze(1))
        total_loss = freq_loss + G_fake + feature_matching

        if return_stats:
            stats = {"rec": freq_loss.detach(), "fm": feature_matching.detach(), "adv": G_fake.detach()}
            return total_loss, stats
        return total_loss
