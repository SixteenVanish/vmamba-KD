import torch
import torch.nn as nn
import math

class HintConv(nn.Module):
    def __init__(self, D1, D2, separate=False, K=4):
        super(HintConv, self).__init__()
        self.K = K
        self.separate = separate
        
        if separate:
            self.conv_layers = nn.ModuleList([nn.Conv1d(D1, D2, kernel_size=1) for _ in range(K)])
        else:
            self.conv_layer = nn.Conv1d(D1, D2, kernel_size=1)

    def forward(self, x):   # x: [B, K*D1, L]
        if x.dim() == 4:
            x = x.squeeze(2)
            
        B, KD1, L = x.shape
        D1 = int(KD1 // self.K) 
        if self.separate:
            split_tensors = torch.split(x, D1, dim=1)
            conv_outputs = [conv_layer(tensor) for conv_layer, tensor in zip(self.conv_layers, split_tensors)]
            output = torch.cat(conv_outputs, dim=1) # [B, K*D2, L]
        else:
            x_reshaped = x.view(B * self.K, D1, L)
            conv_output = self.conv_layer(x_reshaped)
            output = conv_output.view(B, -1, L) # [B, K*D2, L]
        
        return output
    
# class HintConv(nn.Module):
#     def __init__(self, D1, D2, separate=False, K=4):
#         super(HintConv, self).__init__()
#         self.K = K
#         self.separate = separate
        
#         if separate:
#             self.conv_layers = nn.ModuleList([nn.Conv2d(D1, D2, kernel_size=1) for _ in range(K)])
#         else:
#             self.conv_layer = nn.Conv2d(D1, D2, kernel_size=1)

#     def forward(self, x):   # x: [B, K*D1, L]
#         if x.dim() == 4:
#             x = x.squeeze(2)
            
#         B, KD1, L = x.shape
#         D1 = int(KD1 // self.K) 
#         H = int(math.sqrt(L))
#         if self.separate:
#             x = x.view(B, KD1, H, H)    # [B, K*D1, H, W]
#             split_tensors = torch.split(x, D1, dim=1)
#             conv_outputs = [conv_layer(tensor) for conv_layer, tensor in zip(self.conv_layers, split_tensors)]  # 4个[B, D2, H, W]
#             output = torch.cat(conv_outputs, dim=1).view(B, -1, L) # 4个[B, D2, H, W] -> [B, 4*D2, H, W] -> [B, 4*D2, L]
#         else:
#             x_reshaped = x.view(B*self.K, D1, H, H) # [B*K, D1, H, W]
#             conv_output = self.conv_layer(x_reshaped)   # [B*K, D2, H, W]
#             output = conv_output.view(B, -1, L) # [B, K*D2, L]
#         return output