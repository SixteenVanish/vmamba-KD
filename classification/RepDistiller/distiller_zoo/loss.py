import torch
import torch.nn as nn



class L2LossFunction(nn.Module):
    def __init__(self):
        super(L2LossFunction, self).__init__()
        self.mse_loss = nn.MSELoss(reduction='mean')

    def forward(self, features_student, features_teacher):
        assert len(features_student) == 4, "The student features list must contain 4 elements."
        assert len(features_teacher) == 4, "The teacher features list must contain 4 elements."
        
        total_loss = 0
        for student_feature, teacher_feature in zip(features_student, features_teacher):
            total_loss += self.mse_loss(student_feature, teacher_feature)
        mean_loss = total_loss / 4  
        return mean_loss
        
class FeatureDimConverter(nn.Module):
    def __init__(self, config):
        super(FeatureDimConverter, self).__init__()        
        depths=config.MODEL.VSSM.DEPTHS
        dims=config.MODEL.VSSM.EMBED_DIM
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(len(depths))]
            dims = dims[1:]
            dims.append(dims[-1])
        depths_t=config.MODEL_T.VSSM.DEPTHS
        dims_t=config.MODEL_T.VSSM.EMBED_DIM
        if isinstance(dims_t, int):
            dims_t = [int(dims_t * 2 ** i_layer) for i_layer in range(len(depths_t))]
            dims_t = dims_t[1:]
            dims_t.append(dims_t[-1])
        
        self.convs = nn.ModuleList([
            # nn.Conv2d(dims[0], dims_t[0], kernel_size=1),
            # nn.Conv2d(dims[1], dims_t[1], kernel_size=1),
            # nn.Conv2d(dims[2], dims_t[2], kernel_size=1),
            # nn.Conv2d(dims[3], dims_t[3], kernel_size=1),
            nn.Conv2d(dims_t[0], dims[0], kernel_size=1),
            nn.Conv2d(dims_t[1], dims[1], kernel_size=1),
            nn.Conv2d(dims_t[2], dims[2], kernel_size=1),
            nn.Conv2d(dims_t[3], dims[3], kernel_size=1),
        ])

    def forward(self, list1, list2):
        assert len(list1) == 4 and len(list2) == 4, "Each input list must have 4 elements."
        
        # list1_transformed = [self.convs[i](list1[i]) for i in range(4)]
        # list2_transformed = list2

        return list1, [self.convs[i](list2[i]) for i in range(4)]
