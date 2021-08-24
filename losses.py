import torch
import torch.nn as nn
import torch.nn.functional as F

class get_classification_loss(nn.Module):
    def __init__(self, scale=0.0001):
        super().__init__()
        self.scale = scale
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pred, target, feature_matrix):
        loss = self.criterion(pred, target)  # 분류에선 log softmax를 모델에서 안취해줬다.

        bs = pred.size(0)

        id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)

        if pred.is_cuda:
            id64x64 = id64x64.cuda()

        diff_feature = id64x64 - torch.bmm(feature_matrix, feature_matrix.transpose(1, 2))

        regularization = self.scale * (torch.norm(diff_feature) + torch.norm(diff_feature)) / float(bs)
        total_loss = loss + regularization
        return total_loss


# Segmentation Loss
class get_segmentation_loss(nn.Module):
    def __init__(self, scale=0.001):
        super().__init__()
        self.scale = scale

    def forward(self, pred, target, feature_matrix):
        loss = F.nll_loss(pred, target)

        bs = pred.size(0)

        id64x64 = torch.eye(64, requires_grad=True).repeat(bs, 1, 1)

        if pred.is_cuda:
            id64x64 = id64x64.cuda()

        diff_feature = id64x64 - torch.bmm(feature_matrix, feature_matrix.transpose(1, 2))

        regularization = self.scale * (torch.norm(diff_feature) + torch.norm(diff_feature)) / float(bs)
        total_loss = loss + regularization
        return total_loss