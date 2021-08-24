import torch
import torch.nn as nn
import torch.nn.functional as F


class TNet(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

        self.conv1 = nn.Conv1d(k, 64, 1)  # bs, 64, n
        self.conv2 = nn.Conv1d(64, 128, 1)  # bs, 128, n
        self.conv3 = nn.Conv1d(128, 1024, 1)  # bs, 1024, n

        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)

        self.regress = nn.Linear(256, k * k)

        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(1024)
        self.batchnorm4 = nn.BatchNorm1d(512)
        self.batchnorm5 = nn.BatchNorm1d(256)

    def forward(self, x):
        bs = x.size(0)
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))

        pool = nn.MaxPool1d(x.size(-1))(x)

        x = nn.Flatten(1)(pool)

        x = F.relu(self.batchnorm4(self.linear1(x)))
        x = F.relu(self.batchnorm5(self.linear2(x)))

        init = torch.eye(self.k, requires_grad=True).repeat(bs, 1, 1)
        if x.is_cuda:
            init = init.cuda()
        matrix = self.regress(x).view(-1, self.k, self.k) + init

        return matrix


class Trasnform_Classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = TNet(3)
        self.feature_transform = TNet(64)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)

        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(1024)

    def forward(self, x):  # x -> bs, 3, n input_matrix = bs, 3, 3
        bs = x.size(0)
        input_matrix = self.input_transform(x)  # bs, 3, 3

        xb = torch.transpose(torch.bmm(torch.transpose(x, 1, 2), input_matrix), 1, 2)  # x bs, c, n
        x = F.relu(self.batchnorm1(self.conv1(xb)))  # bs, 64, n
        feature_matrix = self.feature_transform(x)  # bs, 64, 64
        xb = torch.transpose(torch.bmm(torch.transpose(x, 1, 2), feature_matrix), 1, 2)

        x = F.relu(self.batchnorm2(self.conv2(xb)))  # bs, 128, n
        x = self.batchnorm3(self.conv3(x))  # bs, 1024, n

        pooled = torch.max(x, 2, keepdim=True)[0]
        output = pooled.view(bs, -1)

        return output, input_matrix, feature_matrix


class PointNet_Classification(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.transform = Trasnform_Classification()
        self.mlp1 = nn.Linear(1024, 512)
        self.mlp2 = nn.Linear(512, 256)
        self.cls = nn.Linear(256, k)  # k: Class Num

        self.batchnorm1 = nn.BatchNorm1d(512)
        self.batchnorm2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x, input_matrix, feature_matrix = self.transform(x)

        x = F.relu(self.batchnorm1(self.mlp1(x)))
        x = F.relu(self.batchnorm2(self.mlp2(x)))

        x = self.cls(x)
        return x, input_matrix, feature_matrix  # softmax 씌워줄 필요 없다. 나중 CrossEntropy에서 log softmax가 씌워짐


class Transform_Segmentation():
    def __init__(self):
        super().__init__()
        self.input_transform = TNet(3)
        self.feature_transform = TNet(128)

        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 128, 1)

        self.conv4 = nn.Conv1d(128, 512, 1)
        self.conv5 = nn.Conv1d(512, 2048, 1)

        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(128)
        self.batchnorm3 = nn.BatchNorm1d(128)
        self.batchnorm4 = nn.BatchNorm1d(512)
        self.batchnorm5 = nn.BatchNorm1d(2048)

    def forward(self, x, label):  # one hot encoding 된 label
        bs, c, N = x.size()
        input_matrix = self.input_transform(x)
        x = torch.transpose(torch.bmm(torch.transpose(x, 1, 2), input_matrix), 1, 2)

        x1 = F.relu(self.batchnorm1(self.conv1(x)))
        x2 = F.relu(self.batchnorm2(self.conv2(x1)))
        x3 = F.relu(self.batchnorm3(self.conv3(x2)))

        feature_matrix = self.feature_transform(3)

        x4 = torch.transpose(torch.bmm(torch.transpose(x3, 1, 2), feature_matrix), 1, 2)  # bs, 128, n

        x5 = F.relu(self.batchnorm4(self.conv4(x4)))
        x = self.batchnorm5(self.conv5(x5))

        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(bs, -1)  # bs, 2048
        x = torch.cat([x, label.squeeze(1)], 1)
        x6 = x.view(bs, -1, 1).repeat(1, 1, N)  # bs, -1, N

        out = torch.cat([x1, x2, x3, x4, x5, x6], 1)  # bs, -1, N
        return out, input_matrix, feature_matrix


class PointNet_Segmentation(nn.Module):
    def __init__(self, seg_part_num=50, label_num=40):
        super().__init__()

        self.transform = Transform_Segmentation()

        self.conv1 = nn.Conv1d(3024 + label_num, 256, 1)
        self.conv2 = nn.Conv1d(256, 256, 1)
        self.conv3 = nn.Conv1d(256, 128, 1)

        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(128)

        self.reg = nn.Conv1d(128, seg_part_num, 1)

    def forward(self, x):
        x, input_matrix, feature_matrix = self.transform(x)

        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))  # bs, c, n

        return F.log_softmax(torch.transpose(self.reg(x), 1, 2), -1)  # bs, n, seg_part_num
