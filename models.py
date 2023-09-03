import torch
import math
import torch.nn as nn
import torchvision
import clip
from pygcn.layers import GraphConvolution
from pygat.layers import GraphAttentionLayer
import torch.nn.functional as F

class Teacher(nn.Module):
    def __init__(self):
        super(Teacher, self).__init__()
        self.alexnet = torchvision.models.alexnet(pretrained=True)
        # self.alexnet = torchvision.models.resnet50(pretrained=True)
        # self.alexnet = torchvision.models.vgg19(pretrained=True)
        # self.alexnet.classifier = self.alexnet.classifier[:-1]
        # self.clip_image_encode, _ = clip.load("ViT-B/16", device='cuda:0') #ViT-L/14-#768 ViT-B/32#512

    def forward(self, x):
        with torch.no_grad():
        #     feat_I = self.clip_image_encode.encode_image(x)
        #     feat_I = feat_I.type(torch.float32)
            feat_I = self.alexnet(x)
        return feat_I


class ImgNet(nn.Module):
    def __init__(self, code_len):
        super(ImgNet, self).__init__()
        self.mobilev2 = torchvision.models.mobilenet_v2(pretrained=True)
        self.mobilev2.classifier = self.mobilev2.classifier[:-1]
        self.hash_layer = nn.Linear(1280, code_len)
        self.alpha = 1.0

    def forward(self, x):
        feat_I = self.mobilev2(x)
        mid = self.hash_layer(torch.relu(feat_I))
        code = torch.tanh(self.alpha * mid)
        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)

class TxtNet(nn.Module):
    def __init__(self, code_len, txt_feat_len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(txt_feat_len, code_len)
        self.alpha = 1.0

    def forward(self, x):
        hid = self.fc1(x)
        code = torch.tanh(self.alpha * hid)
        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GCNet_IMG(nn.Module):
    def __init__(self, bit, gamma, batch_size):
        super(GCNet_IMG, self).__init__()

        self.gc1 = GraphConvolution(1000, bit)
        self.alpha = 1.0
        # self.gamma = gamma
        # self.weight = nn.Parameter(torch.FloatTensor(batch_size, batch_size))
        # nn.init.kaiming_uniform_(self.weight)
        # nn.init.constant_(self.weight, 1e-6)

    def forward(self, x, adj):
        # adj = adj + self.gamma * self.weight
        feat_G_I = self.gc1(x, adj)
        code = torch.tanh(self.alpha * feat_G_I)
        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


class GCNet_TXT(nn.Module):
    def __init__(self, txt_feat_len, bit, gamma, batch_size):
        super(GCNet_TXT, self).__init__()

        self.gc1 = GraphConvolution(txt_feat_len, bit)
        self.alpha = 1.0
        # self.gamma = gamma
        # self.weight = nn.Parameter(torch.FloatTensor(batch_size, batch_size))
        # nn.init.kaiming_uniform_(self.weight)
        # nn.init.constant_(self.weight, 1e-6)

    def forward(self, x, adj):
        # adj = adj + self.gamma * self.weight
        feat_G_T = self.gc1(x, adj)
        code = torch.tanh(self.alpha * feat_G_T)
        return code

    def set_alpha(self, epoch):
        self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)


# class GATNet_IMG(nn.Module):
#     def __init__(self, code_len):
#         super(GATNet_IMG, self).__init__()
#         self.fc1 = nn.Linear(1000, code_len)
#         self.alpha = 1.0
#
#     def forward(self, x):
#         hid = self.fc1(x)
#         code = torch.tanh(self.alpha * hid)
#         return code
#
#     def set_alpha(self, epoch):
#         self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
#
# class GATNet_Txt(nn.Module):
#     def __init__(self, code_len, txt_feat_len):
#         super(GATNet_Txt, self).__init__()
#         self.fc1 = nn.Linear(txt_feat_len, code_len)
#         self.alpha = 1.0
#
#     def forward(self, x):
#         hid = self.fc1(x)
#         code = torch.tanh(self.alpha * hid)
#         return code
#
#     def set_alpha(self, epoch):
#         self.alpha = math.pow((1.0 * epoch + 1.0), 0.5)
class GATNet_IMG(nn.Module):
    def __init__(self, nfeat, bit, nhid=1024, alpha=0.2, nheads=4):
        """Dense version of GAT."""
        super(GATNet_IMG, self).__init__()
        self.alp = 1.0

        self.attentions = [GraphAttentionLayer(nfeat, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, bit, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.out_att(x, adj)
        code = torch.tanh(self.alp * x)
        return code

    def set_alpha(self, epoch):
        self.alp = math.pow((1.0 * epoch + 1.0), 0.5)

class GATNet_Txt(nn.Module):
    def __init__(self, txt_feat_len, code_len, nhid=1024, alpha=0.2, nheads=4):
        """Dense version of GAT."""
        super(GATNet_Txt, self).__init__()
        self.alp = 1.0

        self.attentions = [GraphAttentionLayer(txt_feat_len, nhid, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(nhid * nheads, code_len, alpha=alpha, concat=False)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = self.out_att(x, adj)
        code = torch.tanh(self.alp * x)
        return code

    def set_alpha(self, epoch):
        self.alp = math.pow((1.0 * epoch + 1.0), 0.5)
        
class Discriminator(nn.Module):

    def __init__(self, bits):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(bits, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.discriminator(x)