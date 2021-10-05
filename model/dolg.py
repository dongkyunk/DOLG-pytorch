import timm
import torch
import torch.nn as nn
from model.gem_pool import GeM


class MultiAtrous(nn.Module):
    def __init__(self, in_channel, out_channel, dilation_rates=[3, 6, 9], size=56):
        super().__init__()
        self.dilated_convs = [
            nn.Conv2d(in_channel, int(out_channel/4), kernel_size=3, dilation=rate, padding=rate)
            for rate in dilation_rates
        ]
        self.gap_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channel, int(out_channel/4), kernel_size=1),
            nn.ReLU(),
            nn.Upsample(size=(size, size), mode='bilinear')
        )
        self.dilated_convs.append(self.gap_branch)
        self.dilated_convs = nn.ModuleList(self.dilated_convs)

    def forward(self, x):
        local_feat = []
        for dilated_conv in self.dilated_convs:
            local_feat.append(dilated_conv(x))
        local_feat = torch.cat(local_feat, dim=1)
        return local_feat


class DolgLocalBranch(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_channel=2048):
        super().__init__()
        self.multi_atrous = MultiAtrous(in_channel, hidden_channel)
        self.conv1x1_1 = nn.Conv2d(hidden_channel, out_channel, kernel_size=1)
        self.conv1x1_2 = nn.Conv2d(out_channel, out_channel, kernel_size=1, bias=False)
        self.conv1x1_3 = nn.Conv2d(out_channel, out_channel, kernel_size=1)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(out_channel)
        self.softplus = nn.Softplus()

    def forward(self, x):
        local_feat = self.multi_atrous(x)

        local_feat = self.conv1x1_1(local_feat)
        local_feat = self.relu(local_feat)
        local_feat = self.conv1x1_2(local_feat)
        local_feat = self.bn(local_feat)

        attention_map = self.relu(local_feat)
        attention_map = self.conv1x1_3(attention_map)
        attention_map = self.softplus(attention_map)

        local_feat = F.normalize(local_feat, p=2, dim=1)
        local_feat = local_feat * attention_map

        return local_feat


class OrthogonalFusion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, local_feat, global_feat):
        global_feat_norm = torch.norm(global_feat, p=2, dim=1)
        projection = torch.bmm(global_feat.unsqueeze(1), torch.flatten(
            local_feat, start_dim=2))
        projection = torch.bmm(global_feat.unsqueeze(
            2), projection).view(local_feat.size())
        projection = projection / \
            (global_feat_norm * global_feat_norm).view(-1, 1, 1, 1)
        orthogonal_comp = local_feat - projection
        global_feat = global_feat.unsqueeze(-1).unsqueeze(-1)
        return torch.cat([global_feat.expand(orthogonal_comp.size()), orthogonal_comp], dim=1)


class DolgNet(nn.Module):
    def __init__(self, hidden_channel=1024):
        super().__init__()
        self.cnn = timm.create_model(
            'resnet101',
            pretrained=True,
            features_only=True,
            in_chans=3,
            out_indices=(2, 3)
        )
        self.orthogonal_fusion = OrthogonalFusion()
        self.local_branch = DolgLocalBranch(512, hidden_channel)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gem_pool = GeM()
        self.fc_1 = nn.Linear(1024, hidden_channel)
        self.fc_2 = nn.Linear(int(2*hidden_channel), 512)

    def forward(self, x):
        output = self.cnn(x)

        local_feat = self.local_branch(output[0])  # ,hidden_channel,16,16
        global_feat = self.fc_1(self.gem_pool(output[1]).squeeze())  # ,1024
        feat = self.orthogonal_fusion(local_feat, global_feat)
        feat = self.gap(feat).squeeze()
        feat = self.fc_2(feat)

        return feat
