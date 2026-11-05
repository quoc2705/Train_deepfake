"""Kiến trúc FGI (Audio-Visual Deepfake Detection) — trích xuất nguyên bản từ `pbl6_2_train_A_13k_mouth_v5.ipynb`.

Giữ nguyên 100% tên lớp và tham số để nạp khớp hoàn toàn `state_dict` từ checkpoint.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: F401


def min_max_normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value + 0.00000001)


def conv3x3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=bias)


def conv1x3x3(in_planes, out_planes, stride=1, bias=False):
    return nn.Conv3d(in_planes, out_planes, kernel_size=(1, 3, 3),
                     stride=(1, stride, stride), padding=(0, 1, 1), bias=bias)


class BasicBlock3d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 track_running_stats=True, use_final_relu=True):
        super(BasicBlock3d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = conv3x3x3(inplanes, planes, stride, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, bias=bias)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.use_final_relu:
            out = self.relu(out)
        return out


class BasicBlock2d(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 track_running_stats=True, use_final_relu=True):
        super(BasicBlock2d, self).__init__()
        bias = False
        self.use_final_relu = use_final_relu
        self.conv1 = conv1x3x3(inplanes, planes, stride, bias=bias)
        self.bn1 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1x3x3(planes, planes, bias=bias)
        self.bn2 = nn.BatchNorm3d(planes, track_running_stats=track_running_stats)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        if self.use_final_relu:
            out = self.relu(out)
        return out


class ResNet2d3d_half(nn.Module):
    def __init__(self, block, layers, track_running_stats=True):
        super(ResNet2d3d_half, self).__init__()
        self.inplanes = 64
        self.track_running_stats = track_running_stats
        bias = False
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(1, 7, 7), stride=(1, 2, 2),
                               padding=(0, 3, 3), bias=bias)
        self.bn1 = nn.BatchNorm3d(64, track_running_stats=track_running_stats)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        if not isinstance(block, list):
            block = [block] * 4

        self.layer1 = self._make_layer(block[0], 64, layers[0])
        self.layer2 = self._make_layer(block[1], 128, layers[1], stride=2, is_final=True)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, is_final=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if block == BasicBlock2d:
                customized_stride = (1, stride, stride)
            else:
                customized_stride = stride
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=customized_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion,
                               track_running_stats=self.track_running_stats)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,
                            track_running_stats=self.track_running_stats))
        self.inplanes = planes * block.expansion
        if is_final:
            for i in range(1, blocks - 1):
                layers.append(block(self.inplanes, planes,
                                    track_running_stats=self.track_running_stats))
            layers.append(block(self.inplanes, planes,
                                track_running_stats=self.track_running_stats,
                                use_final_relu=False))
        else:
            for i in range(1, blocks):
                layers.append(block(self.inplanes, planes,
                                    track_running_stats=self.track_running_stats))
        return nn.Sequential(*layers)

    def forward(self, x, return_intermediate=False):
        x0 = self.conv1(x)
        x0 = self.bn1(x0)
        x0 = self.relu(x0)
        x0 = self.maxpool(x0)
        x1 = self.layer1(x0)
        x = self.layer2(x1)
        if return_intermediate:
            return x, x1
        else:
            return x


def resnet18_2d3d_half(**kwargs):
    model = ResNet2d3d_half([BasicBlock3d, BasicBlock3d], [2, 2], **kwargs)
    return model


def resnet34_2d3d_half(**kwargs):
    model = ResNet2d3d_half([BasicBlock3d, BasicBlock3d], [6, 3], **kwargs)
    return model


def select_resnet_half(network, track_running_stats=True):
    param = {'feature_size': 512}
    if network == 'resnet18':
        model = resnet18_2d3d_half(track_running_stats=track_running_stats)
        param['feature_size'] = 128
    elif network == 'resnet34':
        model = resnet34_2d3d_half(track_running_stats=track_running_stats)
        param['feature_size'] = 128
    else:
        raise IOError('model type is wrong')
    return model, param


class My_CNN_RawAud(nn.Module):
    def __init__(self):
        super(My_CNN_RawAud, self).__init__()
        self.netcnnaud_layer1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=80, stride=8),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
        )
        self.netcnnaud_layer2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Conv1d(128, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(192),
            nn.ReLU(inplace=True),
            nn.Conv1d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(192),
        )

    def forward(self, x):
        x1 = self.netcnnaud_layer1(x)
        x2 = self.netcnnaud_layer2(x1)
        return x2


class My_Network(nn.Module):
    def __init__(self, network='resnet18', with_attention=False,
                 residual_conn=False, spatial_size=28):
        super(My_Network, self).__init__()
        self.__nFeatures__ = 24
        self.__nChs__ = 32
        self.__midChs__ = 32
        self.aud2visspace = None
        self.with_attention = with_attention
        self.residual_conn = residual_conn
        self.spatial_size = spatial_size

        self.netcnnaud = My_CNN_RawAud()

        a_size = [192, 1497]
        v_size = [128, 15, spatial_size, spatial_size]
        stride = int(a_size[1] / v_size[1])
        kernel_size = int(a_size[1] - (v_size[1] - 1) * stride)
        self.netcnnaud_to_vis = nn.Sequential(
            nn.Conv1d(a_size[0], v_size[0] // 2, kernel_size=kernel_size, stride=stride),
            nn.BatchNorm1d(v_size[0] // 2),
            nn.ReLU(),
            nn.Conv1d(v_size[0] // 2, v_size[0], kernel_size=3, stride=1, padding=1)
        )

        self.netcnnlip, self.param = select_resnet_half(network, track_running_stats=True)

        map_size = v_size[2] * v_size[3]
        self.final_fc = nn.Sequential(
            nn.Linear(map_size, 1),
            nn.Sigmoid()
        )

        out_emb = v_size[0] // 4
        self.img_emb_layer = nn.Conv3d(v_size[0], out_emb, 1)
        self.aud_emb_layer = nn.Conv1d(v_size[0], out_emb, 1)

    def forward_aud(self, x):
        (B, C) = x.shape
        x = x.view(B, 1, C)
        x = self.netcnnaud(x)
        x = self.netcnnaud_to_vis(x)
        return x

    def forward_lip(self, x):
        (B, N, C, NF, H, W) = x.shape
        x = x.view(B * N, C, NF, H, W)
        x = self.netcnnlip(x, return_intermediate=False)
        if self.spatial_size != 28:
            x = nn.functional.adaptive_avg_pool3d(x, (15, self.spatial_size, self.spatial_size))
        return x

    def forward(self, vid_seq, aud_seq):
        vid_out = self.forward_lip(vid_seq)
        aud_out = self.forward_aud(aud_seq)

        aud_out = aud_out.view(aud_out.shape[0], aud_out.shape[1],
                               aud_out.shape[2], 1, 1)

        vid_aud_distance_ = torch.pow((vid_out - aud_out), 2)
        vid_aud_distance_ = vid_aud_distance_.view(
            vid_aud_distance_.shape[0],
            vid_aud_distance_.shape[1] * vid_aud_distance_.shape[2],
            vid_aud_distance_.shape[3] * vid_aud_distance_.shape[4]
        )
        vid_aud_distance_ = torch.sqrt(torch.sum(vid_aud_distance_, dim=1))

        if self.with_attention:
            img_emb = self.img_emb_layer(vid_out)
            aud_emb = self.aud_emb_layer(
                aud_out.view(aud_out.shape[0], aud_out.shape[1], aud_out.shape[2])
            )
            atts = []
            for i_emb, a_emb in zip(img_emb, aud_emb):
                atts.append(torch.tensordot(i_emb, a_emb, dims=([0, 1], [0, 1])))
            att = torch.stack(atts)
            att = att / (32 * 15)
            att = att.view(att.shape[0], -1)
            att = torch.nn.functional.softmax(att, dim=1)

            if self.residual_conn:
                vid_aud_distance_ = torch.mul(att, vid_aud_distance_) + vid_aud_distance_
            else:
                vid_aud_distance_ = torch.mul(att, vid_aud_distance_)
        else:
            att = None

        final_out = self.final_fc(vid_aud_distance_)
        return final_out, vid_aud_distance_, att

