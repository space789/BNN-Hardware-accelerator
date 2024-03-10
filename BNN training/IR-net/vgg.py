'''VGG for CIFAR10. FC layers are removed.
'''
import numpy as np
import os
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from modules import ir_1w1a

# Choose which one you want to train dataset
CIFAR10 = False 
MNIST = True

# modify the save patten number
SAVE_PIC_NUM = 3

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'vgg_small', 'vgg_small_1w1a','full_bnn_1w1a'
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):
    def __init__(self, features, num_classes=10):
        super(VGG, self).__init__()
        self.features = features
        self.bn2 = nn.BatchNorm1d(512 * 1)
        self.nonlinear2 = nn.Hardtanh(inplace=True)
        self.classifier = nn.Linear(512, num_classes)
        self.bn3 = nn.BatchNorm1d(num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.bn2(x)
        x = self.nonlinear2(x)
        x = self.classifier(x)
        x = self.bn3(x)
        x = self.logsoftmax(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 1
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.Hardtanh(inplace=True)]
            else:
                layers += [conv2d, nn.Hardtanh(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'F': [128, 128, 'M', 512, 512, 'M'],
}


def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model


def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model


def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model


def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model

record_layer_outputs = []  # List to store the layer outputs

# TEST for fine-tuning BNN
class FULL_BNN_1W1A(nn.Module):
    def __init__(self, num_classes = 10):
        super(FULL_BNN_1W1A, self).__init__()
        self.padding = ir_1w1a.IRPadding(1, 1)
        self.conv0  = ir_1w1a.IRConv2dFIN(3, 128, kernel_size=3, bias=False)
        self.IRbn0  = ir_1w1a.IRBatchNorm2d(128)
        self.bn0    = nn.BatchNorm2d(128)
        self.clap   = ir_1w1a.IRCLAP()
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = ir_1w1a.IRConv2dFIN(128, 256, kernel_size=3, bias=False)
        self.IRbn1 = ir_1w1a.IRBatchNorm2d(256)
        self.bn1   = nn.BatchNorm2d(256)
        self.conv2 = ir_1w1a.IRConv2dFIN(256, 256, kernel_size=3, bias=False)
        self.IRbn2 = ir_1w1a.IRBatchNorm2d(256)
        self.bn2   = nn.BatchNorm2d(256)
        self.conv3 = ir_1w1a.IRConv2dFIN(256, 256, kernel_size=3, bias=False)
        self.IRbn3 = ir_1w1a.IRBatchNorm2d(256)
        self.bn3   = nn.BatchNorm2d(256)
        self.conv4 = ir_1w1a.IRConv2dFIN(256, 128, kernel_size=3, bias=False)
        self.IRbn4 = ir_1w1a.IRBatchNorm2d(128)
        self.bn4   = nn.BatchNorm2d(128)
        self.conv5 = ir_1w1a.IRConv2dFIN(128, 128, kernel_size=3, bias=False)
        self.IRbn5 = ir_1w1a.IRBatchNorm2d(128)
        self.bn5   = nn.BatchNorm2d(128)
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(128*4*4, num_classes)
        self.initialize_weights()

        self.layer_outputs = []  # List to store the layer outputs

    def save_output(self, module, input, output):
        self.layer_outputs.append(output.detach().cpu().numpy())

    def clear_record(self):
        record_layer_outputs = []  # Clear the list before each forward pass
    
    def check_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save_txt(self):
        for i in range(SAVE_PIC_NUM):
            self.check_path('./model_txt/pattern{0}'.format(i))
            save_output_3D('./model_txt/pattern{0}/input_pic.txt'.format(i), record_layer_outputs[0][i], True,'%d')
            save_output_3D('./model_txt/pattern{0}/output_padding{1}.txt'.format(i, 0), record_layer_outputs[1][i], True,'%d')
            save_output_3D('./model_txt/pattern{0}/output_conv{1}.txt'.format(i, 0), record_layer_outputs[2][i], False,'%d')
            save_output_3D('./model_txt/pattern{0}/output_bn{1}.txt'.format(i, 0), record_layer_outputs[3][i], False,'%d')
            save_output_3D('./model_txt/pattern{0}/output_clap{1}.txt'.format(i, 0), record_layer_outputs[4][i], True,'%d')
            save_output_3D('./model_txt/pattern{0}/output_padding{1}.txt'.format(i, 1), record_layer_outputs[5][i], True,'%d')
            save_output_3D('./model_txt/pattern{0}/output_conv{1}.txt'.format(i, 1), record_layer_outputs[6][i], False,'%d')
            save_output_3D('./model_txt/pattern{0}/output_bn{1}.txt'.format(i, 1), record_layer_outputs[7][i], False,'%d')
            save_output_3D('./model_txt/pattern{0}/output_clap{1}.txt'.format(i, 1), record_layer_outputs[8][i], True,'%d')
            save_output_3D('./model_txt/pattern{0}/output_pooling{1}.txt'.format(i, 1), record_layer_outputs[9][i], True,'%d')
            save_output_3D('./model_txt/pattern{0}/output_padding{1}.txt'.format(i, 2), record_layer_outputs[10][i], True,'%d')
            save_output_3D('./model_txt/pattern{0}/output_conv{1}.txt'.format(i, 2), record_layer_outputs[11][i], False,'%d')
            save_output_3D('./model_txt/pattern{0}/output_bn{1}.txt'.format(i, 2), record_layer_outputs[12][i], False,'%d')
            save_output_3D('./model_txt/pattern{0}/output_clap{1}.txt'.format(i, 2), record_layer_outputs[13][i], True,'%d')
            save_output_3D('./model_txt/pattern{0}/output_padding{1}.txt'.format(i, 3), record_layer_outputs[14][i], True,'%d')
            save_output_3D('./model_txt/pattern{0}/output_conv{1}.txt'.format(i, 3), record_layer_outputs[15][i], False,'%d')
            save_output_3D('./model_txt/pattern{0}/output_bn{1}.txt'.format(i, 3), record_layer_outputs[16][i], False,'%d')
            save_output_3D('./model_txt/pattern{0}/output_clap{1}.txt'.format(i, 3), record_layer_outputs[17][i], True,'%d')
            save_output_3D('./model_txt/pattern{0}/output_pooling{1}.txt'.format(i, 3), record_layer_outputs[18][i], True,'%d')
            save_output_3D('./model_txt/pattern{0}/output_padding{1}.txt'.format(i, 4), record_layer_outputs[19][i], True,'%d')
            save_output_3D('./model_txt/pattern{0}/output_conv{1}.txt'.format(i, 4), record_layer_outputs[20][i], False,'%d')
            save_output_3D('./model_txt/pattern{0}/output_bn{1}.txt'.format(i, 4), record_layer_outputs[21][i], False,'%d')
            save_output_3D('./model_txt/pattern{0}/output_clap{1}.txt'.format(i, 4), record_layer_outputs[22][i], True,'%d')
            save_output_3D('./model_txt/pattern{0}/output_padding{1}.txt'.format(i, 5), record_layer_outputs[23][i], True,'%d')
            save_output_3D('./model_txt/pattern{0}/output_conv{1}.txt'.format(i, 5), record_layer_outputs[24][i], False,'%d')
            save_output_3D('./model_txt/pattern{0}/output_bn{1}.txt'.format(i, 5), record_layer_outputs[25][i], False,'%d')
            save_output_3D('./model_txt/pattern{0}/output_clap{1}.txt'.format(i, 5), record_layer_outputs[26][i], True,'%d')
            save_output_3D('./model_txt/pattern{0}/output_pooling{1}.txt'.format(i, 5), record_layer_outputs[27][i], True,'%d')
            save_output_1D('./model_txt/pattern{0}/output_review.txt'.format(i), record_layer_outputs[28][i], True,'%f')
            save_output_1D('./model_txt/pattern{0}/output_fc.txt'.format(i), record_layer_outputs[29][i], False,'%f')
    
    def get_layer_outputs(self):
        return self.layer_outputs

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ir_1w1a.IRConv2dFIN):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self,x):
        self.clear_record()

        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.padding(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.conv0(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.IRbn0(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.nonlinear(x)
        x = self.clap(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.padding(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.conv1(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.IRbn1(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.nonlinear(x)
        x = self.clap(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.pooling(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.padding(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.conv2(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.IRbn2(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.nonlinear(x)
        x = self.clap(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.padding(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.conv3(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.IRbn3(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.nonlinear(x)
        x = self.clap(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.pooling(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.padding(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.conv4(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.IRbn4(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.nonlinear(x)
        x = self.clap(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.padding(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.conv5(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.IRbn5(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.nonlinear(x)
        x = self.clap(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.pooling(x)
        record_layer_outputs.append(x.detach().cpu().numpy())

        # x = self.nonlinear(self.IRbn0(self.conv0(self.padding0(x))))
        # x = self.nonlinear(self.pooling(self.IRbn1(self.conv1(self.padding1(x)))))
        # x = self.nonlinear(self.IRbn2(self.conv2(self.padding2(x))))
        # x = self.nonlinear(self.pooling(self.IRbn3(self.conv3(self.padding3(x)))))
        # x = self.nonlinear(self.IRbn4(self.conv4(self.padding4(x))))
        # x = self.nonlinear(self.pooling(self.IRbn5(self.conv5(self.padding5(x)))))

        # x = self.nonlinear(self.conv0(self.padding0(x)))
        # x = self.nonlinear(self.pooling(self.conv1(self.padding0(x))))
        # x = self.nonlinear(self.conv2(self.padding0(x)))
        # x = self.nonlinear(self.pooling(self.conv3(self.padding0(x))))
        # x = self.nonlinear(self.conv4(self.padding0(x)))
        # x = self.nonlinear(self.pooling(self.conv5(self.padding0(x))))
        
        x = self.clap(x)
        x = x.view(x.size(0), -1)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.fc(x)
        record_layer_outputs.append(x.detach().cpu().numpy())

        self.save_txt()
        return x

class FULL_BNN_MNIST(nn.Module):
    def __init__(self, num_classes = 10, bias=False, affine=False):
        super(FULL_BNN_MNIST, self).__init__()
        self.padding = ir_1w1a.IRPadding(1, 1)
        self.conv0  = ir_1w1a.IRConv2dFIN(1, 32, kernel_size=3, bias=bias)
        self.IRbn0  = ir_1w1a.IRBatchNorm2d(32)
        self.bn0    = nn.BatchNorm2d(32, affine=affine)
        self.clap   = ir_1w1a.IRCLAP()
        self.nonlinear = nn.Hardtanh(inplace=True)
        self.relu = nn.ReLU(inplace=True)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.conv1 = ir_1w1a.IRConv2dFIN(32, 64, kernel_size=3, bias=bias)
        self.IRbn1 = ir_1w1a.IRBatchNorm2d(64)
        self.bn1   = nn.BatchNorm2d(64, affine=affine)
        self.conv2 = ir_1w1a.IRConv2dFIN(64, 128, kernel_size=3, bias=bias)
        self.IRbn2 = ir_1w1a.IRBatchNorm2d(128)
        self.bn2   = nn.BatchNorm2d(128, affine=affine)
        self.fc = nn.Linear(128*4*4, num_classes)
        self.solfmax = nn.LogSoftmax()
        self.initialize_weights()

    def clear_record(self):
        record_layer_outputs = []  # Clear the list before each forward pass

    def check_path(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

    def save_txt(self):
        for i in range(SAVE_PIC_NUM):
            self.check_path('./model_txt/mnist_pattern{0}'.format(i))
            save_output_3D('./model_txt/mnist_pattern{0}/input_pic.txt'.format(i), record_layer_outputs[0][i], True,'%d')
            for j in range(3):
                save_output_3D('./model_txt/mnist_pattern{0}/output_padding{1}.txt'.format(i, j), record_layer_outputs[5*j+1][i], True,'%d')
                save_output_3D('./model_txt/mnist_pattern{0}/output_conv{1}.txt'.format(i, j), record_layer_outputs[5*j+2][i], False,'%d')
                save_output_3D('./model_txt/mnist_pattern{0}/output_bn{1}.txt'.format(i, j), record_layer_outputs[5*j+3][i], False,'%d')
                save_output_3D('./model_txt/mnist_pattern{0}/output_pooling{1}.txt'.format(i, j), record_layer_outputs[5*j+4][i], False,'%d')
                save_output_3D('./model_txt/mnist_pattern{0}/output_clap{1}.txt'.format(i, j), record_layer_outputs[5*j+5][i], True,'%d')
            save_output_1D('./model_txt/mnist_pattern{0}/output_view.txt'.format(i), record_layer_outputs[16][i], True,'%f')
            save_output_1D('./model_txt/mnist_pattern{0}/output_fc.txt'.format(i), record_layer_outputs[17][i], False,'%f')

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, ir_1w1a.IRConv2dFIN):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self,x):
        self.clear_record()

        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.padding(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.conv0(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.IRbn0(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.pooling(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.clap(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.padding(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.conv1(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.IRbn1(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.pooling(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.clap(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.padding(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.conv2(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.IRbn2(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.pooling(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.clap(x)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = x.view(x.size(0), -1)
        record_layer_outputs.append(x.detach().cpu().numpy())
        x = self.fc(x)
        record_layer_outputs.append(x.detach().cpu().numpy())

        self.save_txt()

        return x






def save_output_4D(save_pth,output,clap,format):
    N, C, H, W = output.shape
    with open(save_pth, 'w') as f:
        for n in range(N):
            for i in range(C):
                for j in range(H):
                    for k in range(W - 1):
                        if clap == True:
                            if int(output[n, i, j, k]) == -1:
                                f.write('0 ')
                            else:
                                f.write('1 ')
                        else:
                            f.write(str(int(output[n, i, j, k])) + ' ')
                    k += 1
                    if clap == True:
                        if int(output[n, i, j, k]) == -1:
                            f.write('0\n')
                        else:
                            f.write('1\n')
                    else:
                        f.write(str(int(output[n, i, j, k])) + '\n')
                f.write('\n')
            f.write('\n')
    f.close()

def save_output_3D(save_pth,output,clap,format):
    C, H, W = output.shape
    with open(save_pth, 'w') as f:
        for i in range(C):
            for j in range(H):
                for k in range(W - 1):
                    if clap == True:
                        if int(output[i, j, k]) == -1:
                            f.write('0 ')
                        else:
                            f.write('1 ')
                    else:
                        f.write(str(int(output[i, j, k])) + ' ')
                k += 1
                if clap == True:
                    if int(output[i, j, k]) == -1:
                        f.write('0\n')
                    else:
                        f.write('1\n')
                else:
                    f.write(str(int(output[i, j, k])) + '\n')
            f.write('\n')
    f.close()

def save_output_1D(save_pth,output,clap,format):
    output_flat = output.flatten()
    C = output_flat.shape[0]
    with open(save_pth, 'w') as f:
        for i in range(C):
            if clap == True:
                if int(output_flat[i]) == -1:
                    f.write('0\n')
                else:
                    f.write('1\n')
            else:
                if format == '%f':
                    f.write(str(output_flat[i]) + '\n')
                else:
                    f.write(str(int(output_flat[i])) + '\n')
    f.close()

class VGG_SMALL_1W1A(nn.Module):
    if CIFAR10:
        def __init__(self, num_classes=10, bias=True, affine=False):
            super(VGG_SMALL_1W1A, self).__init__()
            self.ratioIn = 1
            self.IRPadding_2 = ir_1w1a.IRPadding(2, 2)
            self.Pooling = nn.MaxPool2d(kernel_size=2, stride=2)
            self.nonlinear = nn.Hardtanh(inplace=True)
            self.conv0 = ir_1w1a.ShiftIRConv2d(3, int(64 * self.ratioIn), kernel_size=3, bias=bias)
            self.bn0 = nn.BatchNorm2d(int(64 * self.ratioIn), affine=affine)
            self.conv1 = ir_1w1a.ShiftIRConv2d(int(64 * self.ratioIn), int(128 * self.ratioIn), kernel_size=3, bias=bias)
            self.bn1 = nn.BatchNorm2d(int(128 * self.ratioIn), affine=affine)
            self.IRPadding_1 = ir_1w1a.IRPadding(1, 1)
            self.conv2 = ir_1w1a.ShiftIRConv2d(int(128 * self.ratioIn), int(128 * self.ratioIn), kernel_size=3, bias=bias)
            self.bn2 = nn.BatchNorm2d(int(128 * self.ratioIn), affine=affine)
            self.conv3 = ir_1w1a.ShiftIRConv2d(int(128 * self.ratioIn), int(256 * self.ratioIn), kernel_size=3, bias=bias)
            self.bn3 = nn.BatchNorm2d(int(256 * self.ratioIn), affine=affine)
            self.conv4 = ir_1w1a.ShiftIRConv2d(int(256 * self.ratioIn), int(512 * self.ratioIn), kernel_size=3, bias=bias)
            self.bn4 = nn.BatchNorm2d(int(512 * self.ratioIn), affine=affine)
            self.conv5 = ir_1w1a.ShiftIRConv2d(int(512 * self.ratioIn), 256, kernel_size=3, bias=bias)
            self.bn5 = nn.BatchNorm2d(256, affine=affine)
            self.BNfc_0 = ir_1w1a.BinarizeLinear(256 * 4 * 4, 4096, bias=bias)
            self.bn6 = nn.BatchNorm1d(4096, affine=affine)
            self.dropout = nn.Dropout(0.5)
            self.BNfc_1 = ir_1w1a.BinarizeLinear(4096, 4096, bias=bias)
            self.bn7 = nn.BatchNorm1d(4096, affine=affine)
            self.BNfc_2 = ir_1w1a.BinarizeLinear(4096, num_classes, bias=bias)
            self.bn8 = nn.BatchNorm1d(num_classes)
            self.solfmax = nn.LogSoftmax()
            self._initialize_weights()

    if MNIST:
        def __init__(self, num_classes=10, bias=True, affine=False):
            super(VGG_SMALL_1W1A, self).__init__()
            self.ratioIn = 1
            self.IRPadding_2 = ir_1w1a.IRPadding(2, 2)
            self.Pooling = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.clap   = ir_1w1a.IRCLAP()
            self.nonlinear = nn.Hardtanh(inplace=True)
            self.relu = nn.ReLU(inplace=True)
            self.conv0 = ir_1w1a.IRConv2d(1, int(32 * self.ratioIn), kernel_size=3, bias=bias)
            self.bn0 = nn.BatchNorm2d(int(32 * self.ratioIn), affine=affine)
            self.conv1 = ir_1w1a.IRConv2d(int(32 * self.ratioIn), int(64 * self.ratioIn), kernel_size=3, bias=bias)
            self.bn1 = nn.BatchNorm2d(int(64 * self.ratioIn), affine=affine)
            self.IRPadding_1 = ir_1w1a.IRPadding(1, 1)
            self.conv2 = ir_1w1a.IRConv2d(int(64* self.ratioIn), int(128 * self.ratioIn), kernel_size=3, bias=bias)
            self.bn2 = nn.BatchNorm2d(int(128 * self.ratioIn), affine=affine)
            # self.conv3 = ir_1w1a.IRConv2d(int(256 * self.ratioIn), int(256 * self.ratioIn), kernel_size=3, bias=bias)
            # self.bn3 = nn.BatchNorm2d(int(256 * self.ratioIn), affine=affine)
            # self.conv4 = ir_1w1a.IRConv2d(int(512 * self.ratioIn), int(512 * self.ratioIn), kernel_size=3, bias=bias)
            # self.bn4 = nn.BatchNorm2d(int(512 * self.ratioIn), affine=affine)
            # self.conv5 = ir_1w1a.IRConv2d(int(512 * self.ratioIn), 256, kernel_size=3, bias=bias)
            # self.bn5 = nn.BatchNorm2d(256, affine=affine)
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(128 * 4 * 4, num_classes, bias=bias)
            self.solfmax = nn.LogSoftmax()
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, ir_1w1a.IRConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                if m.affine is not False:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        if CIFAR10:
            x = self.nonlinear(self.bn0(self.conv0(self.IRPadding_1(x))))
            x = self.nonlinear(self.bn1(self.Pooling(self.conv1(self.IRPadding_1(x)))))
            x = self.nonlinear(self.bn2(self.conv2(self.IRPadding_1(x))))
            x = self.nonlinear(self.bn3(self.Pooling(self.conv3(self.IRPadding_1(x)))))
            x = self.nonlinear(self.bn4(self.conv4(self.IRPadding_1(x))))
            x = self.nonlinear(self.bn5(self.Pooling(self.conv5(self.IRPadding_1(x)))))
            x = x.view(x.size(0), -1)
            x = self.nonlinear(self.bn6(self.BNfc_0(x)))
            # x = self.dropout(x)
            x = self.nonlinear(self.bn7(self.BNfc_1(x)))
            # x = self.dropout(x)
            x = self.bn8(self.BNfc_2(x))
            x = self.solfmax(x)
            return x
        if MNIST:
            x = self.nonlinear(self.bn0(self.Pooling(self.conv0(self.IRPadding_1(x)))))
            x = self.nonlinear(self.bn1(self.Pooling(self.conv1(self.IRPadding_1(x)))))
            x = self.nonlinear(self.bn2(self.Pooling(self.conv2(self.IRPadding_1(x)))))
            x = self.clap(x)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.relu(self.fc(x))
            x = self.solfmax(x)
            return x


def vgg_small_1w1a(**kwargs):
    model = VGG_SMALL_1W1A(**kwargs)
    return model


def full_bnn_1w1a(**kwargs):
    model = FULL_BNN_1W1A(**kwargs)
    return model

def full_bnn_mnist(**kwargs):
    model = FULL_BNN_MNIST(**kwargs)
    return model