import torch.nn as nn
import torch.nn.functional as F
from . import binaryfunction
import torch
import math


class IRConv2d(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10], dtype=torch.float16).cuda()
        self.t = torch.tensor([0.1], dtype=torch.float16).cuda()
    
    def forward(self, input):
        w = self.weight
        a = input
        bw = w - w.view(w.size(0), -1).mean(-1).view(w.size(0), 1, 1, 1)
        bw = bw / bw.view(bw.size(0), -1).std(-1).view(bw.size(0), 1, 1, 1)
        sw = torch.pow(torch.tensor(2 * bw.size(0), dtype=torch.float16), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)) / math.log(2)).round()).view(bw.size(0), 1, 1, 1).detach()
        # bw = w - w.view(w.size(0), -1).cuda().mean(-1).view(w.size(0), 1, 1, 1)
        # bw = bw / bw.view(bw.size(0), -1).cuda().std(-1).view(bw.size(0), 1, 1, 1)
        # sw = torch.pow(torch.tensor(2*bw.size(0)).cuda().half(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)).cuda() / math.log(2)).cuda().round().half()).cuda().view(bw.size(0), 1, 1, 1).detach()
        bw = binaryfunction.BinaryQuantize().apply(bw, self.k, self.t)
        ba = binaryfunction.BinaryQuantize().apply(a, self.k, self.t)
        # bw = (bw * sw).cuda()
        bw = bw * sw
        # bw = binaryfunction.BinaryQuantize().apply(bw, self.k, self.t)
        # bw = bw.half() * sw.half()
        # torch.backends.cudnn.benchmark = False
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output
    
class IRConv2dFIN(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(IRConv2dFIN, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.k = torch.tensor([10]).half().cuda()
        self.t = torch.tensor([0.1]).half().cuda()
    
    def forward(self, input):
        w = self.weight
        a = input
        bw = binaryfunction.BinaryQuantize().apply(w, self.k, self.t)
        ba = binaryfunction.BinaryQuantize().apply(a, self.k, self.t)
        output = F.conv2d(ba, bw, self.bias,
                          self.stride, self.padding,
                          self.dilation, self.groups)
        return output


# class IRPadding(nn.Module):
#     def __init__(self, pad_h, pad_w):
#         super(IRPadding, self).__init__()
#         self.pad_h = pad_h
#         self.pad_w = pad_w
#         self.register_forward_hook(self.hook_fn)  # 添加 register_forward_hook

#     def hook_fn(self, module, input, output):
#         # 在 forward 完成後觸發的函式
#         # print("Forward hook called.")
#         return

#     def forward(self, x):
#         pad_h, pad_w = self.pad_h, self.pad_w
#         padded_image = torch.zeros((x.shape[0], x.shape[1], x.shape[2] + 2 * pad_h, x.shape[3] + 2 * pad_w), device=x.device, dtype=x.dtype)
#         padded_image[:, :, pad_h:-pad_h, pad_w:-pad_w] = x

#         for i in range(0, padded_image.shape[2]):
#             for j in range(0, padded_image.shape[3]):
#                 if i < pad_h or i >= padded_image.shape[2] - pad_h or j < pad_w or j >= padded_image.shape[3] - pad_w:
#                     if i % 2 == j % 2:
#                         padded_image[:,:,i, j] = -1
#                     else:
#                         padded_image[:,:,i, j] = 1
#                 else:
#                     padded_image[:,:,i, j] = x[:,:,i - pad_h, j - pad_w]

#         return padded_image


class IRPadding(nn.Module):
    def __init__(self, pad_h, pad_w):
        super(IRPadding, self).__init__()
        self.pad_h = pad_h
        self.pad_w = pad_w

    def forward(self, x):
        pad_h, pad_w = self.pad_h, self.pad_w
        batch_size, num_channels, height, width = x.shape
        padded_height = height + 2 * pad_h
        padded_width = width + 2 * pad_w

        device = x.device

        padded_image = torch.zeros((batch_size, num_channels, padded_height, padded_width), device=device, dtype=x.dtype)
        padded_image[:, :, pad_h:padded_height-pad_h, pad_w:padded_width-pad_w].copy_(x)

        mask = torch.zeros((batch_size, num_channels, padded_height, padded_width), device=device, dtype=x.dtype)

        mask[:, :, :pad_h, :] = -1
        mask[:, :, padded_height-pad_h:, :] = -1
        mask[:, :, :, :pad_w] = -1
        mask[:, :, :, padded_width-pad_w:] = -1

        mask[:, :, ::2, ::2] = mask[:, :, 1::2, 1::2] = 1

        mask[:, :, pad_h:padded_height-pad_h, pad_w:padded_width-pad_w] = 0

        padded_image = torch.where(mask != 0, mask, padded_image)

        return padded_image



class IRCLAP(nn.Module):
    def __init__(self):
        super(IRCLAP, self).__init__()

    def __call__(self, x):
        x = self.forward(x)
        return x
    
    def forward(self, x):
        out = torch.where(x >= 0, 1, -1).cuda().half()
        return out

class IRBatchNorm2d(nn.Module):
    def __init__(self, num_channels):
        super(IRBatchNorm2d, self).__init__()
        self.num_channels = num_channels

        # Define the learnable offset matrix
        self.offset = nn.Parameter(torch.Tensor(num_channels))

        # Initialize the offset matrix with random values
        # nn.init.normal_(self.offset)

        self.register_forward_hook(self.hook_fn)  # 添加 register_forward_hook

    def hook_fn(self, module, input, output):
        # 在 forward 完成後觸發的函式
        # print("Forward hook called.")
        return

    def forward(self, x):
        # Expand the offset matrix to match the input dimensions
        expanded_offset = self.offset.view(1, self.num_channels, 1, 1)

        # Add the expanded offset to the input value matrix
        output = x + expanded_offset

        return output


# class BinarizeLinear(nn.Linear):

#     def __init__(self, *kargs, **kwargs):
#         super(BinarizeLinear, self).__init__(*kargs, **kwargs)

#     def forward(self, input):

#         if input.size(1) != 784:
#             input.data=binaryfunction.Binarize(input.data)
#         if not hasattr(self.weight,'org'):
#             self.weight.org=self.weight.data.clone()
#         self.weight.data=binaryfunction.Binarize(self.weight.org)
#         out = nn.functional.linear(input, self.weight)
#         if not self.bias is None:
#             self.bias.org=self.bias.data.clone()
#             out += self.bias.view(1, -1).expand_as(out)

#         return out

import torch
import torch.nn as nn

class BinarizeLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super(BinarizeLinear, self).__init__(*args, **kwargs)

    def forward(self, input):
        if input.size(1) != 784:
            input.data = binaryfunction.Binarize(input.data)
        if not hasattr(self.weight, 'org'):
            self.weight.org = self.weight.data.clone()
        self.weight.data = binaryfunction.Binarize(self.weight.org)

        if torch.cuda.is_available():
            input = input.to('cuda')
            self.weight.org = self.weight.org.to('cuda')
            self.weight.data = self.weight.data.to('cuda')

        out = nn.functional.linear(input, self.weight)

        if not self.bias is None:
            if not hasattr(self.bias, 'org'):
                self.bias.org = self.bias.data.clone()
            self.bias.org = self.bias.org.to('cuda')
            self.bias = self.bias.to('cuda')
            out += self.bias.view(1, -1).expand_as(out)

        return out


class ShiftIRConv2d(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=0,bias=False):
        super(ShiftIRConv2d, self).__init__()
        self.shift1 = nn.Parameter(torch.zeros(1,in_ch,1,1), requires_grad=True)
        self.shift2 = nn.Parameter(torch.zeros(1, in_ch, 1, 1), requires_grad=True)
        self.conv = IRConv2d(in_ch,out_ch,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias)

    def forward(self,x):
        x1 = x + self.shift1.expand_as(x)
        x2 = x + self.shift2.expand_as(x)

        out1 = self.conv(x1)
        out2 = self.conv(x2)
        out = out1 + out2
        return out
