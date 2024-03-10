from torch.autograd import Function
import torch


class BinaryQuantize(Function):
    @staticmethod
    def forward(ctx, input, k, t):
        ctx.save_for_backward(input, k, t)
        # input_numpy = input.cpu().numpy()
        # out = torch.where(input >= 0, 1, -1).cuda().half()
        # out = torch.from_numpy(input_numpy).cuda()
        out = torch.sign(input).cuda()
        return out

    @staticmethod
    def backward(ctx, grad_output):
        input, k, t = ctx.saved_tensors
        input_half = input.half()
        tanh_input = torch.tanh(input_half * t)
        # grad_input = k * t * (1 - torch.pow(torch.tanh(input * t), 2).cuda()) * grad_output
        # grad_input = (k * t * (1 - torch.pow(torch.tanh(input.half() * t), 2).cuda()) * grad_output.cuda().half()).cuda()
        grad_input = (k * t * (1 - torch.pow(tanh_input, 2)) * grad_output.half()).cuda()
        return grad_input, None, None
    
def Binarize(tensor,quant_mode='det'):
    if quant_mode=='det':
        return torch.where(tensor >= 0, 1, -1).cuda().half()
    else:
        return tensor.add_(1).div_(2).add_(torch.rand(tensor.size()).add(-0.5)).clamp_(0,1).round().mul_(2).add_(-1)
