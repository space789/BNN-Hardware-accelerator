import argparse
import os
import sys
from pickle import FALSE, TRUE
import time
import math
import random
from turtle import Turtle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import logging

import torch, gc
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.utils
import torchvision.transforms as transforms
import torchvision.datasets as datasets

# import tensorflow as tf

# from torch.utils.tensorboard import SummaryWriter
import vgg
import util
from torchinfo import summary
import warnings

warnings.filterwarnings("ignore")


# Choose which one you want to train dataset
CIFAR10 = False
MNIST = True


model_names = sorted(name for name in vgg.__dict__
                     if name.islower() and not name.startswith("__")
                     and name.startswith("vgg")
                     and callable(vgg.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch Cifar10 and MNIST Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg_small_1w1a', # vgg_small_1w1a TODO: change to full_bnn_1w1a , full_bnn_mnist
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: vgg_small_1w1a)')
parser.add_argument('--archfintune', '-af', metavar='ARCHFINTUNE', default='full_bnn_mnist',
                    choices=model_names,
                    help='model architecture: ' + ' | '.join(model_names) +
                    ' (default: full_bnn_1w1a)')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=400, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.007, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--optimizer', default='sgd', type=str,
                    help='which optimizer to use')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--model', default='', type=str, metavar='PATH',
                    help='path to best model (default: none)')
# parser.add_argument('-a', '--adjustment', dest='adjustment', action='store_true',
#                     help='adjustment model weight')
parser.add_argument('-e', '--evaluate', dest='evaluate',action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-d', '--draw', dest='draw', action='store_true',
                    help='draw the input image')
parser.add_argument('-f', '--fine-tuning', dest='fine_tuning', action='store_true',
                    help='fine-tuning the model')
parser.add_argument('-t', '--test', dest='test', action='store_true',
                    help='test the model on real image')
parser.add_argument('-s', '--save_parameter', dest='save_parameter', action='store_true',
                    help='save the model parameter into txt file')
parser.add_argument('--pre_process', dest='pre_process', action='store_true',
                    help='pre_process the input image')
parser.add_argument('--half', dest='half', action='store_true',
                    help='use half-precision(16-bit) ')
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--save-txt-dir', dest='save_txt_dir',
                    help='The directory used to save the txt trained models',
                    default='./model_txt', type=str)
parser.add_argument('--save-every', dest='save_every',
                    help='Saves checkpoints at every specified number of epochs',
                    type=int, default=20)
parser.add_argument('--warm_up', default=False, type=bool,
                    help='use the warm up training strategy')


best_prec1 = 0

# log_save_name = 'IR-Net_vggsmall'  # IE-Net_resnet18, IE_Net_vggsmall
save_name = 'IR-Net_vggsmall'  # IE-Net_resnet18, IE_Net_vggsmall
save_check_name = 'IR-Net_vggsmall_checkpoint'

if not os.path.exists('log'):
    os.mkdir('log')

log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format)
fh = logging.FileHandler(os.path.join('log/{}.txt'.format(parser.parse_args().save_dir)))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# writer = SummaryWriter('./path/to/log')

activation = {}
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    global args, best_prec1
    args = parser.parse_args()

    is_best = 0.0

    start_t = time.time()
    # Check the save_dir exists or not
    os.makedirs(args.save_dir, exist_ok=True)

    
    model = vgg.__dict__[args.arch]().to(device)
    model = torch.nn.DataParallel(model)
    model.to(device).cuda()

    # model = torch.nn.DataParallel(vgg.__dict__[args.arch]()).cuda()
    model_fintune = torch.nn.DataParallel(vgg.__dict__[args.archfintune]()).cuda()
    model_fintune.to(device).cuda()
    # print(model_fintune)
    model_dict = model.state_dict()
    

    # logging.info(summary(model, input_size=(1, 3, 32, 32)))
    logging.info(args)
    
    # show use what device
    logging.info("Training use device: {}".format(device))

    optimizer = torch.optim.SGD(
        [{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logging.info("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.evaluate, checkpoint['epoch']))
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    # --------------------best model path exist--------------------------------
    elif args.model:
        if os.path.isfile(args.model):
            logging.info("=> loading checkpoint '{}'".format(args.model))
            checkpoint = torch.load(args.model)
            best_prec1 = checkpoint['best_prec1']
            for model_name, model_value in model_dict.items():
                if model_name in checkpoint['state_dict']:
                    # print(model_name)
                    model_value.copy_(checkpoint['state_dict'][model_name])
            # model.load_state_dict(checkpoint['state_dict'])
            logging.info("=> loaded best model prec* : '{:.3f}'"
                  .format(best_prec1))
        else:
            logging.info("=> no best model found at '{}'".format(args.model))

    else :
        logging.info("model: ")
        logging.info(model)
    
    # cudnn.benchmark = True

    if CIFAR10:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])

        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./cifar10_data', train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(root='./cifar10_data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=True)

    if MNIST:
        normalize = transforms.Normalize((0.5,), (0.5,))

        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='./mnist_data', train=True, transform=transforms.Compose([
                transforms.RandomCrop(28, 4),
                transforms.RandomRotation(20),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)

        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(root='./mnist_data', train=False, transform=transforms.Compose([
                # transforms.Pad(padding = 4)
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=True)

    if args.pre_process:
        train_loader = pre_process_img(train_loader)
        val_loader = pre_process_img(val_loader)

    # writer.add_image('four_fashion_cifar10_images', img_grid)

    """
    writer.add_graph(model, images)
    writer.close()
    """

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()

    if args.half:
        model.half()
        model_fintune.half()
        criterion.half()

    if args.resume:
        if os.path.isfile(args.resume):
            pass
        else:
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD(
                    [{'params': model.parameters(), 'initial_lr': args.lr}], lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
            else:
                optimizer = torch.optim.Adam(
                    [{'params': model.parameters, 'initial_lr': args.lr, 'weight_decay': args.weight_decay}], lr=args.lr)

    """
    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
    """

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs-4*args.warm_up, eta_min=0, last_epoch=-1)

    T_min, T_max = 1e-1, 1e1

    def Log_UP(K_min, K_max, epoch):
        Kmin, Kmax = math.log(
            K_min) / math.log(10), math.log(K_max) / math.log(10)
        return torch.tensor([math.pow(10, Kmin + (Kmax - Kmin) / args.epochs * epoch)]).float().cuda()


    if args.fine_tuning:
        fintune_test(model, model_fintune)
        # fintune(model)
        validate(val_loader, model, criterion)
        # best_prec1 = validate(val_loader, model_fintune, criterion)
        # save_bestpoint({
        #         'state_dict': model_fintune.state_dict(),
        #         'best_prec1': best_prec1,
        #     }, best_prec1, model, filename='./mnist_train_model_v5_1/best_model.th')
        # best_prec1 = 0
        return

    if args.save_parameter:
        save_model_parameter_txt(model, val_loader, criterion)
        return

    if args.evaluate:
        validate(val_loader, model, criterion)
        # save_model_parameter_txt(model, val_loader, criterion)
        return

    if args.test:
        real_predict_img(model)
        return

    """
    conv_modules = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print(module)
            conv_modules.append(module)
    logging.info(args)
    """

    # * record names of conv_modules
    try:
        for epoch in range(args.start_epoch, args.epochs):

            t = Log_UP(T_min, T_max, epoch)
            if (t < 1):
                k = 1 / t
            else:
                k = torch.tensor([1]).float().cuda()

            kt(model, k, t)

            # *warm up
            if args.warm_up and epoch < 5:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = args.lr * (epoch+1) / 5

            # train for one epoch
            logging.info('current lr {:.5e}'.format(
                optimizer.param_groups[0]['lr']))

            train(train_loader, model, criterion, optimizer, epoch)

            if epoch >= 4 * args.warm_up:
                lr_scheduler.step()
            # evaluate on validation set
            prec1 = validate(val_loader, model, criterion)
            
            # writer.add_scalar('prec1', prec1, epoch)

            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)

            if epoch > 0 and epoch % args.save_every == 0:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                    'optimizer_state_dict': optimizer.state_dict(),
                }, is_best, filename=os.path.join(args.save_dir, '{}.th'.format(save_check_name)))

            if is_best:
                save_bestpoint({
                    'state_dict': model.state_dict(),
                    'best_prec1': best_prec1,
                }, best_prec1, model, filename=os.path.join(args.save_dir, '{}.th'.format(save_name)))
        training_time = (time.time() - start_t) / 3600
        logging.info('total training time = {} hours'.format(training_time))

    except KeyboardInterrupt:
        print('KeyboardInterrupt\n')
        print('save model ...\n')
        save_checkpoint({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'optimizer_state_dict': optimizer.state_dict(),
        }, is_best, filename=os.path.join(args.save_dir, '{}.th'.format(save_check_name)))
        training_time = (time.time() - start_t) / 3600
        logging.info('total training time = {} hours'.format(training_time))
        print('\n\rquit')


def train(train_loader, model, criterion, optimizer, epoch):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        target = target.cuda()
        input_var = torch.autograd.Variable(input).cuda()
        target_var = torch.autograd.Variable(target).cuda()
        
        # print(input_var.shape)
        # vgg.save_output_4D('./model_txt/inputvar_look.txt', input_var, False, '%d')

        # convert input to 1 or -1
        input_var = torch.where(input_var >= 0, 1, -1).cuda()
        # input_var_numpy = input_var.cpu().numpy()
        # input_var_numpy = np.where(input_var_numpy >= 0.5, 1, -1)
        # # print(input_var_numpy)
        # input_var = torch.from_numpy(input_var_numpy).cuda().half()
        # print(type(input_var))
        # print(input_var)
        # print(target_var)
        # print(input_var[:,None].shape)  
        
        if args.half:
            input_var = input_var.cuda().half()
        
        # measure use gpu
        input_var = input_var.to(device)
        target_var = target_var.to(device)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                             epoch, i, len(train_loader), batch_time=batch_time,
                             data_time=data_time, loss=losses, top1=top1))
    logging.info('Train: Epoch time {batch_time.sum:.3f}\t'
                 'prec@1 {top1.avg:.3f}\t'
                .format(batch_time=batch_time, top1=top1))

def validate(val_loader, model, criterion):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    target_array = {}
    input_array = {}
    output_array = {}
    input_array_draw = {}

    end = time.time()
    for i, (input, target) in enumerate(val_loader):
        target = target.cuda()
        input_var = input.cuda()
        target_var = target.cuda()
        
        input_var_real = input_var

        # convert input to 1 and -1
        input_var = torch.where(input_var >= 0, 1, -1).cuda()
        input_var_save = torch.where(input_var >= 0.5, 1, 0).cuda()
        input_var_draw = torch.where(input_var >= 0.5 , 0.99, 0).cuda()
    
        if args.half:
            input_var = input_var.cuda().half()
        
        target_array[i] = target.cpu().numpy()
        input_array[i]  = input_var_save.cpu().numpy()
        input_array_draw[i] = input_var_draw.cpu().numpy()
        # input_array_draw[i] = input_var_real.cpu().numpy()

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        _, pred = output.topk(1, 1, True, True)
        output_array[i] = pred.cpu().numpy()

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            logging.info('Test: [{0}/{1}]\t'
                         'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                         'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                         'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                             i, len(val_loader), batch_time=batch_time, loss=losses,
                             top1=top1))


    logging.info(' * Prec@1 {top1.avg:.3f}'
          .format(top1=top1))

    # if args.evaluate:
    #     save_goldenvalue(input_array, output_array, target_array, val_loader)
    if args.draw:
        draw_picture(input_array_draw, output_array, target_array)
    
    return top1.avg


def save_goldenvalue(inputvar, outputvar, target, val_loader):
    print('==> Saving golden value ...')
    if not os.path.exists("./model_txt/outputvar/"):
        os.makedirs("./model_txt/outputvar/")
    if not os.path.exists("./model_txt/inputvar/"):
        os.makedirs("./model_txt/inputvar/")
    if not os.path.exists("./model_txt/targetvar/"):
        os.makedirs("./model_txt/targetvar/")
    # if not os.path.exists("./model_txt/inputvar_float/"):
    #     os.makedirs("./model_txt/inputvar_float/")
    """
    if not os.path.exists("./model_txt/test/"):
        os.makedirs("./model_txt/test/")
    parm = {}
    parm['inputvar'] = inputvar.numpy()
    parm['outputvar'] = outputvar.numpy()
    parm['target'] = target.numpy()
    """
    # print(np.shape(target))

    #np.savetxt('./model_txt/target_1.txt', target[1], fmt='%d')
    for i in range(len(val_loader)):
        np.savetxt(
            './model_txt/inputvar/inputvar_{0}.txt'.format(i), util.im2col(inputvar[i], 32, 32).reshape((12288,32)), fmt='%d') # 12288 = 128*32*3 (batch_size * H * kernel_size)
        np.savetxt(
            './model_txt/outputvar/outputvar_{0}.txt'.format(i), outputvar[i], fmt='%d')
        np.savetxt(
            './model_txt/targetvar/targetvar_{0}.txt'.format(i), target[i], fmt='%d')
   # np.savetxt('./model_txt/target.txt', parm['target'], fmt='%.4f')


def draw_picture(inputvar, outputvar, target):
    print('==> Drawing picture ...')
    label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    if not os.path.exists("./model_txt/picture/"):
        os.makedirs("./model_txt/picture/")
    if not os.path.exists("./model_txt/test_picture/"):
        os.makedirs("./model_txt/test_picture/")
    if not os.path.exists("./model_txt/real_picture/"):
        os.makedirs("./model_txt/real_picture/")
    inputvar[0] = inputvar[0].transpose(0, 2, 3, 1)
    # inputvar[0] = inputvar[0] * 255
    # define figure
    for i in range(len(inputvar[0])//16):
        fig=plt.figure(figsize=(10, 10))
        plt.text(x=0.5, y=0.94, s='Prediction', fontsize=22, ha="center", transform=fig.transFigure)
        for j in range(16):
            fig.add_subplot(4, 4, j+1)
            plt.imshow(inputvar[0][j+i*16])
            plt.xticks([])
            plt.yticks([])
            plt.title("{}({})".format(label_name[int(outputvar[0][j+i*16])], label_name[target[0][j+i*16]]))
        # plt.savefig('./model_txt/real_picture/input_{0}.png'.format(i))
        plt.savefig('./model_txt/picture/input_{0}.png'.format(i))

        
def reshape_output(output):
    outputs = np.expand_dims(output, axis=0)
    N, C, H, W = outputs.shape
    output_reshape = outputs.reshape(-1,W)
    return output_reshape

def reshape_parm(parm):
    N, C, H, W = parm.shape
    parm_reshape = parm.reshape(-1,W)
    return parm_reshape

def convert_to_hex(num):
    # Convert to integer
    num_int = int(num)
    
    # Convert to 2's complement for negative numbers
    if num_int < 0:
        num_int = (1 << 16) + num_int
    
    # Convert to 16-bit hexadecimal string
    hex_str = format(num_int, '04x')
    
    # Remove leading zeros for positive numbers
    if num_int >= 0:
        hex_str = hex_str.lstrip('0')
    
    return hex_str

def save_model_parameter_txt(model, val_loader, criterion_ce):
    print('==> Saving weight txt ...')
    if not os.path.exists("./model_txt/mnist_model_weight/"):
        os.makedirs("./model_txt/mnist_model_weight/")
    parm = {}
    for model_name, model_value in model.state_dict().items():
        parm[model_name] = model_value.detach().cpu().numpy()
        # print(model_name)
    conv_layers = ['conv0', 'conv1', 'conv2']
    IRbn_layers = ['IRbn0', 'IRbn1', 'IRbn2']
    formats = ['weight']
    bn_formats = ['offset']

    for layer in conv_layers:
        for format in formats:
            filename = f'./model_txt/mnist_model_weight/{layer}_{format}.txt'
            if format == 'weight':
                # print(parm[f'module.{layer}.{format}'].shape)
                vgg.save_output_4D(filename, parm[f'module.{layer}.{format}'], True, '%d')
                # data = reshape_parm(parm[f'module.{layer}.{format}'])
            else:
                data = parm[f'module.{layer}.{format}']
    
    for layer in IRbn_layers:
        for format in bn_formats:
            filename = f'./model_txt/mnist_model_weight/{layer}_{format}.txt'
            data = parm[f'module.{layer}.{format}']
            
            # Convert data to hexadecimal format
            data_hex = np.vectorize(convert_to_hex)(data)
            
            # Save data to file
            with open(filename, 'w') as file:
                for row in data_hex:
                    file.write(row + ' ')


    np.savetxt('./model_txt/mnist_model_weight/fc_weight.txt', parm['module.fc.weight'], fmt='%.5f')
    np.savetxt('./model_txt/mnist_model_weight/fc_bias.txt', parm['module.fc.bias'], fmt='%.5f')

def fintune_test(model, model_fintune):
    model_state_dict = model.state_dict()
    modify_state_dict = model_fintune.state_dict()
    max_numbar = 0
    min_numbar = 0
    parm1 = {}
    # for model_name, model_value in model_state_dict.items():
    #     # if (model_name == 'module.conv0.weight' or model_name == 'module.conv1.weight' or model_name == 'module.conv2.weight' or model_name == 'module.conv3.weight' or model_name == 'module.conv4.weight' or model_name == 'module.conv5.weight'):
    #     #     print(model_name)
    #     #     print(model_value)
    #     if (model_name == 'module.conv0.bias' or model_name == 'module.conv1.bias' or model_name == 'module.conv2.bias'):
    #         print(model_name)
    #         print(model_value)
    #         number_max = torch.max(model_value)
    #         number_min = torch.min(model_value)
    #         # print(model_value.size())
    #         max_numbar = max(number_max, max_numbar)
    #         min_numbar = min(number_min, min_numbar)
    # print(max_numbar)
    # print(min_numbar)
    for model_name, model_value in model_state_dict.items():
        # parm1[model_name] = model_value.detach().cpu().numpy()
        print(model_name)
        # print(model_value)
        parm1[model_name] = model_value
        # if model_name in modify_state_dict:
        if (model_name == 'module.conv0.weight' or model_name == 'module.conv1.weight' or model_name == 'module.conv2.weight'):
            # print(model_name)
            # print(model_value)
            bw = model_value - model_value.view(model_value.size(0), -1).cuda().mean(-1).view(model_value.size(0), 1, 1, 1)
            bw = bw / bw.view(bw.size(0), -1).cuda().std(-1).view(bw.size(0), 1, 1, 1)
            sw = torch.pow(torch.tensor(2*bw.size(0)).cuda().half(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)).cuda() / math.log(2)).cuda().round().half()).cuda().view(bw.size(0), 1, 1, 1).detach()
            bw = torch.sign(bw).cuda()
            bw = (bw * sw).cuda()
            bw = torch.where(bw >= 0, 1, -1).cuda()
            # bw = torch.where(model_value >= 0, 1, -1).cuda()
            model_state_dict[model_name].copy_(bw)
            modify_state_dict[model_name].copy_(bw)
            # print(bw)
        elif (model_name == 'module.conv0.bias' or model_name == 'module.conv1.bias' or model_name == 'module.conv2.bias'):
        #     # print(model_name)
        #     # print(model_value)
        #     # convert bias to int
            output_tensor = convert_float_to_integer(model_value)
        #     # print(output_tensor)
            model_state_dict[model_name].copy_(output_tensor)
        else:
            modify_state_dict[model_name].copy_(model_value)
        # if (model_name == 'module.conv0.bias' or model_name == 'module.conv1.bias' or model_name == 'module.conv2.bias' or model_name == 'module.conv3.bias' or model_name == 'module.conv4.bias' or model_name == 'module.bn5.weight' or model_name == 'module.bn5.bias'):
        #     print(model_name)
        #     print(model_value)    
    print('=====================')       # TODO: check the bias

    modify_state_dict['module.IRbn0.offset'].copy_(model_state_dict['module.conv0.bias'])
    modify_state_dict['module.IRbn1.offset'].copy_(model_state_dict['module.conv1.bias'])
    modify_state_dict['module.IRbn2.offset'].copy_(model_state_dict['module.conv2.bias'])


    # for model_name, model_value in model_state_dict.items():
    #     if (model_name == 'module.IRbn0.offset'):
    #         model_value.copy_(parm1['module.conv0.bias'])
    #     if (model_name == 'module.IRbn1.offset'):
    #         model_value.copy_(parm1['module.conv1.bias'])
    #     if (model_name == 'module.IRbn2.offset'):
    #         model_value.copy_(parm1['module.conv2.bias'])
    #     if (model_name == 'module.IRbn3.offset'):
    #         model_value.copy_(parm1['module.conv3.bias'])
    #     if (model_name == 'module.IRbn4.offset'):
    #         model_value.copy_(parm1['module.conv4.bias'])
    #     if (model_name == 'module.IRbn5.offset'):
    #         model_value.copy_(parm1['module.conv5.bias'])
    # for model_name, model_value in modify_state_dict.items():
    #     # print(model_name)
    #     # print(model_value)
    #     if (model_name == 'module.IRbn0.offset'):
    #         modify_state_dict['module.IRbn0.offset'].copy_(model_state_dict['module.conv0.bias'])
    #     if (model_name == 'module.IRbn1.offset'):
    #         print(model_value)
    #     if (model_name == 'module.IRbn2.offset'):
    #         print(model_value)
        # if (model_name == 'module.IRbn3.offset'):
        #     print(model_value)
    #     if (model_name == 'module.IRbn4.offset'):
    #         print(model_value)
    #     if (model_name == 'module.IRbn5.offset'):
    #         print(model_value)
    # model.load_state_dict(model_state_dict)

    # offset = {}
    # caloffset(parm1, 'module.bn0', offset)
    # caloffset(parm1, 'module.bn1', offset)
    # caloffset(parm1, 'module.bn2', offset)
    # caloffset(parm1, 'module.bn3', offset)
    # caloffset(parm1, 'module.bn4', offset)
    # caloffset(parm1, 'module.bn5', offset)
        
    # for model_name, model_value in model_state_dict.items():
    #     print(model_name)
    #     if (model_name == 'module.conv0.bias'):
    #         modify_state_dict[model_name].copy_(torch.tensor(offset['module.bn0']) + model_value)
    #     if (model_name == 'module.conv1.bias'):
    #         modify_state_dict[model_name].copy_(torch.tensor(offset['module.bn1']) + model_value)
    #     if (model_name == 'module.conv2.bias'):
    #         modify_state_dict[model_name].copy_(torch.tensor(offset['module.bn2']) + model_value)
    #     if (model_name == 'module.conv3.bias'):
    #         modify_state_dict[model_name].copy_(torch.tensor(offset['module.bn3']))
    #     if (model_name == 'module.conv4.bias'):
    #         modify_state_dict[model_name].copy_(torch.tensor(offset['module.bn4']))
        # if (model_name == 'module.conv5.bias'):
        #     model_state_dict[model_name].copy_(torch.tensor(offset['module.bn5']))

        # if (model_name == 'module.IRbn0.offset'):
        #     modify_state_dict[model_name].copy_(torch.tensor(offset['module.bn0']))
        # if (model_name == 'module.IRbn1.offset'):
        #     modify_state_dict[model_name].copy_(torch.tensor(offset['module.bn1']))
        # if (model_name == 'module.IRbn2.offset'):
        #     modify_state_dict[model_name].copy_(torch.tensor(offset['module.bn2']))
        # if (model_name == 'module.IRbn3.offset'):
        #     modify_state_dict[model_name].copy_(torch.tensor(offset['module.bn3']))
        # if (model_name == 'module.IRbn4.offset'):
        #     modify_state_dict[model_name].copy_(torch.tensor(offset['module.bn4']))
        # if (model_name == 'module.IRbn5.offset'):
        #     modify_state_dict[model_name].copy_(torch.tensor(offset['module.bn5']))

    # print(modify_state_dict['module.conv1.bias'])
    # print(modify_state_dict['module.IRbn0.offset'])
    # print(modify_state_dict['module.IRbn1.offset'])
    # print(modify_state_dict['module.IRbn2.offset'])
    # print(modify_state_dict['module.IRbn3.offset'])
    # print(modify_state_dict['module.IRbn4.offset'])
    # print(modify_state_dict['module.IRbn5.offset'])
    # model_fintune.load_state_dict(modify_state_dict)
    # save model
    # os.path.join(args.save_dir, '{}.th'.format(save_name))
    # torch.save(model_fintune.state_dict(), './cifar10_train_model_2b_padswap_m1/IR-Net_vggsmall_fintune.pth')


def caloffset(parm, name, offset_tenor):
    # var_abs = np.absolute(parm[name + '.running_var'])
    # # var_abs =  parm[name + '.running_var'].abs()
    # # print(var_abs)
    # div_gammer = var_abs / parm[name + '.weight']
    # # print(div_gammer)
    # mul_beta = div_gammer * parm[name + '.bias']
    # # print(mul_beta)
    # offset_tenor[name] = mul_beta - parm[name + '.running_mean']
    # offset_tenor[name] = np.negative(parm[name + '.running_mean'])
    offset_tenor[name] = torch.negative(parm[name + '.running_mean'])
    # print(offset_tenor[name])
    # print(offset_tenor[name].shape)

def fintune(model):
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            if module.bias is not None:
                module.bias.requires_grad = False
            module.weight.requires_grad = False

def torch2numpy(weight):
    bw = weight - weight.view(weight.size(0), -1).cuda().mean(-1).view(weight.size(0), 1, 1, 1)
    bw = bw / bw.view(bw.size(0), -1).cuda().std(-1).view(bw.size(0), 1, 1, 1)
    sw = torch.pow(torch.tensor(2*bw.size(0)).cuda().half(), (torch.log(bw.abs().view(bw.size(0), -1).mean(-1)).cuda() / math.log(2)).cuda().round().half()).cuda().view(bw.size(0), 1, 1, 1).detach()
    bw = torch.sign(bw).cuda()
    bw = (bw * sw).cuda()
    bw = torch.where(bw >= 0, 1, 0).cuda()
    return bw.detach().cpu().numpy()

def convert_float_to_integer(tensor):
    tensor = tensor.to(torch.float32)
    ceil_tensor = torch.ceil(tensor)
    floor_tensor = torch.floor(tensor)
    converted_tensor = torch.where(tensor > 0, ceil_tensor, floor_tensor)
    return converted_tensor.to(torch.float16)

def pre_process_img(loader):
    new_data = []
    with torch.no_grad():
    #     for i, (input, target) in enumerate(loader):
    #         input_var = input.cuda()

    #         # Generate output tensor with expanded channels
    #         out_tensor = torch.empty((input_var.size(0), input_var.size(1) * 3, input_var.size(2), input_var.size(3))).cuda()

    #         # Calculate the range for each equal part
    #         max_num = torch.max(input_var)
    #         min_num = torch.min(input_var)
    #         range_per_part = (max_num - min_num) / 3

    #         # Calculate the thresholds for each equal part
    #         thresholds = [min_num + range_per_part, min_num + 2 * range_per_part]

    #         # Convert input to 1 and -1 based on the thresholds
    #         for c in range(input_var.size(1)):
    #             out_tensor[:, c * 3, :, :] = torch.where(input_var[:, c, :, :] >= thresholds[1], 1, -1)
    #             out_tensor[:, c * 3 + 1, :, :] = torch.where((input_var[:, c, :, :] >= thresholds[1]) & (input_var[:, c, :, :] < thresholds[0]), 1, -1)
    #             out_tensor[:, c * 3 + 2, :, :] = torch.where(input_var[:, c, :, :] < thresholds[0], 1, -1)

    #         num_ones = torch.sum(out_tensor == 1).item()
    #         num_minus_ones = torch.sum(out_tensor == -1).item()

    #         print("Number of ones:", num_ones)
    #         print("Number of -1s:", num_minus_ones)
    #         print(out_tensor.size())
            
    #         input_var = torch.where(input_var >= 0, 1, -1).cuda()
    #         num_ones = torch.sum(input_var == 1).item()
    #         num_minus_ones = torch.sum(input_var == -1).item()

    #         print("Number of ones:", num_ones)
    #         print("Number of -1s:", num_minus_ones)
    #         print(input_var.size())
            
    #         return out_tensor

        for input, target in loader:
            input_var = input.cuda()

            # Split each channel's image values into thirds
            input_split = torch.chunk(input_var, 3, dim=1)

            # Calculate the middle value for each channel
            channel_means = [torch.mean(channel) for channel in input_split]

            # Initialize the new input tensor
            new_input = torch.empty((input_var.size(0), 9, 32, 32)).cuda()

            # Assign values to the new input tensor based on channel means
            for i in range(3):
                new_input[:, i * 3: (i + 1) * 3, :, :] = torch.where(input_split[i] > channel_means[i], 1, -1)

            # num_ones = torch.sum(new_input == 1).item()
            # num_neg_ones = torch.sum(new_input == -1).item()

            # print("Number of ones:", num_ones)
            # print("Number of negative ones:", num_neg_ones)
            # print(new_input.size())

            new_data.append((new_input, target))
    gc.collect()
    torch.cuda.empty_cache()
    return new_data

def real_predict_img(model):
    # input an image
    data = np.full(784, -1)
    index = np.array(
        [117,90,118,89,62,61,34,35,63,36,64,37,65,38,66,39,67,40,68,41,69,42,70,43,71,
         97,98,125,126,153,154,152,180,181,208,209,207,235,236,263,264,262,290,291,317,
         318,345,346,344,372,373,400,401,399,427,428,455,456,483,484,482,510,511,538,539,
         537,565,566,564,592,593,512,429,457,374,402,319,292,265,293,266,294,238,239,267,
         329,302,330,357,358,301,274,])
    data[index] = 1
    data = data.reshape(28, 28)
    img_show(data)

    # numpy to tensor
    data = torch.from_numpy(data).float()

    if args.half:
        data = data.cuda().half()

    model.eval()

    output = model(data.unsqueeze(0).unsqueeze(0).cuda())
    _, pred = output.topk(1, 1, True, True)
    print('predict result is: {0}'.format(pred.item()))


# for show train and test data image
def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    logging.info('==> Saving checkpoint ...')
    torch.save(state, filename)
    # torch.save(state, filename, _use_new_zipfile_serialization = False)
    pass


def save_bestpoint(state, is_best, model, filename='bestpoint.pth.tar'):
    """
    Save the training model
    """
    logging.info('==> Saving best model ...')
    # print(model.state_dict())
    torch.save(state, filename)
    # torch.save(state, filename, _use_new_zipfile_serialization = False)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def kt(model, k, t):
    model.module.conv0.k = k
    model.module.conv1.k = k
    model.module.conv2.k = k
    # model.module.conv3.k = k
    # model.module.conv4.k = k
    model.module.conv0.t = t
    model.module.conv1.t = t
    model.module.conv2.t = t
    # model.module.conv3.t = t
    # model.module.conv4.t = t


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    main()
