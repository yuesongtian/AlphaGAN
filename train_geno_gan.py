# Code derived from AutoGAN/train.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cfg_train
import models
import datasets
from functions import train, validate, LinearLrDecay, load_params, copy_params
from utils.utils import set_log_dir, save_checkpoint, create_logger
from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from mpl_toolkits.axes_grid1 import ImageGrid

import matplotlib.pyplot as plt
import torch
import os
import numpy as np
import torch.nn as nn
from tensorboardX import SummaryWriter
from tqdm import tqdm
from copy import deepcopy

from models import genotypes 
from thop import profile, clever_format

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def main():
    args = cfg_train.parse_args()
    torch.cuda.manual_seed(args.random_seed)

    # set tf env
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

    # import network
    # gen_net = eval('models.' + args.gen_model + '.' + args.gen)(args=args).cuda()
    genotype_gen = eval('genotypes.%s' % args.arch_gen)
    gen_net = eval('models.' + args.gen_model + '.' + args.gen)(args, genotype_gen).cuda()
    # gen_net = eval('models.' + args.gen_model + '.' + args.gen)(args = args).cuda()
    if 'Discriminator' not in args.dis:
      genotype_dis = eval('genotypes.%s' % args.arch_dis)
      dis_net = eval('models.'+args.dis_model+'.'+args.dis)(args, genotype_dis).cuda()
    else:
      dis_net = eval('models.' + args.dis_model + '.'+ args.dis)(args = args).cuda()

    # weight init
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            if args.init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif args.init_type == 'orth':
                nn.init.orthogonal_(m.weight.data)
            elif args.init_type == 'xavier_uniform':
                nn.init.xavier_uniform(m.weight.data, 1.)
            else:
                raise NotImplementedError('{} unknown inital type'.format(args.init_type))
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    gen_net.apply(weights_init)
    dis_net.apply(weights_init)

    # set up data_loader
    dataset = datasets.ImageDataset(args)
    train_loader = dataset.train
    val_loader = dataset.valid

    # set optimizer
    gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_net.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
    dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_net.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
    gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, args.g_lr * 0.01, 260 * len(train_loader), args.max_iter * args.n_critic)
    dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, args.d_lr * 0.01, 260 * len(train_loader), args.max_iter * args.n_critic)

    # fid stat
    if args.dataset.lower() == 'cifar10':
        fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
    elif args.dataset.lower() == 'stl10':
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    elif args.dataset.lower() == 'mnist': 
        fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
    else:
        raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
    assert os.path.exists(fid_stat)

    # epoch number for dis_net
    args.max_epoch = args.max_epoch * args.n_critic
    if args.max_iter:
        args.max_epoch = np.ceil(args.max_iter * args.n_critic / len(train_loader))

    # initial
    fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
    fixed_z_sample = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
    gen_avg_param = copy_params(gen_net)
    start_epoch = 0
    best_fid = 1e4
    best_fid_epoch = 0
    is_with_fid = 0
    std_with_fid = 0.
    best_is = 0
    best_is_epoch = 0
    fid_with_is = 0
    best_dts = 0

    # set writer
    if args.load_path:
        print(f'=> resuming from {args.load_path}')
        assert os.path.exists(args.load_path)
        checkpoint_file = os.path.join(args.load_path, 'Model', 'checkpoint.pth')
        assert os.path.exists(checkpoint_file)
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch']
        best_fid = checkpoint['best_fid']
        gen_net.load_state_dict(checkpoint['gen_state_dict'])
        dis_net.load_state_dict(checkpoint['dis_state_dict'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        avg_gen_net = deepcopy(gen_net)
        avg_gen_net.load_state_dict(checkpoint['avg_gen_state_dict'])
        gen_avg_param = copy_params(avg_gen_net)
        del avg_gen_net

        args.path_helper = checkpoint['path_helper']
        logger = create_logger(args.path_helper['log_path'])
        logger.info(f'=> loaded checkpoint {checkpoint_file} (epoch {start_epoch})')
    else:
        # create new log dir
        assert args.exp_name
        args.path_helper = set_log_dir('logs', args.exp_name)
        logger = create_logger(args.path_helper['log_path'])

    logger.info(args)
    writer_dict = {
        'writer': SummaryWriter(args.path_helper['log_path']),
        'train_global_steps': start_epoch * len(train_loader),
        'valid_global_steps': start_epoch // args.val_freq,
    }
    
    # calculate the FLOPs and param count of G
    input = torch.randn(args.gen_batch_size, args.latent_dim).cuda()
    flops, params = profile(gen_net, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    logger.info('FLOPs is {}, param count is {}'.format(flops, params))
    
    # train loop
    dg_list = []
    worst_lr = 1e-5
    for epoch in tqdm(range(int(start_epoch), int(args.max_epoch)), desc='total progress'):
        lr_schedulers = (gen_scheduler, dis_scheduler) if args.lr_decay else None
        
        train(args, gen_net, dis_net, gen_optimizer, dis_optimizer, gen_avg_param, train_loader, epoch, writer_dict,
              args.consistent, lr_schedulers)

        if epoch and epoch % args.val_freq == 0 or epoch == int(args.max_epoch)-1:
            backup_param = copy_params(gen_net)
            load_params(gen_net, gen_avg_param)
            inception_score, std, fid_score = validate(args, fixed_z, fid_stat, gen_net, writer_dict, args.path_helper, search=False)
            logger.info(f'Inception score: {inception_score}, FID score: {fid_score}+-{std} || @ epoch {epoch}.')
            load_params(gen_net, backup_param)
            if fid_score < best_fid:
                best_fid = fid_score
                best_fid_epoch = epoch
                is_with_fid = inception_score
                std_with_fid = std
                is_best = True
            else:
                is_best = False
            if inception_score > best_is:
                best_is = inception_score
                best_std = std
                fid_with_is = fid_score
                best_is_epoch = epoch
        else:
            is_best = False

        # save generated images
        if epoch % args.image_every == 0:
            gen_noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))
            # gen_images = gen_net(fixed_z_sample)
            # gen_images = gen_images.reshape(args.eval_batch_size, 32, 32, 3)
            # gen_images = gen_images.cpu().detach()
            gen_images = gen_net(fixed_z_sample).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
                                                                                                torch.uint8).numpy()
            fig = plt.figure()
            grid = ImageGrid(fig, 111, nrows_ncols=(10, 10), axes_pad=0)
            for x in range(args.eval_batch_size):
                grid[x].imshow(gen_images[x]) # cmap="gray")
                grid[x].set_xticks([])
                grid[x].set_yticks([])
            plt.savefig(
                os.path.join(args.path_helper['sample_path'], "epoch_{}.png".format(epoch)))
            plt.close()
       
        
        avg_gen_net = deepcopy(gen_net)
        # avg_gen_net = eval('models.'+args.gen_model+'.' + args.gen)(args, genotype_gen).cuda()
        # avg_gen_net = eval('models.' + args.gen_model + '.' + args.gen)(args=args).cuda()
        load_params(avg_gen_net, gen_avg_param)
        save_checkpoint({
            'epoch': epoch + 1,
            'gen_model': args.gen_model,
            'dis_model': args.dis_model,
            'gen_state_dict': gen_net.state_dict(),
            'dis_state_dict': dis_net.state_dict(),
            'avg_gen_state_dict': avg_gen_net.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'best_fid': best_fid,
            'path_helper': args.path_helper
        }, is_best, args.path_helper['ckpt_path'])
        del avg_gen_net
    logger.info('best_is is {}+-{}@{} epoch, fid is {}, best_fid is {}@{}, is is {}+-{}'.format(best_is, best_std, best_is_epoch, fid_with_is, best_fid, best_fid_epoch, is_with_fid, std_with_fid))

if __name__ == '__main__':
    main()
