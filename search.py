import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn

from torch.autograd import Variable
import models.model_search_gan as model_search_gan
import cfg_search
from architect import Architect_gen
from tensorboardX import SummaryWriter

from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from functions import LinearLrDecay, duality_gap, find_worst, copy_params, validate, train_gan_parameter, train_gan_alpha
import utils.utils as utils


args = cfg_search.parse_args()

dataset = {'cifar10': 'CIFAR10',
           'stl10': 'STL10'}

if args.load_path == '':
  args.save = 'logs/{}-{}-{}'.format(args.exp_name, args.save, time.strftime("%Y%m%d-%H%M%S"))
  utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
  path_helper = utils.create_sub_dir(args.save)
else:
  args.save = args.load_path
  path_helper = utils.create_sub_dir(args.save)
  

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
if args.load_path == '':
  fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
else:
  fh = logging.FileHandler(os.path.join(args.load_path, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10


def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  # Create tensorboard logger
  writer_dict = {
    'writer' : SummaryWriter(path_helper['log']),
    'inner_steps' : 0,
    'val_steps' : 0,
    'valid_global_steps': 0 
  }

  # set tf env
  if args.eval:
    _init_inception()
    inception_path = check_or_download_inception(None)
    create_inception_graph(inception_path)

  # fid_stat
  if args.dataset.lower() == 'cifar10':
    fid_stat = 'fid_stat/fid_stats_cifar10_train.npz'
  elif args.dataset.lower() == 'stl10':
    fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
  elif args.dataset.lower() == 'mnist': 
    fid_stat = 'fid_stat/stl10_train_unlabeled_fid_stats_48.npz'
  else:
    raise NotImplementedError(f'no fid stat for {args.dataset.lower()}')
  assert os.path.exists(fid_stat)
  
  # initial
  fixed_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (25, args.latent_dim)))
  FID_best = 1e+4
  IS_best = 0.
  FID_best_epoch = 0
  IS_best_epoch = 0

  # build gen and dis
  gen = eval('model_search_gan.' + args.gen)(args)
  gen = gen.cuda()
  dis = eval('model_search_gan.' + args.dis)(args)
  dis = dis.cuda()
  logging.info("generator param size = %fMB", utils.count_parameters_in_MB(gen))
  logging.info("discriminator param size = %fMB", utils.count_parameters_in_MB(dis))
  if args.parallel:
    gen = nn.DataParallel(gen)
    dis = nn.DataParallel(dis)

  # resume training
  if args.load_path != '':
    gen.load_state_dict(torch.load(os.path.join(args.load_path, 'model', 'weights_gen_' + 'last' + '.pt')))
    dis.load_state_dict(torch.load(os.path.join(args.load_path, 'model', 'weights_dis_' + 'last' + '.pt')))

  # set optimizer for parameters W of gen and dis
  gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
  dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
 
  # set moving average parameters for generator
  gen_avg_param = copy_params(gen)

  img_size = 8 if args.grow else args.img_size
  train_transform, valid_transform = eval('utils.' + '_data_transforms_' + args.dataset + '_resize')(args, img_size)
  if args.dataset == 'cifar10': 
    train_data = eval('dset.' + dataset[args.dataset])(root=args.data, train=True, download=True, transform=train_transform)
  elif args.dataset == 'stl10': 
    train_data = eval('dset.' + dataset[args.dataset])(root=args.data, download=True, transform=train_transform)

  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train))

  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.gen_batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=2)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.gen_batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=2)
  logging.info('length of train_queue is {}'.format(len(train_queue)))
  logging.info('length of valid_queue is {}'.format(len(valid_queue)))
  
  max_iter = len(train_queue) * args.epochs

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        gen_optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, max_iter * args.n_critic)
  dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, max_iter * args.n_critic)
  
  architect = Architect_gen(gen, dis, args, 'duality_gap_with_mm', logging)

  gen.set_gumbel(args.use_gumbel)
  dis.set_gumbel(args.use_gumbel)
  for epoch in range(args.start_epoch + 1, args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)
    logging.info('epoch %d gen_lr %e', epoch, args.g_lr)
    logging.info('epoch %d dis_lr %e', epoch, args.d_lr)

    genotype_gen = gen.genotype()
    logging.info('gen_genotype = %s', genotype_gen)

    if 'Discriminator' not in args.dis:
      genotype_dis = dis.genotype()
      logging.info('dis_genotype = %s', genotype_dis)

    print('up_1: ', F.softmax(gen.alphas_up_1, dim=-1))
    print('up_2: ', F.softmax(gen.alphas_up_2, dim=-1))
    print('up_3: ', F.softmax(gen.alphas_up_3, dim=-1))

    # determine whether use gumbel or not
    if epoch == args.fix_alphas_epochs + 1:
      gen.set_gumbel(args.use_gumbel)
      dis.set_gumbel(args.use_gumbel)    

    # grow discriminator and generator
    if args.grow:
      dis.cur_stage = grow_ctrl(epoch, args.grow_epoch)
      gen.cur_stage = grow_ctrl(epoch, args.grow_epoch)
      if args.restrict_dis_grow and dis.cur_stage > 1:
        dis.cur_stage = 1
        print('debug: dis.cur_stage is {}'.format(dis.cur_stage))
      if epoch in args.grow_epoch:
        train_transform, valid_transform = utils._data_transforms_cifar10_resize(args, 2 ** (gen.cur_stage + 3))
        train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
        num_train = len(train_data)
        indices = list(range(num_train))
        split = int(np.floor(args.train_portion * num_train))
        train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.gen_batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
            pin_memory=True, num_workers=2)
        valid_queue = torch.utils.data.DataLoader(
            train_data, batch_size=args.gen_batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
            pin_memory=True, num_workers=2)
    else:
      gen.cur_stage = 2
      dis.cur_stage = 2      
    
    # training parameters
    train_gan_parameter(args, train_queue, gen, dis, gen_optimizer, dis_optimizer, gen_avg_param, logging, writer_dict)

    # training alphas
    if epoch > args.fix_alphas_epochs:
      train_gan_alpha(args, train_queue, valid_queue, gen, dis, architect, gen_optimizer, gen_avg_param, epoch, lr, writer_dict, logging)
   
    
 
    # evaluate the IS and FID
    if args.eval and epoch % args.eval_every == 0:
      inception_score, std, fid_score = validate(args, fixed_z, fid_stat, gen, writer_dict, path_helper)
      logging.info('epoch {}: IS is {}+-{}, FID is {}'.format(epoch, inception_score, std, fid_score))
      if inception_score > IS_best:
        IS_best = inception_score
        IS_epoch_best = epoch
      if fid_score < FID_best:
        FID_best = fid_score
        FID_epoch_best = epoch
      logging.info('best epoch {}: IS is {}'.format(IS_best_epoch, IS_best))
      logging.info('best epoch {}: FID is {}'.format(FID_best_epoch, FID_best))

    utils.save(gen, os.path.join(path_helper['model'], 'weights_gen_{}.pt'.format('last')))
    utils.save(dis, os.path.join(path_helper['model'], 'weights_dis_{}.pt'.format('last')))
  
  genotype_gen = gen.genotype()
  if 'Discriminator' not in args.dis:
    genotype_dis = dis.genotype()
  logging.info('best epoch {}: IS is {}'.format(IS_best_epoch, IS_best))
  logging.info('best epoch {}: FID is {}'.format(FID_best_epoch, FID_best))
  logging.info('final discovered gen_arch is {}'.format(genotype_gen))
  if 'Discriminator' not in args.dis:
    logging.info('final discovered dis_arch is {}'.format(genotype_dis))



if __name__ == '__main__':
  main() 

