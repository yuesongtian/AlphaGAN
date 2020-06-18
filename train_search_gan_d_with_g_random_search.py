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
from model_search import Network
import model_search_gan
import cfg
from model_search_gan import Network_gen, Network_dis, Network_dis_simplify, Network_dis_Auto, Discriminator
from architect import Architect, Architect_gen, Architect_gen_fix_d
from tensorboardX import SummaryWriter

from utils.inception_score import _init_inception
from utils.fid_score import create_inception_graph, check_or_download_inception
from functions import LinearLrDecay, duality_gap, find_worst, find_worst_wgan, find_worst_BCE, find_worst_fix_d, copy_params, validate, grow_ctrl
import utils.utils as utils

"""
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--dataset', type=str, default='cifar10', help='the name of dataset')
parser.add_argument('--gen', type=str, default='Network_gen', help='the name of generator instance')
parser.add_argument('--dis', type=str, default='Network_dis_Auto', help='the name of discriminator instance')
parser.add_argument('--num_classes', type=int, default=10, help='the number of classes of the dataset')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('-gen_bs', '--gen_batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('-dis_bs', '--dis_batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--eval_batch_size', type=int, default=100)
parser.add_argument('--num_eval_imgs', type=int, default=50000)
parser.add_argument('--channels', type=int, default=3, help='number of image channels')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--g_lr', type=float, default=0.0002, help='adam: gen learning rate')
parser.add_argument('--d_lr', type=float, default=0.0002, help='adam: disc learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--eval_every', type=int, default=4, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
parser.add_argument('--beta1', type=float, default=0.0, help='adam: decay of first order momentum of gradient')
parser.add_argument('--beta2', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
parser.add_argument('--gf_dim', type=int, default=64, help='The base channel num of gen')
parser.add_argument('--df_dim', type=int, default=64, help='The base channel num of disc')
parser.add_argument('--latent_dim', type=int, default=128, help='dimensionality of the latent space')
parser.add_argument('--n_critic', type=int, default=5, help='number of training steps for discriminator per iter')
parser.add_argument('--fix_alphas_epochs', type=int, default=50, help='number of training epochs to fix updating alphas')
parser.add_argument('--bottom_width', type=int, default=4, help="the base resolution of the GAN")
parser.add_argument('--d_spectral_norm', action='store_false', help='add spectral_norm on discriminator?')
parser.add_argument('--parallel', action='store_true', help='use data parallel')
parser.add_argument('--dis_with_bn', action='store_true', help='D has BN layers?')
parser.add_argument('--gen_with_bn', action='store_true', help='G has BN layers?')
parser.add_argument('--update_alphas', action='store_false', help='alphas require grad?')
parser.add_argument('--t', type=float, default=0.5, help='teperature of gumbel softmax')
parser.add_argument('--exp_name', type=str, default='', help='the name of experiment')
"""
args = cfg.parse_args()

args.save = '{}-{}-{}'.format(args.exp_name, args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))
if not os.path.exists(os.path.join(args.save, 'log_tensorboard')):
  os.makedirs(os.path.join(args.save, 'log_tensorboard'))
path_helper = utils.create_sub_dir(args.save)

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
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

  # set optimizer for parameters W of gen and dis
  gen_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, gen.parameters()),
                                     args.g_lr, (args.beta1, args.beta2))
  dis_optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, dis.parameters()),
                                     args.d_lr, (args.beta1, args.beta2))
 
  # set moving average parameters for generator
  gen_avg_param = copy_params(gen)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
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

  max_iter = len(train_queue) * args.epochs

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        gen_optimizer, float(args.epochs), eta_min=args.learning_rate_min)
  gen_scheduler = LinearLrDecay(gen_optimizer, args.g_lr, 0.0, 0, max_iter * args.n_critic)
  dis_scheduler = LinearLrDecay(dis_optimizer, args.d_lr, 0.0, 0, max_iter * args.n_critic)
  
  architect = Architect_gen(gen, dis, args, 'duality_gap_with_mm_BCE')

  gen.set_gumbel(False)
  dis.set_gumbel(False)
  """
  for epoch in range(args.epochs):
    scheduler.step()
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)
    logging.info('epoch %d gen_lr %e', epoch, args.g_lr)
    logging.info('epoch %d dis_lr %e', epoch, args.d_lr)

    genotype_gen = gen.genotype()
    logging.info('gen_genotype = %s', genotype_gen)

    # genotype_dis = dis.genotype()
    # logging.info('dis_genotype = %s', genotype_dis)

    # print(F.softmax(gen.alphas_normal_1, dim=-1))
    # print(F.softmax(gen.alphas_normal_2, dim=-1))
    # print(F.softmax(gen.alphas_normal_3, dim=-1))
    print('up_1: ', F.softmax(gen.alphas_up_1, dim=-1))
    print('up_2: ', F.softmax(gen.alphas_up_2, dim=-1))
    print('up_3: ', F.softmax(gen.alphas_up_3, dim=-1))
    
    # print(F.softmax(gen.alphas_skip_2, dim=-1))
    # print(F.softmax(gen.alphas_skip_3, dim=-1))

    # determine whether use gumbel or not
    if epoch == args.fix_alphas_epochs + 1:
      gen.set_gumbel(False)
      dis.set_gumbel(False)    

    # grow discriminator
    # dis.cur_stage = grow_ctrl(epoch)

    # training parameters
    train_gan_parameter(train_queue, gen, dis, gen_optimizer, dis_optimizer, gen_avg_param, logging, writer_dict)

     
    # evaluate the IS and FID
    if epoch % args.eval_every == 0:
      inception_score, fid_score = validate(args, fixed_z, fid_stat, gen, writer_dict, logging, path_helper)
      logging.info('epoch {}: IS is {}, FID is {}'.format(epoch, inception_score, fid_score))
      if inception_score > IS_best:
        IS_best = inception_score
        IS_epoch_best = epoch
      if fid_score < FID_best:
        FID_best = fid_score
        FID_epoch_best = epoch
      logging.info('best epoch {}: IS is {}'.format(IS_best_epoch, IS_best))
      logging.info('best epoch {}: FID is {}'.format(FID_best_epoch, FID_best))

    utils.save(gen, os.path.join(path_helper['model'], 'weights_gen_{}.pt'.format(epoch)))
    utils.save(dis, os.path.join(path_helper['model'], 'weights_dis_{}.pt'.format(epoch)))
  """ 
 
  dis.load_state_dict(torch.load(os.path.join(args.load_path, 'model', 'weights_dis_99.pt')))
  gen.load_state_dict(torch.load(os.path.join(args.load_path, 'model', 'weights_gen_99.pt')))
  dis = dis.cuda()
  gen = gen.cuda()  
  val_results = []
  for i in range(200):
    arch = gen._sample_arch()
    gen._set_arch(arch)
    inception_score, fid = validate(args, fixed_z, fid_stat, gen, writer_dict, logging, path_helper)
    genotype_gen = gen.genotype()
    val_results.append((genotype_gen, inception_score))
  val_results = sorted(val_results, key = lambda x:-x[1])
  genotype_gen = val_results[0][0]
  IS = val_results[0][1] 

  logging.info('best epoch {}: IS is {}'.format(IS_best_epoch, IS_best))
  logging.info('best epoch {}: FID is {}'.format(FID_best_epoch, FID_best))
  logging.info('final discovered gen_arch is {}, with IS {}'.format(genotype_gen, IS))
  if 'Discriminator' not in args.dis:
    genotype_dis = dis.genotype()
    logging.info('final discovered dis_arch is {}'.format(genotype_dis))

def train_gan_parameter(train_queue, gen, dis, 
              gen_optimizer, dis_optimizer,
              gen_avg_param,
              logging,
              writer_dict, schedulers = None):

  # tensorboard logger
  writer = writer_dict['writer']
  inner_step = writer_dict['inner_steps']

  gen.train()
  dis.train()
  gen_step = 0
  for step, (input, target) in enumerate(train_queue):
    torch.autograd.set_detect_anomaly(True)
    
    # sample arch of G
    arch = gen._sample_arch()
    gen._set_arch(arch)

    # Adversarial ground truths
    real_imgs = input.type(torch.cuda.FloatTensor)

    # Sample noise as generator input
    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (input.shape[0], args.latent_dim)))

    # ---------------------
    #  Train Discriminator
    # ---------------------
    dis_optimizer.zero_grad()

    real_validity = dis(real_imgs)
    fake_imgs = gen(z).detach()
    assert fake_imgs.size() == real_imgs.size()

    fake_validity = dis(fake_imgs)

    # cal loss
    d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
    d_loss.backward()
    dis_optimizer.step()

    writer.add_scalar('d_loss', d_loss.item(), inner_step)

    # -----------------
    #  Train Generator
    # -----------------
    if inner_step % args.n_critic == 0:
      gen_optimizer.zero_grad()

      gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
      gen_imgs = gen(gen_z)
      fake_validity = dis(gen_imgs)

      # cal loss
      g_loss = -torch.mean(fake_validity)
      # g_loss = -torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
      g_loss.backward()
      gen_optimizer.step()

      # adjust learning rate
      if schedulers:
        gen_scheduler, dis_scheduler = schedulers
        g_lr = gen_scheduler.step(inner_step)
        d_lr = dis_scheduler.step(inner_step)
        writer.add_scalar('LR/g_lr', g_lr, inner_step)
        writer.add_scalar('LR/d_lr', d_lr, inner_step)

      # moving average weight
      for p, avg_p in zip(gen.parameters(), gen_avg_param):
        avg_p.mul_(0.999).add_(0.001, p.data)

      logging.info('step {}: g_loss is {}, d_loss is {}'.format(inner_step, g_loss.data, d_loss.data))

      writer.add_scalar('g_loss', g_loss.item(), inner_step)
      gen_step += 1
    
    inner_step += 1
    writer_dict['inner_steps'] += 1

  return

def train_gan_parameter_wgan(train_queue, gen, dis, 
              gen_optimizer, dis_optimizer,
              gen_avg_param,
              logging,
              writer_dict, schedulers = None):

  # tensorboard logger
  writer = writer_dict['writer']
  inner_step = writer_dict['inner_steps']

  gen.train()
  dis.train()
  gen_step = 0
  for step, (input, target) in enumerate(train_queue):

    # Adversarial ground truths
    real_imgs = input.type(torch.cuda.FloatTensor)

    # Sample noise as generator input
    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (input.shape[0], args.latent_dim)))

    # ---------------------
    #  Train Discriminator
    # ---------------------
    dis_optimizer.zero_grad()

    # set labels for discriminator
    one = torch.ones([real_imgs.shape[0], 1]).cuda()
    mone = one * -1

    real_validity = dis(real_imgs)
    fake_imgs = gen(z).detach()
    assert fake_imgs.size() == real_imgs.size()

    fake_validity = dis(fake_imgs)

    # cal loss
    # d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
    #              torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
    d_loss = torch.mean(real_validity - fake_validity)
    real_validity.backward(one)
    fake_validity.backward(mone)
    # d_loss.backward()
    dis_optimizer.step()

    writer.add_scalar('d_loss', d_loss.item(), inner_step)

    # -----------------
    #  Train Generator
    # -----------------
    if inner_step % args.n_critic == 0:
      gen_optimizer.zero_grad()

      gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
      gen_imgs = gen(gen_z)
      fake_validity = dis(gen_imgs)

      # set labels for discriminator
      ones = torch.ones([gen_imgs.shape[0], 1]).cuda()

      # cal loss
      # g_loss = -torch.mean(fake_validity)
      # g_loss = -torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
      g_loss = -torch.mean(fake_validity)
      # g_loss.backward()
      fake_validity.backward(ones)
      gen_optimizer.step()

      # adjust learning rate
      if schedulers:
        gen_scheduler, dis_scheduler = schedulers
        g_lr = gen_scheduler.step(inner_step)
        d_lr = dis_scheduler.step(inner_step)
        writer.add_scalar('LR/g_lr', g_lr, inner_step)
        writer.add_scalar('LR/d_lr', d_lr, inner_step)

      # moving average weight
      for p, avg_p in zip(gen.parameters(), gen_avg_param):
        avg_p.mul_(0.999).add_(0.001, p.data)

      logging.info('step {}: g_loss is {}, d_loss is {}'.format(inner_step, g_loss.data, d_loss.data))

      writer.add_scalar('g_loss', g_loss.item(), inner_step)
      gen_step += 1
    
    inner_step += 1
    writer_dict['inner_steps'] += 1

  return

def train_gan_parameter_BCE(train_queue, gen, dis, 
              gen_optimizer, dis_optimizer,
              gen_avg_param,
              logging,
              writer_dict, schedulers = None):

  # tensorboard logger
  writer = writer_dict['writer']
  inner_step = writer_dict['inner_steps']

  gen.train()
  dis.train()
  gen_step = 0
  for step, (input, target) in enumerate(train_queue):

    # Adversarial ground truths
    real_imgs = input.type(torch.cuda.FloatTensor)

    # Sample noise as generator input
    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (input.shape[0], args.latent_dim)))

    # ---------------------
    #  Train Discriminator
    # ---------------------
    dis_optimizer.zero_grad()

    # set labels for discriminator
    one = torch.ones([real_imgs.shape[0], 1]).cuda()
    zero = one * 0

    real_validity = dis(real_imgs)
    fake_imgs = gen(z).detach()
    assert fake_imgs.size() == real_imgs.size()

    fake_validity = dis(fake_imgs)

    # cal loss
    # d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
    #              torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
    # d_loss = torch.mean(real_validity - fake_validity)
    d_loss = nn.BCEWithLogitsLoss()(real_validity, one) + nn.BCEWithLogitsLoss()(fake_validity, zero)
    d_loss.backward()
    dis_optimizer.step()

    writer.add_scalar('d_loss', d_loss.item(), inner_step)

    # -----------------
    #  Train Generator
    # -----------------
    if inner_step % args.n_critic == 0:
      gen_optimizer.zero_grad()

      gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
      gen_imgs = gen(gen_z)
      fake_validity = dis(gen_imgs)

      # set labels for discriminator
      one = torch.ones([gen_imgs.shape[0], 1]).cuda()

      # cal loss
      # g_loss = -torch.mean(fake_validity)
      # g_loss = -torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
      # g_loss = -torch.mean(fake_validity)
      g_loss = nn.BCEWithLogitsLoss()(fake_validity, one)
      g_loss.backward()
      gen_optimizer.step()

      # adjust learning rate
      if schedulers:
        gen_scheduler, dis_scheduler = schedulers
        g_lr = gen_scheduler.step(inner_step)
        d_lr = dis_scheduler.step(inner_step)
        writer.add_scalar('LR/g_lr', g_lr, inner_step)
        writer.add_scalar('LR/d_lr', d_lr, inner_step)

      # moving average weight
      for p, avg_p in zip(gen.parameters(), gen_avg_param):
        avg_p.mul_(0.999).add_(0.001, p.data)

      logging.info('step {}: g_loss is {}, d_loss is {}'.format(inner_step, g_loss.data, d_loss.data))

      writer.add_scalar('g_loss', g_loss.item(), inner_step)
      gen_step += 1
    
    inner_step += 1
    writer_dict['inner_steps'] += 1

  return

def train_gan_alpha(args, train_loader, val_loader, 
              gen, dis, 
              architect,
              gen_optimizer, gen_avg_param,
              epoch,
              lr, writer_dict, logging):
  writer = writer_dict['writer']
  val_step = writer_dict['val_steps']
  
  # set lr for g_worst and d_worst
  if epoch <= 30:
    g_lr_worst = args.g_lr * 1.
    d_lr_worst = args.d_lr * 1.
  if epoch > 30:
    g_lr_worst = args.g_lr * 0.5
    d_lr_worst = args.d_lr * 0.5
  # build the model of gen_worst and dis_worst
  gen_worst = eval('model_search_gan.' + args.gen)(args)
  gen_worst = gen.copy_alphas(gen_worst)
  gen_worst.set_gumbel(False)
  gen_worst = gen_worst.cuda()
  dis_worst = eval('model_search_gan.' + args.dis)(args)
  dis_worst = dis.copy_alphas(dis_worst)
  dis_worst.set_gumbel(False)
  dis_worst.cur_stage = dis.cur_stage
  dis_worst = dis_worst.cuda()
  if args.parallel:
    gen_worst = nn.DataParallel(gen_worst)
    dis_worst = nn.DataParallel(dis_worst) 
  # find gen_worst and dis_worst
  gen_worst, dis_worst = find_worst(args, train_loader, gen, dis, gen_worst, dis_worst, g_lr_worst, d_lr_worst, gen_avg_param)
 
  outter_steps = 10
  for i in range(outter_steps): 
    # get a random minibatch from val
    input_val, _ = next(iter(val_loader))
    input_val = input_val.cuda()
    
    # get a random minibatch from train
    input_train, _ = next(iter(train_loader))
    input_train = input_train.cuda()

    dg_loss, minmax, maxmin =  architect.step(gen_worst, dis_worst, input_train, input_val, lr, gen_optimizer, unrolled=args.unrolled)
    print('debug@: duality_gap is {}'.format(dg_loss))
    print('debug@: minmax is {}'.format(minmax))
    print('debug@: maxmin is {}'.format(maxmin))
    logging.info('epoch {}: duality_gap is {}'.format(epoch, dg_loss))    
    
    # logging the weights of opr and according grad
    logging.info('epoch_{} up_1: {}'.format(epoch, gen.alphas_up_1))
    logging.info('epoch_{} up_1 architect: {}'.format(epoch, architect.model.alphas_up_1))
    logging.info('epoch_{} up_2: {}'.format(epoch, gen.alphas_up_2))
    logging.info('epoch_{} up_3: {}'.format(epoch, gen.alphas_up_3))
    logging.info('epoch_{} normal_1: {}'.format(epoch, gen.alphas_normal_1))
    logging.info('epoch_{} normal_2: {}'.format(epoch, gen.alphas_normal_2))
    logging.info('epoch_{} normal_3: {}'.format(epoch, gen.alphas_normal_3))
    logging.info('epoch_{} skip_2: {}'.format(epoch, gen.alphas_skip_2))
    logging.info('epoch_{} skip_3: {}'.format(epoch, gen.alphas_skip_3))    
    logging.info('epoch_{} up_1_grad: {}'.format(epoch, gen.alphas_up_1.grad))
    logging.info('epoch_{} up_2_grad: {}'.format(epoch, gen.alphas_up_2.grad))
    logging.info('epoch_{} up_3_grad: {}'.format(epoch, gen.alphas_up_3.grad))
    logging.info('epoch_{} normal_1_grad: {}'.format(epoch, gen.alphas_normal_1.grad))
    logging.info('epoch_{} normal_2_grad: {}'.format(epoch, gen.alphas_normal_2.grad))
    logging.info('epoch_{} normal_3_grad: {}'.format(epoch, gen.alphas_normal_3.grad))
    logging.info('epoch_{} skip_2_grad: {}'.format(epoch, gen.alphas_skip_2.grad))
    logging.info('epoch_{} skip_3_grad: {}'.format(epoch, gen.alphas_skip_3.grad))
    
    # update writer
    # writer.add_scalar('d_loss_arch', d_loss.data, val_step)
    # writer.add_scalar('g_loss_arch', g_loss.data, val_step)
    writer.add_scalar('duality_gap', dg_loss.data, val_step + args.fix_alphas_epochs)
    writer.add_scalar('minmax', minmax.data, val_step + args.fix_alphas_epochs)
    writer.add_scalar('maxmin', maxmin.data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_normal_1.shape[0]):
      for j in range(architect.model.alphas_normal_1.shape[1]):
        writer.add_scalar('cell_1/normal_edge_{}/opr_{}'.format(i, j), architect.model.alphas_normal_1[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_normal_2.shape[0]):
      for j in range(architect.model.alphas_normal_2.shape[1]):
        writer.add_scalar('cell_2/normal_edge_{}/opr_{}'.format(i, j), architect.model.alphas_normal_2[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_normal_3.shape[0]):
      for j in range(architect.model.alphas_normal_3.shape[1]):
        writer.add_scalar('cell_3/normal_edge_{}/opr_{}'.format(i, j), architect.model.alphas_normal_3[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_up_1.shape[0]):
      for j in range(architect.model.alphas_up_1.shape[1]):
        writer.add_scalar('cell_1/up_edge_{}/opr_{}'.format(i, j), architect.model.alphas_up_1[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_up_2.shape[0]):
      for j in range(architect.model.alphas_up_2.shape[1]):
        writer.add_scalar('cell_2/up_edge_{}/opr_{}'.format(i, j), architect.model.alphas_up_2[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_up_3.shape[0]):
      for j in range(architect.model.alphas_up_3.shape[1]):
        writer.add_scalar('cell_3/up_edge_{}/opr_{}'.format(i, j), architect.model.alphas_up_3[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_skip_2.shape[0]):
      for j in range(architect.model.alphas_skip_2.shape[1]):
        writer.add_scalar('skip_2/edge_{}/opr_{}'.format(i, j), architect.model.alphas_skip_2[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_skip_3.shape[0]):
      for j in range(architect.model.alphas_skip_3.shape[1]):
        writer.add_scalar('skip_3/edge_{}/opr_{}'.format(i, j), architect.model.alphas_skip_3[i][j].data, val_step + args.fix_alphas_epochs)
    val_step += 1
    writer_dict['val_steps'] += 1

  return

def train_gan_alpha_wgan(args, train_loader, val_loader, 
              gen, dis, 
              architect,
              gen_optimizer, gen_avg_param,
              epoch,
              lr, writer_dict, logging):
  writer = writer_dict['writer']
  val_step = writer_dict['val_steps']
  
  # set lr for g_worst and d_worst
  if epoch <= 30:
    g_lr_worst = args.g_lr * 1.
    d_lr_worst = args.d_lr * 1.
  if epoch > 30:
    g_lr_worst = args.g_lr * 0.5
    d_lr_worst = args.d_lr * 0.5
  # build the model of gen_worst and dis_worst
  gen_worst = eval('model_search_gan.' + args.gen)(args)
  gen_worst = gen.copy_alphas(gen_worst)
  gen_worst.set_gumbel(False)
  gen_worst = gen_worst.cuda()
  dis_worst = eval('model_search_gan.' + args.dis)(args)
  dis_worst = dis.copy_alphas(dis_worst)
  dis_worst.set_gumbel(False)
  dis_worst = dis_worst.cuda()
  if args.parallel:
    gen_worst = nn.DataParallel(gen_worst)
    dis_worst = nn.DataParallel(dis_worst) 
  # find gen_worst and dis_worst
  gen_worst, dis_worst = find_worst_wgan(args, train_loader, gen, dis, gen_worst, dis_worst, g_lr_worst, d_lr_worst, gen_avg_param)
 
  outter_steps = 10
  for i in range(outter_steps): 
    # get a random minibatch from val
    input_val, _ = next(iter(val_loader))
    input_val = input_val.cuda()
    
    # get a random minibatch from train
    input_train, _ = next(iter(train_loader))
    input_train = input_train.cuda()

    dg_loss, minmax, maxmin =  architect.step(gen_worst, dis_worst, input_train, input_val, lr, gen_optimizer, unrolled=args.unrolled)
    print('debug@: duality_gap is {}'.format(dg_loss))
    print('debug@: minmax is {}'.format(minmax))
    print('debug@: maxmin is {}'.format(maxmin))
    logging.info('epoch {}: duality_gap is {}'.format(epoch, dg_loss))    
    
    # logging the weights of opr and according grad
    logging.info('epoch_{} up_1: {}'.format(epoch, gen.alphas_up_1))
    logging.info('epoch_{} up_1 architect: {}'.format(epoch, architect.model.alphas_up_1))
    logging.info('epoch_{} up_2: {}'.format(epoch, gen.alphas_up_2))
    logging.info('epoch_{} up_3: {}'.format(epoch, gen.alphas_up_3))
    logging.info('epoch_{} normal_1: {}'.format(epoch, gen.alphas_normal_1))
    logging.info('epoch_{} normal_2: {}'.format(epoch, gen.alphas_normal_2))
    logging.info('epoch_{} normal_3: {}'.format(epoch, gen.alphas_normal_3))
    logging.info('epoch_{} skip_2: {}'.format(epoch, gen.alphas_skip_2))
    logging.info('epoch_{} skip_3: {}'.format(epoch, gen.alphas_skip_3))    
    logging.info('epoch_{} up_1_grad: {}'.format(epoch, gen.alphas_up_1.grad))
    logging.info('epoch_{} up_2_grad: {}'.format(epoch, gen.alphas_up_2.grad))
    logging.info('epoch_{} up_3_grad: {}'.format(epoch, gen.alphas_up_3.grad))
    logging.info('epoch_{} normal_1_grad: {}'.format(epoch, gen.alphas_normal_1.grad))
    logging.info('epoch_{} normal_2_grad: {}'.format(epoch, gen.alphas_normal_2.grad))
    logging.info('epoch_{} normal_3_grad: {}'.format(epoch, gen.alphas_normal_3.grad))
    logging.info('epoch_{} skip_2_grad: {}'.format(epoch, gen.alphas_skip_2.grad))
    logging.info('epoch_{} skip_3_grad: {}'.format(epoch, gen.alphas_skip_3.grad))
    
    # update writer
    # writer.add_scalar('d_loss_arch', d_loss.data, val_step)
    # writer.add_scalar('g_loss_arch', g_loss.data, val_step)
    writer.add_scalar('duality_gap', dg_loss.data, val_step + args.fix_alphas_epochs)
    writer.add_scalar('minmax', minmax.data, val_step + args.fix_alphas_epochs)
    writer.add_scalar('maxmin', maxmin.data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_normal_1.shape[0]):
      for j in range(architect.model.alphas_normal_1.shape[1]):
        writer.add_scalar('cell_1/normal_edge_{}/opr_{}'.format(i, j), architect.model.alphas_normal_1[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_normal_2.shape[0]):
      for j in range(architect.model.alphas_normal_2.shape[1]):
        writer.add_scalar('cell_2/normal_edge_{}/opr_{}'.format(i, j), architect.model.alphas_normal_2[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_normal_3.shape[0]):
      for j in range(architect.model.alphas_normal_3.shape[1]):
        writer.add_scalar('cell_3/normal_edge_{}/opr_{}'.format(i, j), architect.model.alphas_normal_3[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_up_1.shape[0]):
      for j in range(architect.model.alphas_up_1.shape[1]):
        writer.add_scalar('cell_1/up_edge_{}/opr_{}'.format(i, j), architect.model.alphas_up_1[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_up_2.shape[0]):
      for j in range(architect.model.alphas_up_2.shape[1]):
        writer.add_scalar('cell_2/up_edge_{}/opr_{}'.format(i, j), architect.model.alphas_up_2[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_up_3.shape[0]):
      for j in range(architect.model.alphas_up_3.shape[1]):
        writer.add_scalar('cell_3/up_edge_{}/opr_{}'.format(i, j), architect.model.alphas_up_3[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_skip_2.shape[0]):
      for j in range(architect.model.alphas_skip_2.shape[1]):
        writer.add_scalar('skip_2/edge_{}/opr_{}'.format(i, j), architect.model.alphas_skip_2[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_skip_3.shape[0]):
      for j in range(architect.model.alphas_skip_3.shape[1]):
        writer.add_scalar('skip_3/edge_{}/opr_{}'.format(i, j), architect.model.alphas_skip_3[i][j].data, val_step + args.fix_alphas_epochs)
    val_step += 1
    writer_dict['val_steps'] += 1

  return

def train_gan_alpha_BCE(args, train_loader, val_loader, 
              gen, dis, 
              architect,
              gen_optimizer, gen_avg_param,
              epoch,
              lr, writer_dict, logging):
  writer = writer_dict['writer']
  val_step = writer_dict['val_steps']
  
  # set lr for g_worst and d_worst
  if epoch <= 30:
    g_lr_worst = args.g_lr * 1.
    d_lr_worst = args.d_lr * 1.
  if epoch > 30:
    g_lr_worst = args.g_lr * 0.5
    d_lr_worst = args.d_lr * 0.5
  # build the model of gen_worst and dis_worst
  gen_worst = eval('model_search_gan.' + args.gen)(args)
  gen_worst = gen.copy_alphas(gen_worst)
  gen_worst.set_gumbel(False)
  gen_worst = gen_worst.cuda()
  dis_worst = eval('model_search_gan.' + args.dis)(args)
  if args.dis != 'Discriminator':
    dis_worst = dis.copy_alphas(dis_worst)
  dis_worst.set_gumbel(False)
  dis_worst = dis_worst.cuda()
  if args.parallel:
    gen_worst = nn.DataParallel(gen_worst)
    dis_worst = nn.DataParallel(dis_worst) 
  # find gen_worst and dis_worst
  gen_worst, dis_worst = find_worst_BCE(args, train_loader, gen, dis, gen_worst, dis_worst, g_lr_worst, d_lr_worst, gen_avg_param)
 
  outter_steps = 10
  for i in range(outter_steps): 
    # get a random minibatch from val
    input_val, _ = next(iter(val_loader))
    input_val = input_val.cuda()
    
    # get a random minibatch from train
    input_train, _ = next(iter(train_loader))
    input_train = input_train.cuda()

    dg_loss, minmax, maxmin =  architect.step(gen_worst, dis_worst, input_train, input_val, lr, gen_optimizer, unrolled=args.unrolled)
    print('debug@: duality_gap is {}'.format(dg_loss))
    print('debug@: minmax is {}'.format(minmax))
    print('debug@: maxmin is {}'.format(maxmin))
    logging.info('epoch {}: duality_gap is {}'.format(epoch, dg_loss))    
    
    # logging the weights of opr and according grad
    logging.info('epoch_{} up_1: {}'.format(epoch, gen.alphas_up_1))
    logging.info('epoch_{} up_1 architect: {}'.format(epoch, architect.model.alphas_up_1))
    logging.info('epoch_{} up_2: {}'.format(epoch, gen.alphas_up_2))
    logging.info('epoch_{} up_3: {}'.format(epoch, gen.alphas_up_3))
    logging.info('epoch_{} normal_1: {}'.format(epoch, gen.alphas_normal_1))
    logging.info('epoch_{} normal_2: {}'.format(epoch, gen.alphas_normal_2))
    logging.info('epoch_{} normal_3: {}'.format(epoch, gen.alphas_normal_3))
    logging.info('epoch_{} skip_2: {}'.format(epoch, gen.alphas_skip_2))
    logging.info('epoch_{} skip_3: {}'.format(epoch, gen.alphas_skip_3))    
    logging.info('epoch_{} up_1_grad: {}'.format(epoch, gen.alphas_up_1.grad))
    logging.info('epoch_{} up_2_grad: {}'.format(epoch, gen.alphas_up_2.grad))
    logging.info('epoch_{} up_3_grad: {}'.format(epoch, gen.alphas_up_3.grad))
    logging.info('epoch_{} normal_1_grad: {}'.format(epoch, gen.alphas_normal_1.grad))
    logging.info('epoch_{} normal_2_grad: {}'.format(epoch, gen.alphas_normal_2.grad))
    logging.info('epoch_{} normal_3_grad: {}'.format(epoch, gen.alphas_normal_3.grad))
    logging.info('epoch_{} skip_2_grad: {}'.format(epoch, gen.alphas_skip_2.grad))
    logging.info('epoch_{} skip_3_grad: {}'.format(epoch, gen.alphas_skip_3.grad))
    
    # update writer
    # writer.add_scalar('d_loss_arch', d_loss.data, val_step)
    # writer.add_scalar('g_loss_arch', g_loss.data, val_step)
    writer.add_scalar('duality_gap', dg_loss.data, val_step + args.fix_alphas_epochs)
    writer.add_scalar('minmax', minmax.data, val_step + args.fix_alphas_epochs)
    writer.add_scalar('maxmin', maxmin.data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_normal_1.shape[0]):
      for j in range(architect.model.alphas_normal_1.shape[1]):
        writer.add_scalar('cell_1/normal_edge_{}/opr_{}'.format(i, j), architect.model.alphas_normal_1[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_normal_2.shape[0]):
      for j in range(architect.model.alphas_normal_2.shape[1]):
        writer.add_scalar('cell_2/normal_edge_{}/opr_{}'.format(i, j), architect.model.alphas_normal_2[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_normal_3.shape[0]):
      for j in range(architect.model.alphas_normal_3.shape[1]):
        writer.add_scalar('cell_3/normal_edge_{}/opr_{}'.format(i, j), architect.model.alphas_normal_3[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_up_1.shape[0]):
      for j in range(architect.model.alphas_up_1.shape[1]):
        writer.add_scalar('cell_1/up_edge_{}/opr_{}'.format(i, j), architect.model.alphas_up_1[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_up_2.shape[0]):
      for j in range(architect.model.alphas_up_2.shape[1]):
        writer.add_scalar('cell_2/up_edge_{}/opr_{}'.format(i, j), architect.model.alphas_up_2[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_up_3.shape[0]):
      for j in range(architect.model.alphas_up_3.shape[1]):
        writer.add_scalar('cell_3/up_edge_{}/opr_{}'.format(i, j), architect.model.alphas_up_3[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_skip_2.shape[0]):
      for j in range(architect.model.alphas_skip_2.shape[1]):
        writer.add_scalar('skip_2/edge_{}/opr_{}'.format(i, j), architect.model.alphas_skip_2[i][j].data, val_step + args.fix_alphas_epochs)
    for i in range(architect.model.alphas_skip_3.shape[0]):
      for j in range(architect.model.alphas_skip_3.shape[1]):
        writer.add_scalar('skip_3/edge_{}/opr_{}'.format(i, j), architect.model.alphas_skip_3[i][j].data, val_step + args.fix_alphas_epochs)
    val_step += 1
    writer_dict['val_steps'] += 1

  return

def infer(valid_queue, model, criterion, writer_dict):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  # tensorboard logger
  writer = writer_dict['writer']
  val_step = writer_dict['val_steps']

  for step, (input, target) in enumerate(valid_queue):
    input = Variable(input, volatile=True).cuda()
    # target = Variable(target, volatile=True).cuda(async=True)
    target = Variable(target, volatile=True).cuda()

    logits = model(input)
    loss = criterion(logits, target)
    writer.add_scalar('val_loss', loss.data, val_step)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    writer.add_scalar('val_prec1', prec1.data, val_step)
    writer.add_scalar('val_prec5', prec5.data, val_step)
    n = input.size(0)
    objs.update(loss.data, n)
    top1.update(prec1.data, n)
    top5.update(prec5.data, n)

    val_step += 1
    writer_dict['val_steps'] += 1

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()
