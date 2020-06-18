import logging
import operator
import os
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import models.model_search_gan as model_search_gan
from imageio import imsave
from torchvision.utils import make_grid
from tqdm import tqdm

from utils.fid_score import calculate_fid_given_paths
from utils.inception_score import get_inception_score
from models.model_search_gan import Network_dis_Auto, Discriminator

class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr

class duality_gap(nn.Module):

  def __init__(self, args, gen_net, dis_net, gen_worst, dis_worst):
    super(duality_gap, self).__init__()
    self.gen = gen_net
    self.dis = dis_net
    self.gen_worst = gen_worst
    self.dis_worst = dis_worst

    self._latent_dim = args.latent_dim

  def forward(self, imgs):
    # calculate minmax
    minmax = 0.
    # Adversarial ground truths
    real_imgs = imgs.type(torch.cuda.FloatTensor)

    # Sample noise as generator input
    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], self._latent_dim)))

    real_validity = self.dis_worst(real_imgs)
    # fake_imgs = gen_net(z).detach()
    fake_imgs = self.gen(z)
    assert fake_imgs.size() == real_imgs.size()

    fake_validity = self.dis_worst(fake_imgs)

    # cal loss
    d_loss = (torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity)))
    g_loss = torch.mean(fake_validity).detach()
    minmax = -d_loss

    # calculate maxmin
    maxmin = 0.
    # Adversarial ground truths
    real_imgs = imgs.type(torch.cuda.FloatTensor)

    # Sample noise as generator input
    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], self._latent_dim)))

    real_validity = self.dis(real_imgs)
    # fake_imgs = gen_worst(z).detach()
    fake_imgs = self.gen_worst(z)
    assert fake_imgs.size() == real_imgs.size()

    fake_validity = self.dis(fake_imgs)

    # cal loss
    d_loss = (torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity)))
    g_loss = torch.mean(fake_validity).detach()
    maxmin = -d_loss

    # compute duality gap
    print('debug@: minmax is {}, maxmin is {}'.format(minmax, maxmin))
    dg = minmax - maxmin

    return dg

class duality_gap_with_mm(nn.Module):

  def __init__(self, args, gen_net, dis_net, gen_worst, dis_worst):
    super(duality_gap_with_mm, self).__init__()
    self.gen = gen_net
    self.dis = dis_net
    self.gen_worst = gen_worst
    self.dis_worst = dis_worst

    self._latent_dim = args.latent_dim

  def forward(self, imgs):
    # calculate minmax
    minmax = 0.
    # Adversarial ground truths
    real_imgs = imgs.type(torch.cuda.FloatTensor)

    # Sample noise as generator input
    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], self._latent_dim)))

    noise = torch.rand(imgs.shape).cuda()
    # real_imgs = real_imgs + noise
    real_validity = self.dis_worst(real_imgs)
    # fake_imgs = gen_net(z).detach()
    fake_imgs = self.gen(z)
    # fake_imgs = fake_imgs + noise
    assert fake_imgs.size() == real_imgs.size()

    fake_validity = self.dis_worst(fake_imgs)

    # cal loss
    d_loss = (torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity)))
    g_loss = torch.mean(fake_validity).detach()
    minmax = -d_loss

    # calculate maxmin
    maxmin = 0.
    # Adversarial ground truths
    real_imgs = imgs.type(torch.cuda.FloatTensor)

    # Sample noise as generator input
    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], self._latent_dim)))

    real_validity = self.dis(real_imgs)
    # fake_imgs = gen_worst(z).detach()
    fake_imgs = self.gen_worst(z)
    assert fake_imgs.size() == real_imgs.size()
 
    # fake_imgs = fake_imgs + noise
    fake_validity = self.dis(fake_imgs)

    # cal loss
    print('debug@: cal DG, real is {}, fake is {}'.format(torch.mean(real_validity), torch.mean(fake_validity)))    
    d_loss = (torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity)))
    g_loss = torch.mean(fake_validity).detach()
    maxmin = -d_loss

    # compute duality gap
    print('debug@: minmax is {}, maxmin is {}'.format(minmax, maxmin))
    dg = minmax - maxmin

    return dg, minmax, maxmin

class duality_gap_with_mm_wgan(nn.Module):

  def __init__(self, args, gen_net, dis_net, gen_worst, dis_worst):
    super(duality_gap_with_mm_wgan, self).__init__()
    self.gen = gen_net
    self.dis = dis_net
    self.gen_worst = gen_worst
    self.dis_worst = dis_worst

    self._latent_dim = args.latent_dim

  def forward(self, imgs):
    # calculate minmax
    minmax = 0.
    # Adversarial ground truths
    real_imgs = imgs.type(torch.cuda.FloatTensor)

    # Sample noise as generator input
    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], self._latent_dim)))

    real_validity = self.dis_worst(real_imgs)
    # fake_imgs = gen_net(z).detach()
    fake_imgs = self.gen(z)
    assert fake_imgs.size() == real_imgs.size()

    fake_validity = self.dis_worst(fake_imgs)

    # cal loss
    # d_loss = (torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
    #              torch.mean(nn.ReLU(inplace=True)(1 + fake_validity)))
    d_loss = -torch.mean(real_validity - fake_validity)
    g_loss = torch.mean(fake_validity).detach()
    minmax = -d_loss

    # calculate maxmin
    maxmin = 0.
    # Adversarial ground truths
    real_imgs = imgs.type(torch.cuda.FloatTensor)

    # Sample noise as generator input
    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], self._latent_dim)))

    real_validity = self.dis(real_imgs)
    # fake_imgs = gen_worst(z).detach()
    fake_imgs = self.gen_worst(z)
    assert fake_imgs.size() == real_imgs.size()

    fake_validity = self.dis(fake_imgs)

    # cal loss
    # d_loss = (torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
    #              torch.mean(nn.ReLU(inplace=True)(1 + fake_validity)))
    d_loss = -torch.mean(real_validity - fake_validity)
    g_loss = torch.mean(fake_validity).detach()
    maxmin = -d_loss

    # compute duality gap
    print('debug@: minmax is {}, maxmin is {}'.format(minmax, maxmin))
    dg = minmax - maxmin

    return dg, minmax, maxmin

def find_worst(args, val_loader, 
               gen_net, dis_net,
               gen_worst, dis_worst,
               g_lr_worst, d_lr_worst,
               gen_avg_param):

  # -------------------------------------------------------
  # find worst generator and discriminator for duality gap
  # return the worst generator and discriminator
  # -------------------------------------------------------

  
  # load parameters from current gen_net and dis_net
  gen_worst.load_state_dict(gen_net.state_dict())
  dis_worst.load_state_dict(dis_net.state_dict())

  # Create optimizer for worst model
  gen_worst_w_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_worst.parameters()),
                                     g_lr_worst, (args.beta1, args.beta2)) 
  dis_worst_w_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_worst.parameters()),
                                     d_lr_worst, (args.beta1, args.beta2))
  gen_worst_alphas_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, gen_worst.arch_parameters()),
                                     args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay) 
  if 'Discriminator' not in args.dis:
    dis_worst_alphas_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, dis_worst.arch_parameters()),
                                       args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
  # find worst discriminator
  for iter_idx, (imgs, _) in enumerate(tqdm(val_loader)):
    

    # Adversarial ground truths
    real_imgs = imgs.type(torch.cuda.FloatTensor)

    # Sample noise as generator input
    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

    # ---------------------
    #  Train Discriminator
    #  Find worst Discriminator
    # ---------------------
    dis_worst_w_optim.zero_grad()
    if 'Discriminator' not in args.dis and args.only_update_alpha:
      dis_worst_alphas_optim.zero_grad()

    noise = torch.rand(real_imgs.shape).cuda()
    # real_imgs = real_imgs + noise
    real_validity = dis_worst(real_imgs)
    fake_imgs = gen_net(z).detach()
    assert fake_imgs.size() == real_imgs.size()

    # fake_imgs = fake_imgs + noise
    fake_validity = dis_worst(fake_imgs)

    # cal loss
    d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
    d_loss.backward()
    dis_worst_w_optim.step()
    if 'Discriminator' not in args.dis and args.only_update_alpha:
      dis_worst_alphas_optim.step()

    if iter_idx > args.worst_steps:
      break

  # -----------------
  #  Train Generator
  #  Find worst Generator
  # -----------------    
  for iter_idx, (imgs, _) in enumerate(tqdm(val_loader)):
    if args.only_update_w_g:
      gen_worst_w_optim.zero_grad()
    if args.only_update_alpha_g:
      gen_worst_alphas_optim.zero_grad()

    gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
    gen_imgs = gen_worst(gen_z)
    noise = torch.rand(gen_imgs.shape).cuda()
    # gen_imgs = gen_imgs + noise
    fake_validity = dis_net(gen_imgs)

    # cal loss
    g_loss = -torch.mean(fake_validity)
    # g_loss = -torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
    g_loss.backward()
    if args.only_update_alpha_g:
      gen_worst_alphas_optim.step()
    if args.only_update_w_g:
      gen_worst_w_optim.step()
            
    # moving average weight
    for p, avg_p in zip(gen_worst.parameters(), gen_avg_param):
      avg_p.mul_(0.999).add_(0.001, p.data)

    if iter_idx > args.worst_steps:
      break

  return gen_worst, dis_worst

def compute_d_loss(real_validity, fake_validity, loss_type):
    """
    compute d_loss according to loss_type
    """
    if loss_type == 'hinge_loss':
        d_loss = torch.mean(nn.ReLU(inplace=True)(1-real_validity) + 
                           nn.ReLU(inplace=True)(1+fake_validity))
    elif loss_type == 'wgan':
        d_loss = torch.mean(fake_validity) - torch.mean(real_validity)
    elif loss_type == 'BCE':
        one = torch.ones(real_validity.shape[0], 1).cuda()
        zero = one * 0.
        d_loss = nn.BCEWithLogitsLoss()(real_validity, one) + nn.BCEWithLogitsLoss()(fake_validity, zero)
    return d_loss

def compute_g_loss(fake_validity, loss_type):
    """
    compute g_loss according to loss_type
    """
    if loss_type == 'hinge_loss':
        g_loss = -torch.mean(fake_validity)
    elif loss_type == 'wgan':
        g_loss = -torch.mean(fake_validity)
    elif loss_type == 'BCE':
        one = torch.ones(fake_validity.shape[0], 1).cuda()
        g_loss = nn.BCEWithLogitsLoss()(fake_validity, one)
    return g_loss



def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader,
          epoch, writer_dict, consistent=False, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()

    
    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        # generate noise added to img and gen_img
        fixed_z = torch.rand(imgs.shape).cuda()

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()
        
        # label for discriminator
        one = torch.ones([real_imgs.shape[0], 1]).cuda()
        zero = one * 0
        mone = one * -1

        # real_imgs = real_imgs + fixed_z
        real_validity = dis_net(real_imgs)
        # real_validity.backward(one)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        # fake_imgs = fake_imgs + fixed_z
        fake_validity = dis_net(fake_imgs)
        # fake_validity.backward(mone)        

        # clamp parameters to a cube
        if args.loss_type == 'wgan':
            for p in dis_net.parameters():
                p.data.clamp_(-0.01, 0.01)

        # cal loss
        # d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
        #          torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss = compute_d_loss(real_validity, fake_validity, args.loss_type)
        print('debug@: real_validity is {}, fake_validity is {}'.format(torch.mean(real_validity), torch.mean(fake_validity)))
        # d_loss.backward()
        # d_loss = -torch.mean(real_validity - fake_validity)
        # d_loss = 0.5 * nn.MSELoss()(real_validity, one) + 0.5 * nn.MSELoss()(fake_validity, torch.zeros([1]).cuda())
        # d_loss = nn.BCEWithLogitsLoss()(real_validity, one) + nn.BCEWithLogitsLoss()(fake_validity, zero)
        d_loss.backward()
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()
            
            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            if fixed_z.shape[0] != args.gen_batch_size:
                fixed_z = torch.rand(gen_imgs.shape).cuda()
            # gen_imgs = gen_imgs + fixed_z
            fake_validity = dis_net(gen_imgs)
            
            # label for discriminator
            one = torch.ones([gen_z.shape[0], 1]).cuda()
            mone = one * -1

            # cal loss
            # if not consistent:
            #     g_loss = -torch.mean(fake_validity)
            # else:
            #     g_loss = -torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
            g_loss = compute_g_loss(fake_validity, args.loss_type)
            # g_loss = nn.BCEWithLogitsLoss()(fake_validity, one)
            # g_loss = 0.5 * nn.MSELoss()(fake_validity, one)
            g_loss.backward()
            # fake_validity.backward(one)
            gen_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            print('debug@: g_loss is {}, d_loss is {}'.format(g_loss.data, d_loss.data))
            
            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1

def train_gan_parameter(args, train_queue, gen, dis, 
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
    if step > args.inner_steps:
      break

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
  if 'Discriminator' not in args.dis:
    dis_worst = dis.copy_alphas(dis_worst)
  dis_worst.set_gumbel(False)
  gen_worst.cur_stage = gen.cur_stage
  dis_worst.cur_stage = dis.cur_stage
  dis_worst = dis_worst.cuda()
  if args.parallel:
    gen_worst = nn.DataParallel(gen_worst)
    dis_worst = nn.DataParallel(dis_worst) 
  # find gen_worst and dis_worst
  gen_worst, dis_worst = find_worst(args, val_loader, gen, dis, gen_worst, dis_worst, g_lr_worst, d_lr_worst, gen_avg_param)
  
  outter_steps = args.outter_steps
  for i in range(outter_steps): 
    # get a random minibatch from 'val'
    input_val, _ = next(iter(val_loader))
    input_val = input_val.cuda()
    
    # get a random minibatch from train
    input_train, _ = next(iter(train_loader))
    input_train = input_train.cuda()

    dg_loss, minmax, maxmin = architect.step(gen_worst, dis_worst, input_train, input_val, lr, gen_optimizer, unrolled=args.unrolled)
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
    if 'Discriminator' not in args.dis:
      for i in range(architect.dis.alphas_normal_channels_raise.shape[0]):
        for j in range(architect.dis.alphas_normal_channels_raise.shape[1]):
          writer.add_scalar('dis/cell_1_raise/normal_edge_{}/opr_{}'.format(i, j), architect.dis.alphas_normal_channels_raise[i][j].data, val_step + args.fix_alphas_epochs)
      for i in range(architect.dis.alphas_normal_1.shape[0]):
        for j in range(architect.dis.alphas_normal_1.shape[1]):
          writer.add_scalar('dis/cell_1/normal_edge_{}/opr_{}'.format(i, j), architect.dis.alphas_normal_1[i][j].data, val_step + args.fix_alphas_epochs)
      for i in range(architect.dis.alphas_normal_2.shape[0]):
        for j in range(architect.dis.alphas_normal_2.shape[1]):
          writer.add_scalar('dis/cell_2/normal_edge_{}/opr_{}'.format(i, j), architect.dis.alphas_normal_2[i][j].data, val_step + args.fix_alphas_epochs)
      for i in range(architect.dis.alphas_normal_3.shape[0]):
        for j in range(architect.dis.alphas_normal_3.shape[1]):
          writer.add_scalar('dis/cell_3/normal_edge_{}/opr_{}'.format(i, j), architect.dis.alphas_normal_3[i][j].data, val_step + args.fix_alphas_epochs)
      for i in range(architect.dis.alphas_normal_4.shape[0]):
        for j in range(architect.dis.alphas_normal_4.shape[1]):
          writer.add_scalar('dis/cell_4/normal_edge_{}/opr_{}'.format(i, j), architect.dis.alphas_normal_4[i][j].data, val_step + args.fix_alphas_epochs)
    val_step += 1
    writer_dict['val_steps'] += 1

  return

def copy_params(model):
  flatten = deepcopy(list(p.data for p in model.parameters()))
  return flatten

def validate(args, fixed_z, fid_stat, gen_net: nn.Module, writer_dict, path_helper, clean_dir=True, search=True):
  writer = writer_dict['writer']
  global_steps = writer_dict['valid_global_steps']

  # eval mode
  gen_net = gen_net.eval()

  # generate images
  if search:
    sample_imgs = gen_net(fixed_z, True)
  else:
    sample_imgs = gen_net(fixed_z)
  img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

  # get fid and inception score
  fid_buffer_dir = os.path.join(path_helper['sample_path'], 'fid_buffer')
  os.makedirs(fid_buffer_dir, exist_ok=True)

  eval_iter = args.num_eval_imgs // args.eval_batch_size
  img_list = list()
  for iter_idx in tqdm(range(eval_iter), desc='sample images'):
    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

    # Generate a batch of images
    gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
                                                                                                torch.uint8).numpy()
    for img_idx, img in enumerate(gen_imgs):
      file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
      imsave(file_name, img)
    img_list.extend(list(gen_imgs))

  # get inception score
  mean, std = get_inception_score(img_list)

  # get fid score
  fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None)

  if clean_dir:
    os.system('rm -r {}'.format(fid_buffer_dir))
  else:
    logger.info(f'=> sampled images are saved to {fid_buffer_dir}')

  writer.add_image('sampled_images', img_grid, global_steps)
  writer.add_scalar('Inception_score/mean', mean, global_steps)
  writer.add_scalar('Inception_score/std', std, global_steps)
  writer.add_scalar('FID_score', fid_score, global_steps)

  writer_dict['valid_global_steps'] = global_steps + 1

  return mean, std, fid_score

class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten

def grow_ctrl(epoch, grow_epoch):
    if epoch < grow_epoch[0]:
      return 0
    elif epoch < grow_epoch[1]:
      return 1
    else:
      return 2
