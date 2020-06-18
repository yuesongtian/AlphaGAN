import torch
import numpy as np
import torch.nn as nn
from torch.autograd import Variable

import functions
from functions import duality_gap, duality_gap_with_mm

def _concat(xs):
  return torch.cat([x.view(-1) for x in xs])


class Architect_gen(object):

  def __init__(self, model, dis, args, dg_which_loss, logging):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.dis = dis
    self.gen_optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    if 'Discriminator' not in args.dis:
      self.dis_optimizer = torch.optim.Adam(self.dis.arch_parameters(),
          lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    self.args = args
    self._dg_which_loss = dg_which_loss
    self._logging = logging

  def _compute_unrolled_model(self, gen_worst, dis_worst, imgs, eta, network_optimizer):
    loss, minmax, maxmin = self._dg_loss(self.model, self.dis, gen_worst, dis_worst, imgs)
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    self._logging.info('debug: moment is {}'.format(moment))
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    self._logging.info('debug: dtheta is {}'.format(dtheta - self.network_weight_decay * theta))
    # dtheta = self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, dtheta))
    return unrolled_model

  def step(self, gen_worst, dis_worst, imgs_train, imgs_val, eta, network_optimizer, unrolled):
    self.gen_optimizer.zero_grad()
    if 'Discriminator' not in self.args.dis:
      self.dis_optimizer.zero_grad()
    if unrolled:
        loss, minmax, maxmin = self._backward_step_unrolled(gen_worst, dis_worst, imgs_train, imgs_val, eta, network_optimizer)
    else:
        if self.args.use_train_val:
            loss, minmax, maxmin = self._backward_step_mix_trainval(gen_worst, dis_worst, imgs_train, imgs_val)
        else:
            loss, minmax, maxmin = self._backward_step(gen_worst, dis_worst, imgs_val)
    self.gen_optimizer.step()
    if 'Discriminator' not in self.args.dis:
      self.dis_optimizer.step()

    # return d_loss.detach(), g_loss.detach()
    return loss.detach(), minmax.detach(), maxmin.detach()

  def _backward_step(self, gen_worst, dis_worst, imgs):
    dg_loss, minmax, maxmin = self._dg_loss(self.model, self.dis, gen_worst, dis_worst, imgs)
    # d_loss.backward()
    # g_loss.backward()
    dg_loss.backward()

    return dg_loss, minmax, maxmin

  def _backward_step_mix_trainval(self, gen_worst, dis_worst, imgs_tr, imgs_val):
    dg_loss_tr, minmax_tr, maxmin_tr = self._dg_loss(self.model, self.dis, gen_worst, dis_worst, imgs_tr)
    dg_loss_val, minmax_val, maxmin_val = self._dg_loss(self.model, self.dis, gen_worst, dis_worst, imgs_val)
    # d_loss.backward()
    # g_loss.backward()
    dg_loss = dg_loss_tr + self.args.lamina * dg_loss_val
    minmax = minmax_tr + self.args.lamina * minmax_val
    maxmin = maxmin_tr + self.args.lamina * maxmin_val
    dg_loss.backward()

    return dg_loss, minmax, maxmin

  def _backward_step_unrolled(self, gen_worst, dis_worst, imgs_train, imgs_val, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(gen_worst, dis_worst, imgs_train, eta, network_optimizer)
    unrolled_loss, minmax, maxmin = self._dg_loss(unrolled_model, self.dis, gen_worst, dis_worst, imgs_val)
    loss, minmax_, maxmin_ = self._dg_loss(self.model, self.dis, gen_worst, dis_worst, imgs_val)
    self._logging.info('debug@: _backward_step_unrolled unrolled_loss is {}, minmax is {}, maxmin is {}'.format(unrolled_loss, minmax, maxmin))
    self._logging.info('debug@: _backward_step_unrolled loss is {}, minmax is {}, maxmin is {}'.format(loss, minmax_, maxmin_))

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    loss.backward()
    dalpha_common = [v.grad for v in self.model.arch_parameters()]
    vector_common = [v.grad.data for v in self.model.parameters()]
    self._logging.info('debug@: dalpha is {}'.format(dalpha))
    self._logging.info('debug@: vector is {}'.format(vector))
    self._logging.info('debug@: dalpha_common is {}'.format(dalpha_common))
    self._logging.info('debug@: vector_common is {}'.format(vector_common))
    implicit_grads = self._hessian_vector_product(vector, gen_worst, dis_worst, imgs_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

    return unrolled_loss, minmax, maxmin

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

  
    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      # params[k] = torch.rand(v.size())
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
     
   
    return model_new.cuda()

  def _hessian_vector_product(self, vector, gen_worst, dis_worst, imgs, r=1e-2):
    R = r / _concat(vector).norm()
    print('debug@: _hessian_vector, R is {}'.format(R))
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self._dg_loss(self.model, self.dis, gen_worst, dis_worst, imgs)
    print('debug@: in hessian_vector_product, loss_1 is {}'.format(loss))
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self._dg_loss(self.model, self.dis, gen_worst, dis_worst, imgs)
    print('debug@: in hessian_vector_product, loss_2 is {}'.format(loss))
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

  def _dg_loss(self, gen, dis, gen_worst, dis_worst, imgs):
    dg = eval('functions'+'.'+self._dg_which_loss)(self.args, gen, dis, gen_worst, dis_worst)
    loss, minmax, maxmin = dg(imgs)

    # d_loss, g_loss = self._common_loss(gen, dis, imgs)
    
    return loss, minmax, maxmin

  def _common_loss(self, gen, dis, imgs):
    # Adversarial ground truths
    real_imgs = imgs.type(torch.cuda.FloatTensor)

    # Sample noise as generator input
    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], self.args.latent_dim)))

    real_validity = dis(real_imgs)
    # fake_imgs = gen_worst(z).detach()
    fake_imgs = gen(z)
    assert fake_imgs.size() == real_imgs.size()

    fake_validity = dis(fake_imgs)

    d_loss = torch.mean(nn.ReLU(inplace=True)(1 - real_validity)) + \
             torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
    
    # Sample noise as generator input
    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (self.args.gen_batch_size, self.args.latent_dim)))
    fake_imgs_g = gen(z)
    fake_validity_g = dis(fake_imgs_g)
    g_loss = -torch.mean(fake_validity_g)

    return d_loss, g_loss

class Architect_gen_fix_d(object):

  def __init__(self, model, dis, args):
    self.network_momentum = args.momentum
    self.network_weight_decay = args.weight_decay
    self.model = model
    self.dis = dis
    self.gen_optimizer = torch.optim.Adam(self.model.arch_parameters(),
        lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    # self.dis_optimizer = torch.optim.Adam(self.dis.arch_parameters(),
    #     lr=args.arch_learning_rate, betas=(0.5, 0.999), weight_decay=args.arch_weight_decay)
    self.args = args

  def _compute_unrolled_model(self, gen_worst, dis_worst, imgs, eta, network_optimizer):
    loss = self._dg_loss(self.model, self.dis, gen_worst, dis_worst, imgs)
    theta = _concat(self.model.parameters()).data
    try:
      moment = _concat(network_optimizer.state[v]['momentum_buffer'] for v in self.model.parameters()).mul_(self.network_momentum)
    except:
      moment = torch.zeros_like(theta)
    dtheta = _concat(torch.autograd.grad(loss, self.model.parameters())).data + self.network_weight_decay*theta
    unrolled_model = self._construct_model_from_theta(theta.sub(eta, moment+dtheta))
    return unrolled_model

  def step(self, gen_worst, dis_worst, imgs_train, imgs_val, eta, network_optimizer, unrolled):
    self.gen_optimizer.zero_grad()
    # self.dis_optimizer.zero_grad()
    if unrolled:
        loss = self._backward_step_unrolled(gen_worst, dis_worst, imgs_train, imgs_val, eta, network_optimizer)
    else:
        loss, minmax, maxmin = self._backward_step(gen_worst, dis_worst, imgs_val)
    self.gen_optimizer.step()
    # self.dis_optimizer.step()

    # return d_loss.detach(), g_loss.detach()
    return loss.detach(), minmax.detach(), maxmin.detach()

  def _backward_step(self, gen_worst, dis_worst, imgs):
    dg_loss, minmax, maxmin = self._dg_loss(self.model, self.dis, gen_worst, dis_worst, imgs)
    # d_loss.backward()
    # g_loss.backward()
    dg_loss.backward()

    return dg_loss, minmax, maxmin

  def _backward_step_unrolled(self, gen_worst, dis_worst, imgs_train, imgs_val, eta, network_optimizer):
    unrolled_model = self._compute_unrolled_model(gen_worst, dis_worst, imgs_train, eta, network_optimizer)
    unrolled_loss = self._dg_loss(unrolled_model, self.dis, gen_worst, dis_worst, imgs_val)
    print('debug@: _backward_step_unrolled unrolled_loss is {}'.format(unrolled_loss))

    unrolled_loss.backward()
    dalpha = [v.grad for v in unrolled_model.arch_parameters()]
    vector = [v.grad.data for v in unrolled_model.parameters()]
    print('debug@: dalpha is {}'.format(dalpha))
    print('debug@: vector is {}'.format(vector))
    implicit_grads = self._hessian_vector_product(vector, gen_worst, dis_worst, imgs_train)

    for g, ig in zip(dalpha, implicit_grads):
      g.data.sub_(eta, ig.data)

    for v, g in zip(self.model.arch_parameters(), dalpha):
      if v.grad is None:
        v.grad = Variable(g.data)
      else:
        v.grad.data.copy_(g.data)

    return unrolled_loss

  def _construct_model_from_theta(self, theta):
    model_new = self.model.new()
    model_dict = self.model.state_dict()

    params, offset = {}, 0
    for k, v in self.model.named_parameters():
      v_length = np.prod(v.size())
      params[k] = theta[offset: offset+v_length].view(v.size())
      print('debug@: in _construct_model_from_theta, params[k] is {}'.format(params[k]))
      offset += v_length

    assert offset == len(theta)
    model_dict.update(params)
    model_new.load_state_dict(model_dict)
       
    return model_new.cuda()

  def _hessian_vector_product(self, vector, gen_worst, dis_worst, imgs, r=1e-2):
    R = r / _concat(vector).norm()
    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)
    loss = self._dg_loss(self.model, self.dis, gen_worst, dis_worst, imgs)
    grads_p = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.sub_(2*R, v)
    loss = self._dg_loss(self.model, self.dis, gen_worst, dis_worst, imgs)
    grads_n = torch.autograd.grad(loss, self.model.arch_parameters())

    for p, v in zip(self.model.parameters(), vector):
      p.data.add_(R, v)

    return [(x-y).div_(2*R) for x, y in zip(grads_p, grads_n)]

  def _dg_loss(self, gen, dis, gen_worst, dis_worst, imgs):
    dg = duality_gap_with_mm(self.args, gen, dis, gen_worst, dis_worst)
    loss = dg(imgs)

    # d_loss, g_loss = self._common_loss(gen, dis, imgs)
    
    return loss

  def _common_loss(self, gen, dis, imgs):
    # Adversarial ground truths
    real_imgs = imgs.type(torch.cuda.FloatTensor)

    # Sample noise as generator input
    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], self.args.latent_dim)))

    real_validity = dis(real_imgs)
    # fake_imgs = gen_worst(z).detach()
    fake_imgs = gen(z)
    assert fake_imgs.size() == real_imgs.size()

    fake_validity = dis(fake_imgs)

    d_loss = torch.mean(nn.ReLU(inplace=True)(1 - real_validity)) + \
             torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
    
    # Sample noise as generator input
    z = torch.cuda.FloatTensor(np.random.normal(0, 1, (self.args.gen_batch_size, self.args.latent_dim)))
    fake_imgs_g = gen(z)
    fake_validity_g = dis(fake_imgs_g)
    g_loss = -torch.mean(fake_validity_g)

    return d_loss, g_loss
