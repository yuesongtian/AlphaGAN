import torch
import torch.nn as nn
import torch.nn.functional as F
from models.operations_gan import *
import models.genotypes as genotypes

class MixedOp(nn.Module):

  def __init__(self, C_in, C_out, primitive_list, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in primitive_list:
      op = OPS[primitive](C_in, C_out, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights, eval=False, gene=None):
    if not eval:
      return sum(w * op(x) for w, op in zip(weights, self._ops))
    else:
      return sum(w * op(x) * discre for w, op, discre in zip(weights, self._ops, gene))

class MixedOp_first(nn.Module):

  def __init__(self, C_in, C_out, stride, with_bn):
    super(MixedOp_first, self).__init__()
    self._ops = nn.ModuleList()
    if with_bn:
      PRIMITIVES = ['conv_1x1_bn', 'conv_3x3_bn', 'conv_5x5_bn']
    else:
      PRIMITIVES = ['conv_1x1', 'conv_3x3', 'conv_5x5']
    for primitive in PRIMITIVES:
      op = OPS[primitive](C_in, C_out, stride, False)
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))

class MixedOp_up(nn.Module):

  def __init__(self, C, primitives, stride=2):
    super(MixedOp_up, self).__init__()
    self._ops = nn.ModuleList()
    for primitive in primitives:
      op = OPS[primitive](C, C, stride, False)
      self._ops.append(op)

  def forward(self, x, weights, eval=False, gene=None):
    if not eval:
      return sum(w * op(x) for w, op in zip(weights, self._ops))
    else:
      return sum(w * op(x) * discre for w, op, discre in zip(weights, self._ops, gene))

class MixedOp_down(nn.Module):

  def __init__(self, C, stride=2, skip=False):
    super(MixedOp_down, self).__init__()
    self._ops = nn.ModuleList()
    if skip:
      op = OPS['none'](C, stride, False)
      self._ops.append(op)
    for primitive in PRIMITIVES_DOWN:
      op = OPS[primitive](C, stride, False)
      self._ops.append(op)

  def forward(self, x, weights):
    return sum(w * op(x) for w, op in zip(weights, self._ops))


class Cell_dis_Auto(nn.Module):
    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            use_gumbel,
            hidden_channels=None,
            ksize=3,
            pad=1,
            activation=nn.ReLU(),
            downsample=False):
        super(Cell_dis_Auto, self).__init__()
        self._use_gumbel = use_gumbel
        self._t = args.t
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        
        PRIMITIVES = eval('genotypes.' + args.dis_normal_opr)
        self.c1 = MixedOp(in_channels, out_channels, PRIMITIVES, 1)
        self.c2 = MixedOp(out_channels, out_channels, PRIMITIVES, 1)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0)
            if args.d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x, weights):
        h = x
        h = self.activation(h)
        if self._use_gumbel:
          a = calculate_gumbel_softmax(weights[0], self._t)
        else:
          a = F.softmax(weights[0], -1)
        h = self.c1(h, a)
        h = self.activation(h)
        if self._use_gumbel:
          a = calculate_gumbel_softmax(weights[1], self._t)
        else:
          a = F.softmax(weights[1], -1)
        h = self.c2(h, a)
        if self.downsample:
            h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return _downsample(x)
            else:
                return x
        else:
            return x

    def forward(self, x, weights):
        return self.residual(x, weights) + self.shortcut(x)

class Cell_dis_Auto_first(nn.Module):
    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            use_gumbel,
            hidden_channels=None,
            ksize=3,
            pad=1,
            activation=nn.ReLU()):
        super(Cell_dis_Auto_first, self).__init__()
        self.activation = activation
        self.learnable_sc = (in_channels != out_channels)
        hidden_channels = in_channels if hidden_channels is None else hidden_channels
        self._use_gumbel = use_gumbel
        self._t = args.t

        self.c1 = MixedOp_first(in_channels, out_channels, 1, args.dis_with_bn)
        PRIMITIVES = eval('genotypes.' + args.dis_normal_opr)  
        self.c2 = MixedOp(out_channels, out_channels, PRIMITIVES, 1)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0)
            if args.d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x, weights_normal, weights_channels_raise):
        h = x
        h = self.activation(h)
        if self._use_gumbel:
          a = calculate_gumbel_softmax(weights_channels_raise[0], self._t)
        else:
          a = weights_channels_raise[0]
        h = self.c1(h, a)
        h = self.activation(h)
        if self._use_gumbel:
          a = calculate_gumbel_softmax(weights_normal[0], self._t)
        else:
          a = weights_normal[0]
        h = self.c2(h, a)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            return _downsample(x)
        else:
            return x

    def forward(self, x, weights_normal, weights_channels_raise):
        return self.residual(x, weights_normal, weights_channels_raise) + self.shortcut(x)

class Cell_gen_Auto(nn.Module):
  """
  Architecture of cell in Auto-GAN paper
  """
  def __init__(self, args, C_in, C_out, use_gumbel, prev=False, prev_prev=False):
    super(Cell_gen_Auto, self).__init__()
    self._use_gumbel = use_gumbel    

    self._mixed_ops_prev = nn.ModuleList()
    self._prev_prev = prev_prev
    self._t = args.t
    primitives_up = eval('genotypes.' + args.gen_up_opr)
    if prev_prev:
      op = MixedOp_up(C_in, primitives_up, stride=4)
      self._mixed_ops_prev.append(op)
      print('debug@: length of skip_up_op is {}'.format(len(op._ops)))
      op = MixedOp_up(C_in, primitives_up, stride=2)
      self._mixed_ops_prev.append(op)
      print('debug@: length of skip_up_op is {}'.format(len(op._ops)))
    self._prev = prev
    if prev:
      op = MixedOp_up(C_in, primitives_up, stride=2)
      self._mixed_ops_prev.append(op)
      print('debug@: length of skip_up_op is {}'.format(len(op._ops)))

    self._up_ops = nn.ModuleList()
    for i in range(2):
      op = MixedOp_up(C_in, primitives_up, stride=2)
      self._up_ops.append(op)

    self._normal_ops = nn.ModuleList()
    for i in range(3):
      primitives = eval('genotypes.' + args.gen_normal_opr)
      op = MixedOp(C_in, C_out, primitives, 1)
      self._normal_ops.append(op)
    
  def calculate_ops(self):
    len_up_ops = len(self._up_ops)
    len_norm_ops = len(self._normal_ops)
    len_skip_ops = len(self._mixed_ops_prev)

    return len_up_ops, len_norm_ops, len_skip_ops

  def forward(self, x, weights_up, weights_norm, weights_skip=None, prev_ft=None, eval=False, gene=None, gene_skip=None):
    torch.autograd.set_detect_anomaly(True)
    # handle the skip features
    skip_ft = []
    if self._prev_prev:
      assert len(prev_ft) == 2
      assert len(weights_skip) == 2
      for i in range(len(weights_skip)):
        if self._use_gumbel:
          a = calculate_gumbel_softmax(weights_skip[i], self._t)
        else:
          a = F.softmax(weights_skip[i], dim=-1)
        if eval:
          ft = self._mixed_ops_prev[i](prev_ft[i], a, True, gene_skip[i])
        else:
          ft = self._mixed_ops_prev[i](prev_ft[i], a)
        skip_ft.append(ft)
    if self._prev:
      assert len(prev_ft) == 1
      assert len(weights_skip) == 1
      for i in range(len(weights_skip)):
        if self._use_gumbel:
          a = calculate_gumbel_softmax(weights_skip[i], self._t)
        else:
          a = F.softmax(weights_skip[i], dim=-1)
        if eval:
          ft = self._mixed_ops_prev[i](prev_ft[i], a, True, gene_skip[i])
        else:
          ft = self._mixed_ops_prev[i](prev_ft[i], a)
        skip_ft.append(ft)
    skip_ft = sum(ft for ft in skip_ft)

    # upsample the feature
    # the output feature of node 1 and node 3 (short cut)
    # ft_up = [ft_1, ft_3(short cut)]
    ft_up = []
    for i in range(len(self._up_ops)):
      if self._use_gumbel:
        a = calculate_gumbel_softmax(weights_up[i], self._t)
      else:
        a = F.softmax(weights_up[i], dim=-1)
      if gene != None:
        ft = self._up_ops[i](x, a, True, gene['up'][i]) 
      else:
        ft = self._up_ops[i](x, a) 
      ft_up.append(ft)
    ft_1 = ft_up[0]
    ft_3 = ft_up[1]

    # norm operation
    # calculate the output feature of node 2
    if self._use_gumbel:
      a = calculate_gumbel_softmax(weights_norm[0], self._t)
    else:
      a = F.softmax(weights_norm[0], dim=-1)
    if gene != None:
      ft_2 = self._normal_ops[0](ft_1, a, True, gene['normal'][0])
    else:
      ft_2 = self._normal_ops[0](ft_1, a)
    
    # norm operation
    # calculate the right input feature of node 4
    if self._use_gumbel:
      a = calculate_gumbel_softmax(weights_norm[1], self._t)
    else:
      a = F.softmax(weights_norm[1], dim=-1)
    if gene != None:
      ft_4_right = self._normal_ops[1](ft_2, a, True, gene['normal'][1])
    else:
      ft_4_right = self._normal_ops[1](ft_2, a)
    
    # norm operation (short cut)
    # calculate the left input feature of node 4
    if self._use_gumbel:
      a = calculate_gumbel_softmax(weights_norm[2], self._t)
    else:
      a = F.softmax(weights_norm[2], dim=-1)
    if gene != None:
      ft_4_left = self._normal_ops[2](ft_3, a, True, gene['normal'][2])
    else:
      ft_4_left = self._normal_ops[2](ft_3, a)
 
    ft_4 = ft_4_left + ft_4_right

    # add the skip feature from prev cell
    if self._prev or self._prev_prev:
      ft_4 = ft_4 + skip_ft

    return ft_1, ft_4

def calculate_gumbel_softmax(weights, t):
  noise = torch.rand(weights.shape).cuda()
  y = (weights - torch.log(-torch.log(noise + 1e-20))) / t
  out = F.softmax(y, dim=-1)
  return out