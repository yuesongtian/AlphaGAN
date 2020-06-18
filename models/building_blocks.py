

from torch import nn
import torch.nn.functional as F

import models.operations_gan as operations_gan
from models.operations_gan import *

UP_MODES = ['nearest', 'bilinear', 'deconv']

class Cell(nn.Module):
    def __init__(self, in_channels, out_channels, up_mode, ksize=3, num_skip_in=0, short_cut=False, norm=None):
        super(Cell, self).__init__()
        self.c1 = nn.Conv2d(in_channels, out_channels, ksize, padding=ksize//2)
        self.c2 = nn.Conv2d(out_channels, out_channels, ksize, padding=ksize//2)
        self.deconv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2)
        assert up_mode in UP_MODES
        self.up_mode = up_mode
        self.norm = norm
        if norm:
            assert norm in NORMS
            if norm == 'bn':
                self.n1 = nn.BatchNorm2d(in_channels)
                self.n2 = nn.BatchNorm2d(out_channels)
            elif norm == 'in':
                self.n1 = nn.InstanceNorm2d(in_channels)
                self.n2 = nn.InstanceNorm2d(out_channels)
            else:
                raise NotImplementedError(norm)

        # inner shortcut
        self.c_sc = None
        if short_cut:
            self.c_sc = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        # cross scale skip
        self.skip_in_ops = None
        if num_skip_in:
            self.skip_in_ops = nn.ModuleList([nn.Conv2d(out_channels, out_channels, kernel_size=1) for _ in range(num_skip_in)])

    def forward(self, x, skip_ft=None):
        residual = x

        # first conv
        if self.norm:
            residual = self.n1(residual)
        h = nn.ReLU()(residual)
        if self.up_mode == 'deconv':
            h = self.deconv(h)
        else:
            h = F.interpolate(h, scale_factor=2, mode=self.up_mode)
        _, _, ht, wt = h.size()
        h = self.c1(h)
        h_skip_out = h

        # second conv
        if self.skip_in_ops:
            assert len(self.skip_in_ops) == len(skip_ft)
            for ft, skip_in_op in zip(skip_ft, self.skip_in_ops):
                if self.up_mode == 'deconv':
                    expand_num = int(wt / ft.size()[-1] / 2)
                    deconv_list = nn.ModuleList([self.deconv for _ in range(expand_num)])
                    for oper in deconv_list:
                        ft = oper(ft)
                    h += skip_in_op(ft)
                else:
                    h += skip_in_op(F.interpolate(ft, size=(ht, wt), mode=self.up_mode))
        if self.norm:
            h = self.n2(h)
        h = nn.ReLU()(h)
        final_out = self.c2(h)

        # shortcut
        if self.c_sc:
            if self.up_mode == 'deconv':
                final_out += self.c_sc(self.deconv(x))
            else:
                final_out += self.c_sc(F.interpolate(x, scale_factor=2, mode=self.up_mode))

        return h_skip_out, final_out


class Cell_gen_Auto(nn.Module):
  """
  Architecture of cell in AutoGAN paper
  """
  def __init__(self, C_in, C_out, genotype, cur_stage, prev=False, prev_prev=False):
    super(Cell_gen_Auto, self).__init__()
    
    self._prev_prev = prev_prev 
    self._prev = prev

    if prev or prev_prev:
      self._mixed_ops_prev = nn.ModuleList()
    self._up_ops = nn.ModuleList() 
    self._normal_ops = nn.ModuleList()

    op_names_up, indices_up = zip(*(genotype.up[cur_stage]))
    print('debug@: genotype.up is {}'.format(genotype.up[cur_stage]))
    # op_names_up, indices_up = zip(*(genotype.up['1']))
    self._compile(C_in, self._up_ops, op_names_up, indices_up, up=True)
    op_names_normal, indices_normal = zip(*(genotype.normal[cur_stage]))
    # op_names_normal, indices_normal = zip(*(genotype.normal['1']))
    self._compile(C_in, self._normal_ops, op_names_normal, indices_normal)
    if prev:
      op_names_skip, indices_skip = zip(*(genotype.skip_2))
      self._compile(C_in, self._mixed_ops_prev, op_names_skip, indices_skip, skip=True)
    elif prev_prev:
      op_names_skip, indices_skip = zip(*(genotype.skip_3))
      self._compile(C_in, self._mixed_ops_prev, op_names_skip, indices_skip, skip=True)
  
  def _compile(self, C, op_list, op_names, indices, up=False, skip=False):
    assert len(op_names) == len(indices)

    for name, index in zip(op_names, indices):
      stride = 2 if up or skip else 1
      if skip and index == 0 and len(indices) == 2:
        stride = 4                        # From cell 1 to cell 3
      op = OPS[name](C, C, stride, True)
      op_list += [op]
    self._indices = indices
  
  def forward(self, x, prev_ft=None):
    # handle the skip features
    skip_ft = []
    if self._prev_prev:
      assert len(prev_ft) == 2
      for i in range(len(self._mixed_ops_prev)):
        ft = self._mixed_ops_prev[i](prev_ft[i])
        skip_ft.append(ft)
    if self._prev:
      assert len(prev_ft) == 1
      for i in range(len(self._mixed_ops_prev)):
        ft = self._mixed_ops_prev[i](prev_ft[i])
        skip_ft.append(ft)
    skip_ft = sum(ft for ft in skip_ft)

    # upsample the feature
    # the output feature of node 1 and node 3 (short cut)
    # ft_up = [ft_1, ft_3(short cut)]
    ft_up = []
    for i in range(len(self._up_ops)):
      ft = self._up_ops[i](x)
      ft_up.append(ft)
    ft_1 = ft_up[0]
    ft_3 = ft_up[1]

    # norm operation
    # calculate the output feature of node 2
    ft_2 = self._normal_ops[0](ft_1)
    
    # norm operation
    # calculate the right input feature of node 4
    ft_4_right = self._normal_ops[1](ft_2)
    
    # norm operation (short cut)
    # calculate the left input feature of node 4
    ft_4_left = self._normal_ops[2](ft_3)

    # calculate the feature of node 4
    ft_4 = ft_4_left + ft_4_right    
 
    # add the skip feature from prev cell
    if self._prev or self._prev_prev:
      ft_4 = ft_4 + skip_ft

    return ft_1, ft_4


class Cell_dis_Auto(nn.Module):
  """
  Imitate the architecture of Discriminator in Auto-GAN
  Parse the alphas searched by NAS-GAN and fix the architecture of Discriminator 
  """
  def __init__(self, args, C_in, C_out, genotype, cur_stage, downsample=False):
    super(Cell_dis_Auto, self).__init__()
    
    self._downsample = downsample
    self._cur_stage = cur_stage    
 
    if cur_stage == '1':
      self._first_ops = nn.ModuleList()
      op_names_first, indices_first = zip(*(genotype.normal['channels_raise']))
      self._compile(C_in, C_out, self._first_ops, op_names_first, indices_first) 
      self._normal_ops = nn.ModuleList()
      op_names_normal, indices_normal = zip(*(genotype.normal[cur_stage]))
      self._compile(C_out, C_out, self._normal_ops, op_names_normal, indices_normal) 
    else:
      self._normal_ops = nn.ModuleList()
      op_names_normal, indices_normal = zip(*(genotype.normal[cur_stage]))
      self._compile(C_in, C_out, self._normal_ops, op_names_normal, indices_normal)

    if C_in != C_out or downsample:
       self._sc = nn.Conv2d(C_in, C_out, kernel_size=1)

    self._node_2_in = 1
    self._node_3_in = 1
    self._node_4_in = 1
    
  def _compile(self, C_in, C_out, op_list, op_names, indices, down=False, skip=False):
    assert len(op_names) == len(indices)

    for name, index in zip(op_names, indices):
      stride = 2 if down or skip else 1
      if skip:
        stride = 2 ** (len(indices) - index + 1)                        # For skip operation
      op = OPS[name](C_in, C_out, stride, True)
      print('debug@: op is {}'.format(op))
      op_list += [op]
    self._indices = indices

  
  def forward(self, x, prev_ft=None):
    
    # norm operation
    # calculate the output feature of node 2
    ft_2_in = [x]
    ft_2_tmp = []
    offset = 0
    for i in range(self._node_2_in):
      if self._cur_stage == '1':
        ft = self._first_ops[i](ft_2_in[i])
      else:
        ft = self._normal_ops[i + offset](ft_2_in[i])
      ft_2_tmp.append(ft)
    ft_2_out = sum(ft for ft in ft_2_tmp)
    offset += self._node_2_in

    # norm operation
    # calculate the output feature of node 3
    # ft_3_in = [x]
    # ft_3_tmp = []
    # for i in range(self._node_3_in):
    #   ft = self._normal_ops[i + offset](ft_3_in[i])
    #   ft_3_tmp.append(ft)
    # ft_3_out = sum(ft for ft in ft_3_tmp)

    # norm operation
    # the output feature of node 4
    ft_4_in = [ft_2_out]
    ft_down = []
    for i in range(self._node_4_in):
      if self._cur_stage == '1':
        ft = self._normal_ops[i](ft_4_in[i])
      else:
        ft = self._normal_ops[i + offset](ft_4_in[i])
      ft_down.append(ft)
    ft_4_out = sum(ft for ft in ft_down)
    if self._downsample:
      ft_4_out = _downsample(ft_4_out)

    # short cut
    if self._downsample:
      short_cut = self._sc(x)
      short_cut = _downsample(short_cut)
    else:
      short_cut = x

    return ft_4_out + short_cut

def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)


class OptimizedDisBlock(nn.Module):
    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            ksize=3,
            pad=1,
            activation=nn.ReLU()):
        super(OptimizedDisBlock, self).__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.c_sc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class OptimizedDisBlock_bn(nn.Module):
    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            ksize=3,
            pad=1,
            activation=nn.ReLU()):
        super(OptimizedDisBlock_bn, self).__init__()
        self.activation = activation

        self.c1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.c2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.c_sc = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            padding=0)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)
            self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.bn1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = self.bn2(h)
        h = self.activation(h)
        h = _downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(_downsample(x))

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class DisBlock(nn.Module):
    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            hidden_channels=None,
            ksize=3,
            pad=1,
            activation=nn.ReLU(),
            downsample=False):
        super(DisBlock, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.c1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=ksize,
            padding=pad)
        self.c2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0)
            if args.d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
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

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class DisBlock_bn(nn.Module):
    def __init__(
            self,
            args,
            in_channels,
            out_channels,
            hidden_channels=None,
            ksize=3,
            pad=1,
            activation=nn.ReLU(),
            downsample=False):
        super(DisBlock_bn, self).__init__()
        self.activation = activation
        self.downsample = downsample
        self.learnable_sc = (in_channels != out_channels) or downsample
        hidden_channels = in_channels if hidden_channels is None else hidden_channels

        self.c1 = nn.Conv2d(
            in_channels,
            hidden_channels,
            kernel_size=ksize,
            padding=pad)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.c2 = nn.Conv2d(
            hidden_channels,
            out_channels,
            kernel_size=ksize,
            padding=pad)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if args.d_spectral_norm:
            self.c1 = nn.utils.spectral_norm(self.c1)
            self.c2 = nn.utils.spectral_norm(self.c2)

        if self.learnable_sc:
            self.c_sc = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                padding=0)
            if args.d_spectral_norm:
                self.c_sc = nn.utils.spectral_norm(self.c_sc)

    def residual(self, x):
        h = x
        h = self.activation(h)
        h = self.bn1(h)
        h = self.c1(h)
        h = self.activation(h)
        h = self.bn2(h)
        h = self.c2(h)
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

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False)
            )
        # nn.BatchNorm2d(outchannel)

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out
