import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from models.operations_gan import *
from models.genotypes import *
from models.cell import *
import models.genotypes as genotypes


class Network_dis_Auto(nn.Module):

  def __init__(self, args, activation=nn.ReLU()):
    super(Network_dis_Auto, self).__init__()
    self._C = args.df_dim
    self._activation = activation
    self._use_gumbel = False
    self._args = args   
    self.cur_stage = 0    
    self._normal_primitives = eval('genotypes.' + args.dis_normal_opr)
 
    self.cell1 = Cell_dis_Auto_first(args, args.channels, args.df_dim, self._use_gumbel)
    self.cell2 = Cell_dis_Auto(args, args.df_dim, args.df_dim, self._use_gumbel, downsample=True)
    self.cell3 = Cell_dis_Auto(args, args.df_dim, args.df_dim, self._use_gumbel, downsample=False)
    self.cell4 = Cell_dis_Auto(args, args.df_dim, args.df_dim, self._use_gumbel, downsample=False)

    self.l5 = nn.Linear(self._C, 1, bias=False)
    if args.d_spectral_norm:
      self.l5 = nn.utils.spectral_norm(self.l5)
    self.layers = [self.cell2, self.cell3, self.cell4]

    self._initialize_alphas()

  def new(self):
    model_new = Network_dis_Auto(self._args).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def copy_alphas(self, model_new):
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new
 
  def set_gumbel(self, use_gumbel):
    self._use_gumbel = use_gumbel
    self.cell1._use_gumbel = use_gumbel  
    self.cell2._use_gumbel = use_gumbel  
    self.cell3._use_gumbel = use_gumbel  
    self.cell4._use_gumbel = use_gumbel  

  def forward(self, x):
    y = self.cell1(x, self.alphas_normal_1, self.alphas_normal_channels_raise)
    y = self.cell2(y, self.alphas_normal_2)
    if self.cur_stage > 0:
      y = self.cell3(y, self.alphas_normal_3)
    if self.cur_stage > 1:
      y = self.cell4(y, self.alphas_normal_4)
    y = self._activation(y)
    y = y.sum(2).sum(2)
    out = self.l5(y)
    return out

  def _initialize_alphas(self):
    num_ops_normal = len(self._normal_primitives)
    num_ops_normal_channels_raise = 3    # totally 3 operations (conv_1x1, conv_3x3, conv_5x5) when channels increase from 3 to df_dim
    self.k_normal = 2 
    self.alphas_normal_channels_raise = Variable(1e-3*torch.randn(1, num_ops_normal_channels_raise).cuda(), requires_grad=True)
    self.alphas_normal_1 = Variable(1e-3*torch.randn(self.k_normal - 1, num_ops_normal).cuda(), requires_grad=True)
    self.alphas_normal_2 = Variable(1e-3*torch.randn(self.k_normal, num_ops_normal).cuda(), requires_grad=True)
    self.alphas_normal_3 = Variable(1e-3*torch.randn(self.k_normal, num_ops_normal).cuda(), requires_grad=True)
    self.alphas_normal_4 = Variable(1e-3*torch.randn(self.k_normal, num_ops_normal).cuda(), requires_grad=True)
    self._arch_parameters = [
        self.alphas_normal_channels_raise,
        self.alphas_normal_1,
        self.alphas_normal_2,
        self.alphas_normal_3,
        self.alphas_normal_4
    ]

  def _sample_arch(self): 
    num_ops_norm = len(self._normal_primitives)
    num_ops_normal_channels_raise = 3
    arch = {}
    arch['select_normal_channels_raise'] = [np.random.choice(num_ops_normal_channels_raise, 1)]
    arch['select_normal_1'] = [np.random.choice(num_ops_norm, 1) for i in range(self.k_normal - 1)]
    arch['select_normal_2'] = [np.random.choice(num_ops_norm, 1) for i in range(self.k_normal)]
    arch['select_normal_3'] = [np.random.choice(num_ops_norm, 1) for i in range(self.k_normal)] 
    arch['select_normal_4'] = [np.random.choice(num_ops_norm, 1) for i in range(self.k_normal)] 
    return arch
  
  def _set_arch(self, arch):
    self._reset_arch()
    normal = [('select_normal_1', self.alphas_normal_1), ('select_normal_2', self.alphas_normal_2), ('select_normal_3', self.alphas_normal_3), ('select_normal_4', self.alphas_normal_4)]
    for i in normal:
      select = arch[i[0]]
      alphas = i[1]
      for j in range(len(select)):
        alphas[j][select[j]] = 1 
    for j in range(len(arch['select_normal_channels_raise'])):
      self.alphas_normal_channels_raise[j][arch['select_normal_channels_raise'][j]] = 1 
    return

  def _reset_arch(self):
    self.alphas_normal_1 = self.alphas_normal_1 * 0.
    self.alphas_normal_2 = self.alphas_normal_2 * 0.
    self.alphas_normal_3 = self.alphas_normal_3 * 0.
    self.alphas_normal_4 = self.alphas_normal_4 * 0.
    self.alphas_normal_channels_raise = self.alphas_normal_channels_raise * 0.

    return

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights, num_edge, OPS, skip=False):
      gene = []
      for i in range(num_edge):
        W = weights[i]
        k_best = None
        for k in range(len(W)):
          if 'none' not in OPS:
            if skip and k == 0:
              continue
            if k_best is None or W[k] > W[k_best]:
              k_best = k
          else:
            if k_best is None or W[k] > W[k_best]:
              k_best = k
        if skip:
          gene.append((OPS[k_best - 1], i))
        else:
          gene.append((OPS[k_best], i))
      return gene

    gene_normal = {}
    gene_normal['1'] = _parse(F.softmax(self.alphas_normal_1, dim=-1).data.cpu().numpy(), self.k_normal - 1, self._normal_primitives)
    gene_normal['2'] = _parse(F.softmax(self.alphas_normal_2, dim=-1).data.cpu().numpy(), self.k_normal, self._normal_primitives)
    gene_normal['3'] = _parse(F.softmax(self.alphas_normal_3, dim=-1).data.cpu().numpy(), self.k_normal, self._normal_primitives)
    gene_normal['4'] = _parse(F.softmax(self.alphas_normal_4, dim=-1).data.cpu().numpy(), self.k_normal, self._normal_primitives)
    gene_normal['channels_raise'] = _parse(F.softmax(self.alphas_normal_channels_raise, dim=-1).data.cpu().numpy(), self.k_normal - 1, ['conv_1x1', 'conv_3x3', 'conv_5x5'])
    
    genotype = Genotype_dis_Auto(
      normal=gene_normal
    )
    return genotype

class Network_gen_Auto(nn.Module):
  """
  Composed of Cell in AutoGAN
  """
  def __init__(self, args):
    super(Network_gen_Auto, self).__init__()
    self._C = args.gf_dim
    self._num_classes = args.num_classes
    self._bottom_width = args.bottom_width
    self._use_gumbel = False
    self._args = args
    self._normal_primitives = eval('genotypes.' + args.gen_normal_opr)
    self._up_primitives = eval('genotypes.' + args.gen_up_opr)
    self._update_alphas = args.update_alphas

    self.l1 = nn.Linear(args.latent_dim, (self._bottom_width ** 2) * args.gf_dim)
    self.cell1 = Cell_gen_Auto(args, args.gf_dim, args.gf_dim, self._use_gumbel)
    self.cell2 = Cell_gen_Auto(args, args.gf_dim, args.gf_dim, self._use_gumbel, prev=True)
    self.cell3 = Cell_gen_Auto(args, args.gf_dim, args.gf_dim, self._use_gumbel, prev=False, prev_prev=True)
    self.to_rgb = nn.Sequential(
        nn.BatchNorm2d(args.gf_dim),
        nn.ReLU(),
        nn.Conv2d(args.gf_dim, 3, 3, 1, 1),
        nn.Tanh()
    )
    
    self.k_up, self.k_normal, self.skip_2 = self.cell2.calculate_ops()
    _, _, self.skip_3 = self.cell3.calculate_ops()
    
    self._initialize_alphas()

  def new(self):
    model_new = Network_gen_Auto(self._args).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        # print('debug@: x size is {}, y size is {}'.format(x.shape, y.shape))
        x.data.copy_(y.data)
    return model_new

  def copy_alphas(self, model_new):
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def set_gumbel(self, use_gumbel):
    self._use_gumbel = use_gumbel
    self.cell1._use_gumbel = use_gumbel
    self.cell2._use_gumbel = use_gumbel
    self.cell3._use_gumbel = use_gumbel

  def forward(self, input, eval=False):
    if eval:
      gene = self.genotype_discrete()
    ft_linear = self.l1(input).view(-1, self._C, self._bottom_width, self._bottom_width)
    if eval:
      s1, ft_1 = self.cell1(ft_linear, self.alphas_up_1, self.alphas_normal_1, eval=True, gene=gene['1'])
      s2, ft_2 = self.cell2(ft_1, self.alphas_up_2, self.alphas_normal_2, self.alphas_skip_2, [s1], eval=True, gene=gene['2'], gene_skip=gene['skip_2'])
      _, ft_3 = self.cell3(ft_2, self.alphas_up_3, self.alphas_normal_3, self.alphas_skip_3, [s1, s2], eval=True, gene=gene['3'], gene_skip=gene['skip_3'])
    else:
      s1, ft_1 = self.cell1(ft_linear, self.alphas_up_1, self.alphas_normal_1)
      s2, ft_2 = self.cell2(ft_1, self.alphas_up_2, self.alphas_normal_2, self.alphas_skip_2, [s1])
      _, ft_3 = self.cell3(ft_2, self.alphas_up_3, self.alphas_normal_3, self.alphas_skip_3, [s1, s2])
    ft_final = self.to_rgb(ft_3)
    
    return ft_final

  def _loss(self):
    raise NotImplementedError

  def _initialize_alphas(self):
    
    num_ops_norm = len(self._normal_primitives)
    num_ops_up = len(self._up_primitives)
    num_ops_up_skip = len(self._up_primitives)
    self.alphas_normal_1 = Variable(1e-3*torch.randn(self.k_normal, num_ops_norm).cuda(), requires_grad=self._update_alphas)
    self.alphas_normal_2 = Variable(1e-3*torch.randn(self.k_normal, num_ops_norm).cuda(), requires_grad=self._update_alphas)
    self.alphas_normal_3 = Variable(1e-3*torch.randn(self.k_normal, num_ops_norm).cuda(), requires_grad=self._update_alphas)
    self.alphas_up_1 = Variable(1e-3*torch.randn(self.k_up, num_ops_up).cuda(), requires_grad=self._update_alphas)
    self.alphas_up_2 = Variable(1e-3*torch.randn(self.k_up, num_ops_up).cuda(), requires_grad=self._update_alphas)
    self.alphas_up_3 = Variable(1e-3*torch.randn(self.k_up, num_ops_up).cuda(), requires_grad=self._update_alphas)
    self.alphas_skip_2 = Variable(1e-3*torch.randn(self.skip_2, num_ops_up_skip).cuda(), requires_grad=self._update_alphas) 
    self.alphas_skip_3 = Variable(1e-3*torch.randn(self.skip_3, num_ops_up_skip).cuda(), requires_grad=self._update_alphas)
    self._arch_parameters = [
      self.alphas_normal_1,
      self.alphas_normal_2,
      self.alphas_normal_3,
      self.alphas_up_1,
      self.alphas_up_2,
      self.alphas_up_3,
      self.alphas_skip_2,
      self.alphas_skip_3
    ]
    
    

  def _sample_arch(self): 
    num_ops_norm = len(self._normal_primitives)
    num_ops_up = len(PRIMITIVES_UP)
    num_ops_up_skip = len(PRIMITIVES_UP) + 1
    arch = {}
    arch['select_normal_1'] = [np.random.choice(num_ops_norm, 1) for i in range(self.k_normal)]
    arch['select_normal_2'] = [np.random.choice(num_ops_norm, 1) for i in range(self.k_normal)]
    arch['select_normal_3'] = [np.random.choice(num_ops_norm, 1) for i in range(self.k_normal)] 
    arch['select_up_1'] = [np.random.choice(num_ops_up, 1) for i in range(self.k_up)]
    arch['select_up_2'] = [np.random.choice(num_ops_up, 1) for i in range(self.k_up)]
    arch['select_up_3'] = [np.random.choice(num_ops_up, 1) for i in range(self.k_up)]
    arch['select_skip_2'] = [np.random.choice(num_ops_up_skip, 1) for i in range(self.skip_2)]
    arch['select_skip_3'] = [np.random.choice(num_ops_up_skip, 1) for i in range(self.skip_3)]
    return arch
  
  def _set_arch(self, arch):
    self._reset_arch()
    normal = [('select_normal_1', self.alphas_normal_1), ('select_normal_2', self.alphas_normal_2), ('select_normal_3', self.alphas_normal_3)]
    for i in normal:
      select = arch[i[0]]
      alphas = i[1]
      for j in range(len(select)):
        alphas[j][select[j]] = 1
    up = [('select_up_1', self.alphas_up_1), ('select_up_2', self.alphas_up_2), ('select_up_3', self.alphas_up_3)]
    for i in up:
      select = arch[i[0]]
      alphas = i[1]
      for j in range(len(select)):
        alphas[j][select[j]] = 1
    for j in range(len(arch['select_skip_2'])):
      self.alphas_skip_2[j][arch['select_skip_2'][j]] = 1
    for j in range(len(arch['select_skip_3'])):
      self.alphas_skip_3[j][arch['select_skip_3'][j]] = 1
    
    return

  def _reset_arch(self):
    self.alphas_normal_1 = self.alphas_normal_1 * 0.
    self.alphas_normal_2 = self.alphas_normal_2 * 0.
    self.alphas_normal_3 = self.alphas_normal_3 * 0.
    self.alphas_up_1 = self.alphas_up_1 * 0.
    self.alphas_up_2 = self.alphas_up_2 * 0.
    self.alphas_up_3 = self.alphas_up_3 * 0.
    self.alphas_skip_2 = self.alphas_skip_2 * 0.
    self.alphas_skip_3 = self.alphas_skip_3 * 0.

    return

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):
   
    def _parse(weights, num_edge, OPS):
      gene = []
      for i in range(num_edge):
        W = weights[i]
        k_best = None
        for k in range(len(W)):
          if k_best is None or W[k] > W[k_best]:
            k_best = k
        gene.append((OPS[k_best], i))
      return gene

    gene_normal = {}
    gene_normal['1'] = _parse(F.softmax(self.alphas_normal_1, dim=-1).data.cpu().numpy(), self.k_normal, self._normal_primitives)
    gene_normal['2'] = _parse(F.softmax(self.alphas_normal_2, dim=-1).data.cpu().numpy(), self.k_normal, self._normal_primitives)
    gene_normal['3'] = _parse(F.softmax(self.alphas_normal_3, dim=-1).data.cpu().numpy(), self.k_normal, self._normal_primitives)
    gene_up = {}
    gene_up['1'] = _parse(F.softmax(self.alphas_up_1, dim=-1).data.cpu().numpy(), self.k_up, self._up_primitives)
    gene_up['2'] = _parse(F.softmax(self.alphas_up_2, dim=-1).data.cpu().numpy(), self.k_up, self._up_primitives)
    gene_up['3'] = _parse(F.softmax(self.alphas_up_3, dim=-1).data.cpu().numpy(), self.k_up, self._up_primitives)
    gene_skip_2 = _parse(F.softmax(self.alphas_skip_2, dim=-1).data.cpu().numpy(), self.skip_2, self._up_primitives)
    gene_skip_3 = _parse(F.softmax(self.alphas_skip_3, dim=-1).data.cpu().numpy(), self.skip_3, self._up_primitives)

    genotype = Genotype_gen(
      normal=gene_normal,
      up=gene_up,
      skip_2=gene_skip_2,
      skip_3=gene_skip_3
    )
    return genotype

  def genotype_discrete(self):

    def _parse(weights, num_edge, skip=False):
      code = torch.zeros(weights.shape).cuda()
      for i in range(num_edge):
        idx = np.argmax(weights[i])
        code[i][idx] = 1. 
      return code

    gene = {'1': {},
            '2': {},
            '3': {},
            'skip_2': {},
            'skip_3': {}}
    gene['1']['normal'] = _parse(F.softmax(self.alphas_normal_1, dim=-1).data.cpu().numpy(), self.k_normal)
    gene['2']['normal'] = _parse(F.softmax(self.alphas_normal_2, dim=-1).data.cpu().numpy(), self.k_normal)
    gene['3']['normal'] = _parse(F.softmax(self.alphas_normal_3, dim=-1).data.cpu().numpy(), self.k_normal)
    gene['1']['up'] = _parse(F.softmax(self.alphas_up_1, dim=-1).data.cpu().numpy(), self.k_up)
    gene['2']['up'] = _parse(F.softmax(self.alphas_up_2, dim=-1).data.cpu().numpy(), self.k_up)
    gene['3']['up'] = _parse(F.softmax(self.alphas_up_3, dim=-1).data.cpu().numpy(), self.k_up)
    gene['skip_2'] = _parse(F.softmax(self.alphas_skip_2, dim=-1).data.cpu().numpy(), self.skip_2, skip=True)
    gene['skip_3'] = _parse(F.softmax(self.alphas_skip_3, dim=-1).data.cpu().numpy(), self.skip_3, skip=True)

    return gene

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
        h = nn.AvgPool2d(kernel_size=2)(h)
        return h

    def shortcut(self, x):
        return self.c_sc(nn.AvgPool2d(kernel_size=2)(x))

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
            h = nn.AvgPool2d(kernel_size=2)(h)
        return h

    def shortcut(self, x):
        if self.learnable_sc:
            x = self.c_sc(x)
            if self.downsample:
                return nn.AvgPool2d(kernel_size=2)(x)
            else:
                return x
        else:
            return x

    def forward(self, x):
        return self.residual(x) + self.shortcut(x)

class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.cur_stage = 0
        self.block1 = OptimizedDisBlock(args, args.channels, self.ch)
        self.block2 = DisBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=True)
        self.block3 = DisBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.block4 = DisBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x):
        h = x
        layers = [self.block1, self.block2, self.block3]
        model = nn.Sequential(*layers[:(self.cur_stage + 1)])
        h = model(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output

    def set_gumbel(self, use_gumbel):
        return



def _downsample(x):
    # Downsample (Mean Avg Pooling with 2x2 kernel)
    return nn.AvgPool2d(kernel_size=2)(x)

"""
def calculate_gumbel_softmax(weights, t):
  q = F.softmax(weights, dim = -1)
  new_q = q.clone()
  noise = [np.random.normal(0, 1, (1,))[0] for i in range(q.shape[0])]
  noise = torch.tensor(noise, requires_grad=False)
  noise = noise.type(torch.cuda.FloatTensor)
  for i in range(q.shape[0]):
    new_q[i] = (q[i] + noise[i]) / t
  out = F.softmax(new_q, dim = -1)
  return out
"""


