

from torch import nn
from models.building_blocks import Cell, Cell_gen_Auto, Cell_dis_Auto, DisBlock, DisBlock_bn, OptimizedDisBlock, OptimizedDisBlock_bn, ResidualBlock


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * args.gf_dim)
        self.cell1 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=0, short_cut=False)
        self.cell2 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=1, short_cut=True)
        self.cell3 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=1, short_cut=True)
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(args.gf_dim),
            nn.ReLU(),
            nn.Conv2d(args.gf_dim, args.channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        h = self.l1(z).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h1_skip_out, h1 = self.cell1(h)
        h2_skip_out, h2 = self.cell2(h1, (h1_skip_out, ))
        _, h3 = self.cell3(h2, (h1_skip_out, ))
        output = self.to_rgb(h3)

        return output


class Network_gen_Auto(nn.Module):

  def __init__(self, args, genotype):
    super(Network_gen_Auto, self).__init__()
    self._C = args.gf_dim
    # self._num_classes = args.num_classes
    self._bottom_width = args.bottom_width
    self._steps = 4
    self._multiplier = 4

    self.l1 = nn.Linear(args.latent_dim, (self._bottom_width ** 2) * args.gf_dim)
    self.cell1 = Cell_gen_Auto(args.gf_dim, args.gf_dim, genotype, '1')
    self.cell2 = Cell_gen_Auto(args.gf_dim, args.gf_dim, genotype, '2', prev=True)
    self.cell3 = Cell_gen_Auto(args.gf_dim, args.gf_dim, genotype, '3', prev=False, prev_prev=True)
    self.to_rgb = nn.Sequential(
        nn.BatchNorm2d(args.gf_dim),
        nn.ReLU(),
        nn.Conv2d(args.gf_dim, 3, 3, 1, 1),
        nn.Tanh()
    ) 
  
  def forward(self, input):
    ft_linear = self.l1(input).view(-1, self._C, self._bottom_width, self._bottom_width)
    s1, ft_1 = self.cell1(ft_linear)
    s2, ft_2 = self.cell2(ft_1, [s1])
    _, ft_3 = self.cell3(ft_2, [s1, s2])
    ft_final = self.to_rgb(ft_3)
    
    return ft_final

class Network_dis_Auto(nn.Module):

  def __init__(self, args, genotype):
    super(Network_dis_Auto, self).__init__()
    self._C = args.gf_dim
    self._bottom_width = args.bottom_width
    self._steps = 4
    self._multiplier = 4

    
    self.cell1 = Cell_dis_Auto(args, args.channels, args.df_dim, genotype, '1', downsample=True)
    self.cell2 = Cell_dis_Auto(args, args.df_dim, args.df_dim, genotype, '2', downsample=True)
    self.cell3 = Cell_dis_Auto(args, args.df_dim, args.df_dim, genotype, '3')
    self.cell4 = Cell_dis_Auto(args, args.df_dim, args.df_dim, genotype, '4')
    self.l5 = nn.Linear(args.df_dim, 1, bias=False)
    if args.d_spectral_norm:
      self.l5 = nn.utils.spectral_norm(self.l5)  
 
  def forward(self, input):
    ft_1 = self.cell1(input)
    ft_2 = self.cell2(ft_1)
    ft_3 = self.cell3(ft_2)
    ft_4 = self.cell4(ft_3)
    ft_4 = ft_4.sum(2).sum(2)
    ft_final = self.l5(ft_4)
    
    return ft_final

class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
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
        model = nn.Sequential(*layers)
        h = model(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)
        
        return output

class Discriminator_bn(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator_bn, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock_bn(args, args.channels, self.ch)
        self.block2 = DisBlock_bn(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=True)
        self.block3 = DisBlock_bn(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.block4 = DisBlock_bn(
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
        model = nn.Sequential(*layers)
        h = model(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)
        
        output = output.mean(0)
        return output.view(1)

class Discriminator_ResNet(nn.Module):
    def __init__(self, args, ResidualBlock = ResidualBlock, num_classes=1):
        super(Discriminator_ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
