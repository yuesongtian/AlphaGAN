import torch
import torch.nn as nn
import torch.nn.functional as F

OPS = {
  'none' : lambda C_in, C_out, stride, affine: Zero(C_in, C_out, stride),
  'none_up' : lambda C_in, C_out, stride, affine: Zero_up(stride),
  'avg_pool_3x3' : lambda C, stride, affine: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
  'max_pool_3x3' : lambda C, stride, affine: nn.MaxPool2d(3, stride=stride, padding=1),
  'skip_connect' : lambda C_in, C_out, stride, affine: Identity() if stride == 1 else FactorizedReduce(C_in, C_out, affine=affine),
  'sep_conv_3x3' : lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 3, stride, 1, affine=affine),
  'sep_conv_5x5' : lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 5, stride, 2, affine=affine),
  'sep_conv_7x7' : lambda C_in, C_out, stride, affine: SepConv(C_in, C_out, 7, stride, 3, affine=affine),
  'sep_conv_3x3_single' : lambda C_in, C_out, stride, affine: SepConv_single(C_in, C_out, 3, stride, 1, affine=affine),
  'sep_conv_5x5_single' : lambda C_in, C_out, stride, affine: SepConv_single(C_in, C_out, 5, stride, 2, affine=affine),
  'sep_conv_7x7_single' : lambda C_in, C_out, stride, affine: SepConv_single(C_in, C_out, 7, stride, 3, affine=affine),
  'dil_conv_3x3' : lambda C, stride, affine: DilConv(C, C, 3, stride, 2, 2, affine=affine),
  'dil_conv_5x5' : lambda C, stride, affine: DilConv(C, C, 5, stride, 4, 2, affine=affine),
  'conv_7x1_1x7' : lambda C, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C, C, (1,7), stride=(1, stride), padding=(0, 3), bias=False),
    nn.Conv2d(C, C, (7,1), stride=(stride, 1), padding=(3, 0), bias=False),
    nn.BatchNorm2d(C, affine=affine)
    ),
  'conv_1x1' : lambda C_in, C_out, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C_in, C_out, 1, stride=stride),
    ),
  'conv_3x3' : lambda C_in, C_out, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C_in, C_out, 3, stride=stride, padding=3 // 2),
    ),
  'conv_5x5' : lambda C_in, C_out, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.Conv2d(C_in, C_out, 5, stride=stride, padding= 5 // 2),
    ),
  'conv_1x1_bn' : lambda C_in, C_out, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.BatchNorm2d(C_in),
    nn.utils.spectral_norm(nn.Conv2d(C_in, C_out, 1, stride=stride)),
    ),
  'conv_3x3_bn' : lambda C_in, C_out, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.BatchNorm2d(C_in),
    nn.utils.spectral_norm(nn.Conv2d(C_in, C_out, 3, stride=stride, padding=3 // 2)),
    ),
  'conv_5x5_bn' : lambda C_in, C_out, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.BatchNorm2d(C_in),
    nn.utils.spectral_norm(nn.Conv2d(C_in, C_out, 5, stride=stride, padding= 5 // 2)),
    ),
  'conv_1x1_SN' : lambda C_in, C_out, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.utils.spectral_norm(nn.Conv2d(C_in, C_out, 1, stride=stride)),
    ),
  'conv_3x3_SN' : lambda C_in, C_out, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.utils.spectral_norm(nn.Conv2d(C_in, C_out, 3, stride=stride, padding=3 // 2)),
    ),
  'conv_5x5_SN' : lambda C_in, C_out, stride, affine: nn.Sequential(
    nn.ReLU(inplace=False),
    nn.utils.spectral_norm(nn.Conv2d(C_in, C_out, 5, stride=stride, padding= 5 // 2)),
    ),
  'deconv' : lambda C_in, C_out, stride, affine: deconv_wrapper(C_in, C_out, stride=stride),
  'deconv_norm' : lambda C_in, C_out, stride, affine: nn.Sequential(
                                                        deconv_wrapper(C_in, C_out, stride=stride),  
                                                        nn.InstanceNorm2d(C_out, affine=True, track_running_stats=True),
                                                        nn.ReLU(inplace=True),
                                                        ),
  'nearest' : lambda C_in, C_out, stride, affine: Interpolate_wrapper('nearest', stride=stride),
  'bilinear' : lambda C_in, C_out, stride, affine: Interpolate_wrapper('bilinear', stride=stride),
  'nearest_conv' : lambda C_in, C_out, stride, affine: Interpolate_wrapper_conv('nearest', C_in, C_out, stride=stride),
  'bilinear_conv' : lambda C_in, C_out, stride, affine: Interpolate_wrapper_conv('bilinear', C_in, C_out, stride=stride),
}

# nn.BatchNorm2d(C, affine=affine)

class ReLUConvBN(nn.Module):

  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(ReLUConvBN, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
      nn.BatchNorm2d(C_out, affine=affine)
    )

  def forward(self, x):
    return self.op(x)

class DilConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, dilation, affine=True):
    super(DilConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)


class SepConv(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=1, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_out, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_out, affine=affine),
      )

  def forward(self, x):
    return self.op(x)

class SepConv_single(nn.Module):
    
  def __init__(self, C_in, C_out, kernel_size, stride, padding, affine=True):
    super(SepConv_single, self).__init__()
    self.op = nn.Sequential(
      nn.ReLU(inplace=False),
      nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=C_in, bias=False),
      nn.Conv2d(C_in, C_in, kernel_size=1, padding=0, bias=False),
      nn.BatchNorm2d(C_in, affine=affine),
      )

  def forward(self, x):
    return self.op(x)

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class Interpolate_wrapper(nn.Module):

  def __init__(self, mode, stride=2):
    super(Interpolate_wrapper, self).__init__()
    self._mode = mode
    self._opr_num = stride // 2

  def forward(self, x):
    ft = x
    for i in range(self._opr_num):
      ft = F.interpolate(ft, scale_factor = 2, mode = self._mode)
    return ft

class Interpolate_wrapper_conv(nn.Module):

  def __init__(self, mode, C_in, C_out, stride=2):
    super(Interpolate_wrapper_conv, self).__init__()
    self._mode = mode
    self._opr_num = stride // 2
    self._conv = nn.Conv2d(C_in, C_out, kernel_size=1, stride=1, bias=False)
  
  def forward(self, x):
    ft = self._conv(x)
    for i in range(self._opr_num):
      ft = F.interpolate(ft, scale_factor = 2, mode = self._mode)
    return ft

class deconv_wrapper(nn.Module):

  def __init__(self, C_in, C_out, stride=2):
    super(deconv_wrapper, self).__init__()
    self._opr_num = stride // 2
    self._func = nn.ConvTranspose2d(C_in, C_out, kernel_size=2, stride=2)

  def forward(self, x):
    ft = x
    for i in range(self._opr_num):
      ft = self._func(ft)
    return ft

class Zero(nn.Module):

  def __init__(self, C_in, C_out, stride):
    super(Zero, self).__init__()
    self.stride = stride
    self.C_in = C_in
    self.C_out = C_out

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    if self.C_in != self.C_out:
      out = nn.Conv2d(C_in, C_out, kernel_size=1, stride=self.stride)(x)
      return out.mul(0.)
    return x[:,:,::self.stride,::self.stride].mul(0.)

class Zero_up(nn.Module):

  def __init__(self, stride):
    super(Zero_up, self).__init__()
    self.stride = stride
    self._opr_num = stride // 2

  def forward(self, x):
    if self.stride == 1:
      return x.mul(0.)
    ft = x
    for i in range(self._opr_num): 
      ft = F.interpolate(ft, scale_factor = 2)
    return ft.mul(0.)

class FactorizedReduce(nn.Module):

  def __init__(self, C_in, C_out, affine=True):
    super(FactorizedReduce, self).__init__()
    assert C_out % 2 == 0
    self.relu = nn.ReLU(inplace=False)
    self.conv_1 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
    self.conv_2 = nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False) 
    self.bn = nn.BatchNorm2d(C_out, affine=affine)

  def forward(self, x):
    x = self.relu(x)
    out = torch.cat([self.conv_1(x), self.conv_2(x[:,:,1:,1:])], dim=1)
    out = self.bn(out)
    return out

