from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
Genotype_gen = namedtuple('Genotype_gen', 'normal up skip_2 skip_3')
Genotype_dis = namedtuple('Genotype_dis', 'normal normal_concat down down_concat skip_2 skip_2_concat skip_3 skip_3_concat skip_4 skip_4_concat')
Genotype_dis_Auto = namedtuple('Genotype_dis_Auto', 'normal')
Genotype_dis_simplify = namedtuple('Genotype_dis_simplify', 'normal normal_concat down down_concat sc sc_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]

# I remove skip_connect from PRIMITIVES_NORMAL_GEN
PRIMITIVES_NORMAL_GEN = [
    'none',
    'conv_1x1',
    'conv_3x3',
    'conv_5x5',
    'skip_connect'
]

PRIMITIVES_NORMAL_GEN_wo_skip_none = [
    'conv_1x1',
    'conv_3x3',
    'conv_5x5',
]

PRIMITIVES_NORMAL_GEN_wo_skip_none_sep = [
    'conv_1x1',
    'conv_3x3',
    'conv_5x5',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7'
]

PRIMITIVES_NORMAL_GEN_wo_skip_none_sepSingle = [
    'conv_1x1',
    'conv_3x3',
    'conv_5x5',
    'sep_conv_3x3_single',
    'sep_conv_5x5_single',
    'sep_conv_7x7_single'
]

PRIMITIVES_NORMAL_GEN_sep = [
    'none',
    'conv_1x1',
    'conv_3x3',
    'conv_5x5',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'sep_conv_7x7',
    'skip_connect'
]

PRIMITIVES_NORMAL_GEN_BN = [
    'none',
    'conv_1x1_bn',
    'conv_3x3_bn',
    'conv_5x5_bn',
    'skip_connect'
]

PRIMITIVES_NORMAL_DIS = [
    'none',
    'conv_1x1_SN',
    'conv_3x3_SN',
    'conv_5x5_SN',
    'skip_connect'
]

PRIMITIVES_NORMAL_DIS_wo_skip_none = [
    'conv_1x1_SN',
    'conv_3x3_SN',
    'conv_5x5_SN',
]

PRIMITIVES_NORMAL_DIS_BN = [
    'none',
    'conv_1x1_bn',
    'conv_3x3_bn',
    'conv_5x5_bn',
    'skip_connect'
]

PRIMITIVES_UP = [
    'deconv',
    'nearest',
    'bilinear'  
]

PRIMITIVES_UP_conv = [
    'deconv',
    'nearest_conv',
    'bilinear_conv'  
]

PRIMITIVES_UP_norm = [
    'deconv_norm',
    'nearest',
    'bilinear'  
]

PRIMITIVES_DOWN = [
    'conv_1x1_SN',
    'conv_3x3_SN',
    'conv_5x5_SN',
    'max_pool_3x3',
    'avg_pool_3x3',
]

 
alphaGAN_l = Genotype_gen(normal={'1': [('conv_5x5', 0), ('conv_5x5', 1), ('sep_conv_5x5', 2)], '2': [('conv_5x5', 0), ('conv_1x1', 1), ('conv_3x3', 2)], '3': [('conv_1x1', 0), ('conv_5x5', 1), ('sep_conv_3x3', 2)]}, up={'1': [('deconv', 0), ('bilinear', 1)], '2': [('deconv', 0), ('bilinear', 1)], '3': [('nearest', 0), ('bilinear', 1)]}, skip_2=[('nearest', 0)], skip_3=[('bilinear', 0), ('nearest', 1)])

alphaGAN_s = Genotype_gen(normal={'1': [('conv_1x1', 0), ('sep_conv_3x3', 1), ('conv_1x1', 2)], '2': [('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)], '3': [('sep_conv_5x5', 0), ('sep_conv_3x3', 1), ('sep_conv_3x3', 2)]}, up={'1': [('bilinear', 0), ('deconv', 1)], '2': [('deconv', 0), ('deconv', 1)], '3': [('deconv', 0), ('nearest', 1)]}, skip_2=[('nearest', 0)], skip_3=[('deconv', 0), ('bilinear', 1)])
