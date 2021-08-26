# alphaGAN ([TPAMI-2021](https://ieeexplore.ieee.org/abstract/document/9496081))

AlphaGAN is a fully differentable architecture search framework for GAN. We implement alphaGAN referring to the existing public repositories of [DARTS](https://github.com/quark0/darts) and [AutoGAN](https://github.com/TAMU-VITA/AutoGAN). 

## Install

To run this code, you need:
- PyTorch 1.3.0
- TensorFlow 1.15.0
- cuda 10.0

Other requirements are in requirements.txt

## Prepare
First, you need to create logging directory.
```bash
cd /root_dir/alphaGAN && mkdir logs
```

Second, you need to create "fid_stat" directory and download the statistical files of real images.
```bash
mkdir fid_stat
```

## Searching

To search with alphaGAN$_{(s)}$,
```bash
CUDA_VISIBLE_DEVICES=0 python search.py --gen Network_gen_Auto --dis Discriminator --gf_dim 256 --df_dim 128 --fix_alphas_epochs -1 --only_update_w_g --gen_normal_opr PRIMITIVES_NORMAL_GEN_wo_skip_none_sep --inner_steps 20 --worst_steps 20 --outter_steps 20 --exp_name search_test  --eval_every 4 --dataset cifar10
```
To search with alphaGAN$_{(l)}$,
```bash
CUDA_VISIBLE_DEVICES=0 python search.py --gen Network_gen_Auto --dis Discriminator --gf_dim 256 --df_dim 128 --fix_alphas_epochs -1 --only_update_w_g --gen_normal_opr PRIMITIVES_NORMAL_GEN_wo_skip_none_sep --inner_steps 390 --worst_steps 390 --outter_steps 20 --exp_name search_test  --eval_every 4 --dataset cifar10
```
Currently, alphaGAN supports searching on CIFAR-10 or STL-10. You just need to replace cifar10 of '--dataset' with stl10. According to our experiments, searching on STL-10 can stabalize the searching process.

## Re-training

After searching, a genotype representing the architecture of the generator is obtained. You need to copy the genotype to models/genotypes.py and name the genotype. Supposed that the genotype is named as ‘arch’. To re-train the searched generator on CIFAR-10,
```bash
CUDA_VISIBLE_DEVICES=0 python train_geno_gan.py -gen_bs 128 -dis_bs 64 --dataset cifar10 --bottom_width 4 --img_size 32 --max_iter 50000 --gen_model alphaGAN_network --dis_model alphaGAN_network --latent_dim 128 --gf_dim 256 --df_dim 128 --g_spectral_norm False --d_spectral_norm True --g_lr 2e-4 --d_lr 2e-4 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 20 --arch_gen arch --gen Network_gen_Auto --dis Discriminator --exp_name re-train_searched_G --eval_dg
```
To re-train the searched generator on STL-10,
```bash
CUDA_VISIBLE_DEVICES=0 python train_geno_gan.py -gen_bs 128 -dis_bs 64 --dataset stl10 --bottom_width 6 --img_size 48 --max_iter 80000 --gen_model alphaGAN_network --dis_model alphaGAN_network --latent_dim 128 --gf_dim 256 --df_dim 128 --g_spectral_norm False --d_spectral_norm True --g_lr 2e-4 --d_lr 2e-4 --beta1 0.5 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 20 --arch_gen arch --gen Network_gen_Auto --dis Discriminator --exp_name re-train_searched_G_onSTL10 --eval_dg
```
If you want to re-train the network based on the alphaGAN$_{(l)}$ architecture reported in the paper,
```bash
CUDA_VISIBLE_DEVICES=0 python train_geno_gan.py -gen_bs 128 -dis_bs 64 --dataset stl10 --bottom_width 6 --img_size 48 --max_iter 80000 --gen_model alphaGAN_network --dis_model alphaGAN_network --latent_dim 128 --gf_dim 256 --df_dim 128 --g_spectral_norm False --d_spectral_norm True --g_lr 2e-4 --d_lr 2e-4 --beta1 0.5 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 20 --arch_gen alphaGAN_l --gen Network_gen_Auto --dis Discriminator --exp_name re-train_searched_G_onSTL10 --eval_dg
```

## Big-Data

I am still tuning alphaGAN on celebA or LSUN. The code is coming soon.
