# <p align="center">Do Counterfactually Fair Image Classifiers Satisfy Group Fairness? – A Theoretical and Empirical Study (NeurIPS 2024)</p>

Official Pytorch Implementation of the paper: "Do Counterfactually Fair Image Classifiers Satisfy Group Fairness? – A Theoretical and Empirical Study" (NeurIPS 2024) | [Paper]

[Sangwon Jung](https://scholar.google.com/citations?user=WdC_a5IAAAAJ&hl=ko)<sup>1</sup> [Sumin Yu](https://sites.google.com/view/sumin-yu)<sup>1</sup> [Sanghyuk Chun](https://sanghyukchun.github.io/home/)<sup>2</sup> [Taesup Moon](https://scholar.google.com/citations?user=lQlioBoAAAAJ&hl=ko)<sup>1, 3</sup>

<sup>1</sup><sub>Department of Electrical and Computer Engineering, Seoul National University<br>
<sup>2</sup><sub>[NAVER AI LAB](https://naver-career.gitbook.io/en/teams/clova-cic)</sub><br>
<sup>3</sup><sub>ASRI/INMC/IPAI/AIIS, Seoul National University

The notion of algorithmic fairness has been actively explored from various aspects of fairness, such as counterfactual fairness (CF) and group fairness (GF). However, the exact relationship between CF and GF remains to be not clearly understood, especially in image classification tasks; we often cannot collect counterfactual samples regarding a sensitive attribute, essential for evaluating CF, from the existing images (\eg, a photo of the same person but with different secondary sex characteristics). In this paper, we construct new image datasets for evaluating CF by using a high-quality image editing method and carefully labeling with human annotators. Our datasets, \oursceleb and \ourslfw, build upon the popular image GF benchmarks; hence, we can evaluate CF and GF simultaneously. We empirically observe that CF does not imply GF in image classification, whereas previous studies on tabular datasets observed the opposite. We theoretically show that it could be due to a latent attribute $G$ correlated with, but not caused by, the sensitive attribute (\eg, secondary sex characteristics are highly correlated with hair length). From this observation, we propose a simple baseline Counterfactual Knowledge Distillation (CKD) to mitigate the problem. Extensive experimental results on \oursceleb and \ourslfw demonstrate that CF-achieving models satisfy GF if we successfully reduce the reliance to $G$ (\eg, using CKD). 


# Real-img-CF datasets for CD evaluation

- CelebA-CF: [link](https://figshare.com/s/62b6f7f69d0eab9c3c71)

- LFW-CF: [link](https://figshare.com/s/39f2daac58148e10e5fe)

# CKD implementation

## Prerequisites

- Environment

    - Python 3.9.7

    - Torch 1.12.1


- Data Preparation

    - [CelebA]

        - download
            - the dataset (images, txt files): [link](https://drive.google.com/drive/folders/0B7EVK8r0v71pWEZsZE9oNnFzTm8?resourcekey=0-5BR16BdXnb8hVj6CNHKzLg&usp=sharing)
                (Anno/list_attr_celeba.txt, Eval/list_eval_partition.txt, Img/img_align_celeba.zip)
            
        - You should generate ctf images for training dataset (e.g. by IP2P) and place them in the folder named "img_align_celeba_edited_Male"

        - move them to **./data/celeba/**

        - You must also place "list_attr_celeba_cd_eval.txt" to **./data/celeba/** to properly evaluate CD.

    - [LFW]

        - download
            - images: [link](http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz)
            - txt files: [link](https://figshare.com/s/8ee3107511f07af6aa5c)

        - You should generate ctf images for training dataset (e.g., by IP2P) and place them in the folder named "lfw_funneled_Male"

        - move them to **./data/lfw-py/**

        - You must also place "lfw_attributes_binary_cd_eval.txt" to **./data/lfw-py/** to properly evaluate CD.

## CKD Training commands

- First, train a vanilla-trained teacher model.

    ```
    # for CelebA
    CUDA_VISIBLE_DEVICES=0 \
    python main.py \
    --model resnet \
    --method scratch \
    --dataset celeba \
    --lr 0.001 \
    --epochs 70 \
    --batch-size 128 \
    --optimizer AdamW \
    --date '000000'
    --seed 0 \
    --sensitive Male \
    --target Blond_Hair

    # for LFW
    CUDA_VISIBLE_DEVICES=0 \
    python main.py \
    --model resnet \
    --method scratch \
    --dataset lfw \
    --lr 0.001 \
    --epochs 50 \
    --batch-size 128 \
    --optimizer AdamW \
    --date '000000'
    --seed 0 \
    --sensitive Male \
    --target Smiling

    # for CIFAR-10B
    CUDA_VISIBLE_DEVICES=0 \
    python main.py \
    --model resnet56 \
    --method scratch \
    --dataset cifar_10b \
    --lr 0.001 \
    --epochs 50 \
    --batch-size 256 \
    --optimizer Adam \
    --date '000000'
    --seed 0 \
    --editing-bias-alpha 0.8

    ```
- Then, train a CKD

    ```
    # for CelebA
    CUDA_VISIBLE_DEVICES=0 \
    python main.py \
    --model resnet \
    --method ckd \
    --dataset celeba_aug \
    --lr 0.001 \
    --epochs 70 \
    --batch-size 128 \
    --optimizer AdamW \
    --date '000000'
    --seed 0 \
    --sensitive Male \
    --target Blond_Hair \
    --sampling gcBal \
    --kd-lambf 1000.0 \
    --rep feature
    --cp-lambf 7.0 \
    --teacher-path ./trained_models/000000/celeba/scratch/resnet_seed0_epochs70_bs128_lr0.001_Blond_Hair_Male.pt

    # for LFW
    CUDA_VISIBLE_DEVICES=0 \
    python main.py \
    --model resnet \
    --method ckd \
    --dataset lfw_aug \
    --lr 0.001 \
    --epochs 50 \
    --batch-size 128 \
    --optimizer AdamW \
    --date '000000'
    --seed 0 \
    --sensitive Male \
    --target Smiling \
    --kd-lambf 10.0 \
    --rep logit
    --cp-lambf 30.0 \
    --teacher-path ./trained_models/000000/lfw/scratch/resnet_seed0_epochs50_bs128_lr0.001_Smiling_Male.pt

    # for CIFAR-10B
    CUDA_VISIBLE_DEVICES=0 \
    python main.py \
    --model resnet56 \
    --method ckd \
    --dataset cifar_10b_aug \
    --lr 0.001 \
    --epochs 50 \
    --batch-size 256 \
    --optimizer Adam \
    --date '000000'
    --seed 0 \
    --editing-bias-alpha 0.8 \
    --kd-lambf 100.0 \
    --rep logit
    --cp-lambf 100.0 \
    --teacher-path ./trained_models/000000/cifar_10b/scratch/resnet56_seed0_epochs50_bs256_lr0.001_skewed0.8_editbias_alpha0.8.pt

    ```

## Evaluation commands

    ```
    # for CelebA
    CUDA_VISIBLE_DEVICES=0 \
    python main.py \
    --model resnet \
    --method ckd \
    --dataset celeba_aug \
    --lr 0.001 \
    --epochs 70 \
    --batch-size 128 \
    --optimizer AdamW \
    --date '000000'
    --seed 0 \
    --sensitive Male \
    --target Blond_Hair \
    --rep feature
    --mode eval \
    --modelpath ./trained_models/000000/celeba_aug/ckd/resnet_seed0_epochs70_bs128_lr0.001_cp7.0_ckd_f1000.0_gcBal_Blond_Hair_Male.pt

    # for LFW
    CUDA_VISIBLE_DEVICES=0 \
    python main.py \
    --model resnet \
    --method ckd \
    --dataset lfw_aug \
    --lr 0.001 \
    --epochs 50 \
    --batch-size 128 \
    --optimizer AdamW \
    --date '000000'
    --seed 0 \
    --sensitive Male \
    --target Smiling \
    --rep logit
    --mode eval \
    --modelpath ./trained_models/000000/lfw_aug/ckd/resnet_seed0_epochs50_bs128_lr0.001_cp30.0_ckd_l10.0_gcBal_Smiling_Male.pt

    # for CIFAR-10B
    CUDA_VISIBLE_DEVICES=0 \
    python main.py \
    --model resnet56 \
    --method ckd \
    --dataset cifar_10b_aug \
    --lr 0.001 \
    --epochs 50 \
    --batch-size 256 \
    --optimizer Adam \
    --date '000000'
    --seed 0 \
    --editing-bias-alpha 0.8 \
    --rep logit
    --mode eval \
    --modelpath ./trained_models/000000/cifar_10b_aug/ckd/resnet56_seed0_epochs50_bs128_lr0.001_cp100.0_ckd_l100.0_gcBal_skewed0.8_editbias_alpha0.8.pt


    ```
## How to cite

```
@inproceedings{jung2024ckd,
    title={Do Counterfactually Fair Image Classifiers Satisfy Group Fairness? – A Theoretical and Empirical Study}, 
    author={Sangwon Jung and Sumin Yu and Sanghyuk Chun and Taesup Moon},
    year={2024},
    booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
}
```
