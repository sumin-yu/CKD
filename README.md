# <p align="center">Do Counterfactually Fair Image Classifiers Satisfy Group Fairness?</p>

paper: "Do Counterfactually Fair Image Classifiers Satisfy Group Fairness? – A Theoretical and Empirical Study"

# Real-img-CF datasets for CD evaluation

- CelebA-CF: [link]

- LFW-CF: [link]

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
            - txt files: [link](https://drive.google.com/file/d/1I49GG6Jvl1jUzv8aKLb2SdOvE8oS6XKG/view?usp=share_link)

        - You should generate ctf images for training dataset (e.g., by IP2P) and place them in the folder named "lfw_funneled_Male"

        - move them to **./data/lfw-py/**

        - You must also place "lfw_attributes_binary_cd_eval.txt" to **./data/lfw-py/** to properly evaluate CD.

## Training commands

### CKD

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
    --cp-lambf 100.0 \
    --teacher-path ./trained_models/000000/cifar_10b/scratch/resnet56_seed0_epochs50_bs256_lr0.001_skewed0.8_editbias_alpha0.8.pt

    ```

## Citation
