import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='Fairness')
    parser.add_argument('--log-dir', default='./results/',
                        help='directory to save logs (default: ./results/)')
    parser.add_argument('--data-dir', default='./data/',
                        help='data directory (default: ./data/)')
    parser.add_argument('--save-dir', default='./trained_models/',
                        help='directory to save trained models (default: ./trained_models/)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--device', default=0, type=int, help='cuda device number')
    parser.add_argument('--t-device', default=0, type=int, help='teacher cuda device number')


    parser.add_argument('--mode', default='train', choices=['train', 'eval'])
    parser.add_argument('--modelpath', default=None)
    parser.add_argument('--evalset', default='all', choices=['all', 'train', 'test', 'val'])

    parser.add_argument('--dataset', required=True, default='', choices=['utkface', 'celeba', 'cifar10', 'cifar10_aug', 'cifar10_all', 'celeba_aug','celeba_aug2','celeba_aug3'])
    parser.add_argument('--skew-ratio', default=0.8, type=float, help='skew ratio for cifar-10s')
    parser.add_argument('--img-size', default=224, type=int, help='img size for preprocessing')

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='mini batch size')
    parser.add_argument('--seed', default=0, type=int, help='seed for randomness')
    parser.add_argument('--date', default='20xxxxxx', type=str, help='experiment date')
    parser.add_argument('--method', default='scratch', type=str, required=True,
                        choices=['scratch', 'scratch_aug',
                                 'kd_mfd', 'kd_indiv', 'kd_indiv_aug', 'kd_indiv_multi',
                                 'kd_hinton_aug', 'kd_fitnet_aug',
                                 'scratch_aug','logit_pairing','logit_pairing_aug', 'group_dro',
                                 'kd_hinton', 'kd_fitnet', 'kd_at',
                                 'scratch_mmd', 'kd_nst', 'adv_debiasing', 'kd_mfd_ctf','kd_mfd_aug'])

    parser.add_argument('--optimizer', default='Adam', type=str, required=False,
                        choices=['SGD', 'SGD_momentum_decay', 'Adam','AdamW'],
                        help='(default=%(default)s)')

    parser.add_argument('--alpha-J', default=1, type=float, help='kd strenth hyperparameter for Junyi-Fair-KD')
    parser.add_argument('--lambh', default=0, type=float, help='kd strength hyperparameter')
    parser.add_argument('--lambf', default=1, type=float, help='feature distill strength hyperparameter')
    parser.add_argument('--kd-temp', default=3, type=float, help='temperature for KD')
    parser.add_argument('--q-step-size', default=0.001, type=float, help='q step size for GDRO epoch')

    parser.add_argument('--model', default='', required=True, choices=['resnet', 'shufflenet', 'mlp', 'cifar_net', 'resnet152'])
    parser.add_argument('--parallel', default=False, action='store_true', help='data parallel')
    parser.add_argument('--teacher-type', default=None, choices=['resnet', 'shufflenet', 'cifar_net'])
    parser.add_argument('--teacher-path', default=None, help='teacher model path')

    parser.add_argument('--pretrained', default=False, action='store_true', help='load imagenet pretrained model')
    parser.add_argument('--num-workers', default=2, type=int, help='the number of thread used in dataloader')
    parser.add_argument('--term', default=20, type=int, help='the period for recording train acc')
    parser.add_argument('--target', default='Blond_Hair', type=str, help='target attribute for celeba')
    parser.add_argument('--sensitive', default='Male', type=str, help='sensitive attribute for celeba')

    parser.add_argument('--fitnet-simul', default=False, action='store_true', help='no hint-training')

    parser.add_argument('--eta', default=0.0003, type=float, help='adversary training learning rate')
    parser.add_argument('--adv-lambda', default=2.0, type=float, help='adversary loss strength')

    parser.add_argument('--sigma', default=1.0, type=float, help='sigma for rbf kernel')
    parser.add_argument('--kernel', default='rbf', type=str, choices=['rbf', 'poly', 'linear'], help='kernel for mmd')
    parser.add_argument('--labelwise', default=False, action='store_true', help='labelwise loader')
    parser.add_argument('--jointfeature', default=False, action='store_true', help='mmd with both joint')
    parser.add_argument('--get-inter', default=False, action='store_true',
                        help='get penultimate features for TSNE visualization')
    parser.add_argument('--with-perturbed', default=False, action='store_true', help='CE loss with perturbed images and no mmd loss')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.mode == 'train' and (args.method.startswith('kd')):
        if args.teacher_path is None:
            raise Exception('A teacher model path is not specified.')

    if args.mode == 'eval' and args.modelpath is None:
        raise Exception('Model path to load is not specified!')
    
    return args
