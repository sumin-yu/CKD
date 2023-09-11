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

    parser.add_argument('--dataset', required=True, default='',
                        choices=['utkface', 'celeba', 'cifar10', 'cifar10_aug', 'cifar10_all',
                                 'celeba_aug','celeba_aug2','celeba_aug3', 'celeba_aug_ukn', 'celeba_aug_ukn_wo_org',
                                 'spucobirds', 'spucobirds_aug', 'celeba_pseudo'])
    parser.add_argument('--skew-ratio', default=0.8, type=float, help='skew ratio for cifar-10s')
    parser.add_argument('--img-size', default=224, type=int, help='img size for preprocessing')

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='mini batch size')
    parser.add_argument('--seed', default=0, type=int, help='seed for randomness')
    parser.add_argument('--date', default='20xxxxxx', type=str, help='experiment date')
    parser.add_argument('--method', default='scratch', type=str, required=True,
                        choices=['scratch', 'scratch_aug',
                                 'kd_mfd', 'kd_mfd_ctf', 'kd_mfd_ctf_ukn', 'kd_mfd_ctf_ukn3', 'kd_mfd_aug', 'kd_indiv', 'kd_indiv_aug', 'kd_indiv_multi',
                                 'kd_mfd_logit_pairing',
                                 'kd_indiv_output',
                                 'kd_indiv_logit_pairing',
                                 'kd_indiv_ukn1', 'kd_indiv_ukn2','kd_indiv_ukn3',
                                 'kd_hinton_aug', 'kd_fitnet_aug',
                                 'scratch_aug','logit_pairing','logit_pairing_ukn', 'logit_pairing_aug', 'group_dro',
                                 'kd_hinton', 'kd_fitnet', 'kd_at',
                                 'scratch_mmd', 'kd_nst', 'adv_debiasing', 'cgdro', 'sensei', 'group_predict'])

    parser.add_argument('--optimizer', default='Adam', type=str, required=False,
                        choices=['SGD', 'Adam','AdamW'],
                        help='(default=%(default)s)')

    parser.add_argument('--alpha-J', default=1, type=float, help='kd strenth hyperparameter for Junyi-Fair-KD')
    parser.add_argument('--lambh', default=0, type=float, help='kd strength hyperparameter')
    parser.add_argument('--lambf', default=1, type=float, help='feature distill strength hyperparameter')
    parser.add_argument('--gamma', default=0, type=float, help='lopgitpairing strength hyperparameter for MFD-indiv')
    parser.add_argument('--rho', default=0.5, type=float, help='the radioi of chi divergence ball')
    parser.add_argument('--kd-temp', default=3, type=float, help='temperature for KD')
    parser.add_argument('--q-step-size', default=0.001, type=float, help='q step size for GDRO epoch')
    parser.add_argument('--num-aug', default=1, type=int, help='the number of augmentation for MFD_indiv')

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
    parser.add_argument('--sampling', default='noBal', type=str, choices=['noBal', 'gBal','cBal', 'gcBal'], help='balanced sampling')
    parser.add_argument('--get-inter', default=False, action='store_true',
                        help='get penultimate features for TSNE visualization')

    parser.add_argument('--sensei-rho', default=5.0, type=float, help='rho for SenSeI')
    parser.add_argument('--sensei-eps', default=0.1, type=float, help='epsilon for SenSeI')
    parser.add_argument('--auditor-nsteps', default=100, type=int, help='auditor nsteps for SenSeI')
    parser.add_argument('--auditor-lr', default=1e-3, type=float, help='auditor lr for SenSeI')
    parser.add_argument('--epochs-dist', default=10, type=int, help='epochs for distance metric learning')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.mode == 'train' and (args.method.startswith('kd')):
        if args.teacher_path is None:
            raise Exception('A teacher model path is not specified.')

    if args.mode == 'eval' and args.modelpath is None:
        raise Exception('Model path to load is not specified!')
    
    return args
