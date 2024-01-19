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
                        choices=['utkface', 'celeba', 'cifar10', 'cifar10_aug',
                                 'cifar10_b', 'cifar10_b_aug', 'cifar10_b_same', 'cifar10_b_same_aug',
                                 'Ccifar10_b', 'Ccifar10_b_aug', 'Ccifar10_b_same', 'Ccifar10_b_same_aug',
                                 'celeba_aug','celeba_aug2','celeba_aug3', 'celeba_aug_ukn', 'celeba_aug_ukn_wo_org', 'celeba_aug_filtered',
                                 'spucobirds', 'spucobirds_aug', 'spucobirds_aug_filtered',
                                 'raf', 'raf_aug', 'celeba_pseudo', 'celeba_hq', 'celeba_hq_aug',
                                 'lfw', 'lfw_aug'])
    parser.add_argument('--skew-ratio', default=0.9, type=float, help='skew ratio for cifar-10s')
    parser.add_argument('--group-bias-type', default='color', type=str, choices=['color', 'Contrast'], help='group bias type for cifar-10s')
    parser.add_argument('--group-bias-degree', default=1, type=int, choices=[1,2,3,4,5], help='group bias degree for cifar-10s')
    parser.add_argument('--domain-gap-degree', default=0, type=int, choices=[0, 1, 3, 5], help='domain gap degree for cifar-10s')
    parser.add_argument('--editing-bias-beta', default=0, type=int, choices=[0,1,2,3, 4], help='editing bias degree for cifar-10s. degree0 means that ctf image is perfect <-> degree3 means that ctf image is made by overly biased editing model')
    parser.add_argument('--editing-bias-alpha', default=0.0, type=float, help='editing bias alpha for cifar10-s')
    parser.add_argument('--noise-degree', default=1, type=int, choices=[1,3,5], help='noise degree for cifar-10s')
    parser.add_argument('--noise-type', default='Spatter', type=str, choices=['Gaussian_Noise', 'Zoom_Blur', 'Motion_Blur', 'Snow', 'Spatter', 'Elastic', 'Contrast'], help='noise type for cifar-10s')
    parser.add_argument('--noise-corr', default='neg', type=str, choices=['pos', 'neg'], help='noise correlation for cifar-10s')
    parser.add_argument('--test-alpha-pc', default=False, action='store_true', help='evaluate pc w.r.t. alpha')
    parser.add_argument('--test-beta2-pc', default=False, action='store_true', help='evaluate pc with beta2 setting')
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
                                 'kd_mfd_logit_pairing', 'kd_mfd_balCE',
                                 'kd_indiv_output','kd_logit_pairing', 'kd_logit_pairing_to_org',
                                 'kd_indiv_logit_pairing',
                                 'kd_indiv_ukn1', 'kd_indiv_ukn2','kd_indiv_ukn3',
                                 'kd_hinton_aug', 'kd_fitnet_aug',
                                 'scratch_aug','logit_pairing','logit_pairing_ukn', 'logit_pairing_aug', 'group_dro',
                                 'feature_pairing','feature_pairing_mmd', 'kd_feature_pairing_to_org', 'logit_pairing_kd_logit_pairing', 'logit_pairing_kd_feature_pairing', 'logit_pairing_kd_mfd',
                                 'kd_hinton', 'kd_fitnet', 'kd_at','cov','kd_feature_pairing', 'kd_hinton_logit',
                                 'logit_pairing_kd_hinton',
                                 'scratch_mmd', 'kd_nst', 'adv_debiasing', 'fairdro','groupdro','lbc', 'sensei','sensei_2', 'group_predict','fairbatch','fairhsic',
                                 'logit_pairing_groupdro','logit_pairing_lbc',
                                 'ck_lp'])

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
    parser.add_argument('--kd-lambf', default=1, type=float, help='feature distill strength hyperparameter for kd-lp loss')

    parser.add_argument('--model', default='', required=True, choices=['resnet','resnet56', 'shufflenet', 'mlp', 'cifar_net', 'resnet152', 'cifar_net_v2'])
    parser.add_argument('--parallel', default=False, action='store_true', help='data parallel')
    parser.add_argument('--teacher-type', default=None, choices=['resnet', 'shufflenet', 'cifar_net', 'cifar_net_v2'])
    parser.add_argument('--teacher-path', default=None, help='teacher model path')

    parser.add_argument('--pretrained', default=False, action='store_true', help='load imagenet pretrained model')
    parser.add_argument('--num-workers', default=2, type=int, help='the number of thread used in dataloader')
    parser.add_argument('--term', default=20, type=int, help='the period for recording train acc')

    parser.add_argument('--target', default='Blond_Hair', type=str, help='target attribute for celeba')
    parser.add_argument('--sensitive', default='Male', type=str, help='sensitive attribute for celeba')
    parser.add_argument('--test-pc-G', default=None, type=str, help='test pc w.r.t. G, celeba (Wearing_Hat, Hair_Length, Bangs), celeba_hq (Hair_Length, Hair_Curl)')
    parser.add_argument('--test-set', default='original', type=str, choices=['original', 'strong_f', 'weak_f', 'pc_G'], help='test set for celeba / pc_g test set for celeba_hq')

    parser.add_argument('--fitnet-simul', default=False, action='store_true', help='no hint-training')

    parser.add_argument('--eta', default=0.0003, type=float, help='adversary training learning rate or lbc parameter')
    parser.add_argument('--adv-lambda', default=2.0, type=float, help='adversary loss strength')
    parser.add_argument('--iter', default=5, type=int, help='# of iteraion for lbc')

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
    parser.add_argument('--sensei-dist-path', type=str, help='mahalanobis distance file path')

    parser.add_argument('--clip-filtering', default=False, action='store_true', help='apply clip based filtering on generated counterfactual images')
    parser.add_argument('--test-img-cfg', default=2.0, type=float, help='test ctf image cfg')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.mode == 'train' and (args.method.startswith('kd')):
        if args.teacher_path is None:
            raise Exception('A teacher model path is not specified.')

    if args.mode == 'eval' and args.modelpath is None:
        raise Exception('Model path to load is not specified!')
    
    return args
