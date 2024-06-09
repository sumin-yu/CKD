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

    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='SGD momentum')
    parser.add_argument('--epochs', default=50, type=int, help='number of training epochs')
    parser.add_argument('--batch-size', default=128, type=int, help='mini batch size')
    parser.add_argument('--seed', default=0, type=int, help='seed for randomness')

    parser.add_argument('--optimizer', default='Adam', type=str, required=False,
                        choices=['SGD', 'Adam','AdamW'],
                        help='(default=%(default)s)')

    # save dir
    parser.add_argument('--date', default='20xxxxxx', type=str, help='experiment date')

    # dataset
    parser.add_argument('--dataset', required=True, default='',
                        choices=['celeba', 'cifar10_b_same', 'cifar10_b_same_aug', 'celeba_aug', 'lfw', 'lfw_aug'])
    
    # CIFAR-10B 
    parser.add_argument('--skew-ratio', default=0.8, type=float, help='skew ratio (gamma) for cifar-10s')
    parser.add_argument('--editing-bias-alpha', default=0.8, type=float, help='editing bias alpha for cifar10-s')
    parser.add_argument('--group-bias-type', default='Contrast', type=str, choices=['Contrast'], help='group bias type for cifar-10s')
    parser.add_argument('--noise-type', default='Gaussian_Noise', type=str, choices=['Gaussian_Noise'], help='noise type for cifar-10s')
    parser.add_argument('--group-bias-degree', default=1, type=int, help='group bias degree for cifar-10s')
    parser.add_argument('--domain-gap-degree', default=0, type=int, help='domain gap degree for cifar-10s')
    parser.add_argument('--editing-bias-beta', default=0, type=int, help='editing bias degree for cifar-10s. degree0 means that ctf image is perfect <-> degree3 means that ctf image is made by overly biased editing model')
    parser.add_argument('--noise-degree', default=1, type=int, help='noise degree for cifar-10s')
    parser.add_argument('--noise-corr', default='pos', type=str, help='noise correlation for cifar-10s')
    parser.add_argument('--test-alpha-pc', default=False, action='store_true', help='evaluate pc w.r.t. alpha') # CIFAR-10B CD evaluation
    # parser.add_argument('--test-beta2-pc', default=False, action='store_true', help='evaluate pc with beta2 setting')
    parser.add_argument('--img-size', default=224, type=int, help='img size for preprocessing')
    
    parser.add_argument('--method', default='scratch', type=str, required=True,
                        choices=['scratch'
                                 ,'sensei', 'logit_pairing',
                                 'kd_mfd', 'cov', 'lbc', 'rw', 'kd_hinton',
                                 'logit_pairing_kd_mfd', 'logit_pairing_cov', 'logit_pairing_rw', 'logit_pairing_lbc',
                                 'kd_logit_pairing', 'kd_feature_pairing',
                                 'logit_pairing_kd_logit_pairing', 'logit_pairing_kd_feature_pairing',])
    parser.add_argument('--model', default='', required=True, choices=['resnet','resnet56'])
    parser.add_argument('--parallel', default=False, action='store_true', help='data parallel')
    parser.add_argument('--pretrained', default=False, action='store_true', help='load imagenet pretrained model')
    parser.add_argument('--num-workers', default=2, type=int, help='the number of thread used in dataloader')
    parser.add_argument('--term', default=20, type=int, help='the period for recording train acc')

    # sampling option
    parser.add_argument('--sampling', default='noBal', type=str, choices=['noBal', 'gBal','cBal', 'gcBal'], help='balanced sampling')
    # +aug option
    parser.add_argument('--ce-aug', default=False,action='store_true', help='use edited images in CE')
    # KD teacher model path
    parser.add_argument('--teacher-path', default=None, help='teacher model path')
    

    # CKD param
    parser.add_argument('--kd-lambf', default=1, type=float, help='distillation strength hyperparameter for kd-lp loss')

    # CP param
    parser.add_argument('--cp_lambf', default=1, type=float, help='feature distill strength hyperparameter')
    parser.add_argument('--kd-temp', default=3, type=float, help='temperature for KD')
    # SenSeI param
    parser.add_argument('--sensei-rho', default=5.0, type=float, help='rho for SenSeI')
    parser.add_argument('--sensei-eps', default=0.1, type=float, help='epsilon for SenSeI')
    parser.add_argument('--auditor-nsteps', default=100, type=int, help='auditor nsteps for SenSeI')
    parser.add_argument('--auditor-lr', default=1e-3, type=float, help='auditor lr for SenSeI')
    parser.add_argument('--epochs-dist', default=10, type=int, help='epochs for distance metric learning')
    parser.add_argument('--sensei-dist-path', type=str, help='mahalanobis distance file path')
    # COV param
    parser.add_argument('--cov-lambf', default=0, type=float, help='cov lambda when COV+CP')
    # LBC param
    parser.add_argument('--eta', default=0.0003, type=float, help='adversary training learning rate or lbc parameter')
    parser.add_argument('--iter', default=5, type=int, help='# of iteraion for lbc')
    # MFD param
    parser.add_argument('--mfd-lambf', default=1, type=float, help='feature distill strength hyperparameter')
    parser.add_argument('--lambh', default=0, type=float, help='kd strength hyperparameter')
    parser.add_argument('--sigma', default=1.0, type=float, help='sigma for rbf kernel')
    parser.add_argument('--kernel', default='rbf', type=str, choices=['rbf', 'poly', 'linear'], help='kernel for mmd')

    # target / sensitive attribute
    parser.add_argument('--target', default='Blond_Hair', type=str, help='target attribute for celeba')
    parser.add_argument('--sensitive', default='Male', type=str, help='sensitive attribute for celeba')

    parser.add_argument('--test-pc-G', default=None, type=str, help='test pc w.r.t. G, celeba (Hair_Length)')
    parser.add_argument('--test-set', default='original', type=str, choices=['original', 'strong_f', 'weak_f', 'pc_G'], help='test set for celeba / pc_g test set for celeba_hq')
    parser.add_argument('--test-img-cfg', default=2.0, type=float, help='test ctf image cfg')
    parser.add_argument('--get-inter', default=False, action='store_true',
                        help='get penultimate features for TSNE visualization')

    # parser.add_argument('--teacher-type', default=None, choices=['resnet', 'shufflenet', 'cifar_net', 'cifar_net_v2'])
    # parser.add_argument('--gamma', default=0, type=float, help='lopgitpairing strength hyperparameter for MFD-indiv')
    # parser.add_argument('--rho', default=0.5, type=float, help='the radioi of chi divergence ball')
    # parser.add_argument('--adv-lambda', default=2.0, type=float, help='adversary loss strength')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.mode == 'train' and (args.method.startswith('kd')):
        if args.teacher_path is None:
            raise Exception('A teacher model path is not specified.')

    if args.mode == 'eval' and args.modelpath is None:
        raise Exception('Model path to load is not specified!')
    
    return args
