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
    parser.add_argument('--evalset', default='test', choices=['all', 'train', 'test', 'val'])

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
                        choices=['celeba', 'cifar_10b', 'cifar_10b_aug', 'celeba_aug', 'lfw', 'lfw_aug'])
    parser.add_argument('--img-size', default=224, type=int, help='img size for preprocessing')
    
    # CIFAR-10B 
    parser.add_argument('--skew-ratio', default=0.8, type=float, help='skew ratio (gamma) for cifar-10s')
    parser.add_argument('--editing-bias-alpha', default=0.8, type=float, help='editing bias alpha for cifar_10b')
    
    parser.add_argument('--method', default='scratch', type=str, required=True,
                        choices=['scratch'
                                 ,'sensei', 'cp',
                                 'kd_mfd', 'cov', 'lbc', 'rw', 'kd_hinton',
                                 'kd_mfd_cp', 'cov_cp', 'rw_cp', 'lbc_cp',
                                 'ckd'])
    parser.add_argument('--model', default='', required=True, choices=['resnet','resnet56'])
    parser.add_argument('--parallel', default=False, action='store_true', help='data parallel')
    parser.add_argument('--pretrained', default=False, action='store_true', help='load imagenet pretrained model')
    parser.add_argument('--num-workers', default=2, type=int, help='the number of thread used in dataloader')
    parser.add_argument('--term', default=20, type=int, help='the period for recording train acc')

    # sampling option
    parser.add_argument('--sampling', default='noBal', type=str, choices=['noBal', 'gBal','cBal', 'gcBal'], help='balanced sampling')
    # +aug option
    parser.add_argument('--ce-aug', default=False,action='store_true', help='use ctf images in CE loss')
    # KD teacher model path
    parser.add_argument('--teacher-path', default=None, help='teacher model path')

    # CKD param
    parser.add_argument('--kd-lambf', default=0.0, type=float, help='CKD distillation strength hyperparameter')
    parser.add_argument('--rep', default=None, type=str, choices=['logit', 'feature'], help='representation vector type for CKD')

    # CP param
    parser.add_argument('--cp-lambf', default=0.0, type=float, help='CP distillation strength hyperparameter')
    # SenSeI param
    parser.add_argument('--sensei-rho', default=5.0, type=float, help='rho for SenSeI')
    parser.add_argument('--sensei-eps', default=0.1, type=float, help='epsilon for SenSeI')
    parser.add_argument('--auditor-nsteps', default=100, type=int, help='auditor nsteps for SenSeI')
    parser.add_argument('--auditor-lr', default=1e-3, type=float, help='auditor lr for SenSeI')
    parser.add_argument('--epochs-dist', default=10, type=int, help='epochs for distance metric learning')
    parser.add_argument('--sensei-dist-path', type=str, help='mahalanobis distance file path')
    # COV param
    parser.add_argument('--cov-lambf', default=0, type=float, help='cov hyperparameter')
    # LBC param
    parser.add_argument('--eta', default=0.0003, type=float, help='adversary training learning rate or lbc parameter')
    parser.add_argument('--iter', default=5, type=int, help='# of iteraion for lbc')
    # MFD param
    parser.add_argument('--mfd-lambf', default=1, type=float, help='mfd feature distill strength hyperparameter')
    parser.add_argument('--sigma', default=1.0, type=float, help='sigma for rbf kernel')
    parser.add_argument('--kernel', default='rbf', type=str, choices=['rbf', 'poly', 'linear'], help='kernel for mmd')
    # HKD param
    parser.add_argument('--lambh', default=0, type=float, help='kd_Hinton distillation strength hyperparameter')
    parser.add_argument('--kd-temp', default=3, type=float, help='temperature for kd_Hinton')

    # target / sensitive attribute
    parser.add_argument('--target', default='Blond_Hair', type=str, help='target attribute for celeba')
    parser.add_argument('--sensitive', default='Male', type=str, help='sensitive attribute for celeba')

    parser.add_argument('--test-set', default='original', type=str, choices=['original', 'cd'], help='test set for celeba lfw')
    parser.add_argument('--get-inter', default=False, action='store_true',
                        help='get penultimate features for TSNE visualization')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.mode == 'train' and (args.method.startswith('kd')):
        if args.teacher_path is None:
            raise Exception('A teacher model path is not specified.')
    if args.method == 'ckd':
        if args.rep is None:
            raise Exception('Representation type is not specified.')

    if args.mode == 'eval' and args.modelpath is None:
        raise Exception('Model path to load is not specified!')
    
    return args
