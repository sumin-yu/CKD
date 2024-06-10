import torch
import numpy as np
import random
import os
import torch.nn.functional as F
import pickle
import data_handler

def list_files(root, suffix, prefix=False):
    root = os.path.expanduser(root)
    files = list(
        filter(
            lambda p: os.path.isfile(os.path.join(root, p)) and p.endswith(suffix),
            os.listdir(root)
        )
    )
    if prefix is True:
        files = [os.path.join(root, d) for d in files]
    return files

def set_seed(seed):
    torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def get_accuracy(output, labels, binary=False):
    #if multi-label classification
    if len(labels.size())>1:
        output = (output>0.0).float()
        correct = ((output==labels)).float().sum()
        total = torch.tensor(labels.shape[0] * labels.shape[1], dtype=torch.float)
        avg = correct / total
        return avg.item()
    if binary:
        predictions = (torch.sigmoid(output) >= 0.5).float()
    else:
        predictions = torch.argmax(output, 1)
    c = (predictions == labels).float().squeeze()
    accuracy = torch.mean(c)
    return accuracy.item()

def check_log_dir(log_dir):
    try:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    except OSError:
        print("Failed to create directory!!")


def get_cd(model, args, n_data):
    _,_,_,_,dataloader = data_handler.DataloaderFactory.get_dataloader(args.dataset, img_size=args.img_size,
                                                            batch_size=1, seed=args.seed,
                                                            num_workers=args.num_workers,
                                                            target=args.target,
                                                            sensitive=args.sensitive,
                                                            skew_ratio=args.skew_ratio,
                                                            sampling=args.sampling,
                                                            method=args.method,
                                                            editing_bias_alpha=args.editing_bias_alpha,
                                                            test_alpha_pc=True,
                                                            test_set='cd',
                                                            )
    model.eval()
    cd_n_data = dataloader.dataset.n_data
    with torch.no_grad():
        num_as_pc = 0.
        pred_dist = 0.
        num_tot = 0.
        pc_mat = np.zeros((2, 2))
        num_tot_mat = np.zeros((2, 2))
        for data in dataloader:
            input, _, group, label , identity = data
            group = group[0]
            num_tot += 1
            input = input.view(-1, *input.shape[2:])
            label = torch.reshape(label.permute((1,0)), (-1,))

            input = input.cuda()
            label = label.cuda()

            output = model(input, get_inter=True)
            logits = output[-1]

            preds = torch.argmax(logits, 1)

            # symKL
            logits = F.log_softmax(logits, dim=1)
            pred_dist += (F.kl_div(logits[0], logits[1], log_target=True, reduction='sum') +
                        F.kl_div(logits[1], logits[0], log_target=True, reduction='sum')) / 2
            
            # pc
            num_as_pc +=  (preds[0] == (preds[1])).sum()

            pc_mat[int(group[0].item()), int(label[0].item())] += (preds[0] == (preds[1])).sum()
            num_tot_mat[int(group[0].item()), int(label[0].item())] += 1

        pred_dist /= num_tot
        pc = num_as_pc / num_tot
        pc_mat = pc_mat / num_tot_mat
        print('our CD: ', 100 - 100* ((((pc_mat[0,0] * (n_data[0,0]/cd_n_data[0,0]) + pc_mat[1,0] * (n_data[1,0]/cd_n_data[1,0])) / ((n_data[0,0]/cd_n_data[0,0]) + (n_data[1,0]/cd_n_data[1,0])) )
                                       + ((pc_mat[1,1] * (n_data[1,1]/cd_n_data[1,1]) + pc_mat[0,1] * (n_data[0,1]/cd_n_data[0,1])) / ((n_data[1,1]/cd_n_data[1,1]) + (n_data[0,1]/cd_n_data[0,1])))) / 2) )

    return pred_dist, pc, pc_mat

def make_log_name(args):
    log_name = args.model

    if args.mode == 'eval':
        log_name = args.modelpath.split('/')[-1]
        # remove .pt from name
        log_name = log_name[:-3]
    else:
        if args.pretrained:
            log_name += '_pretrained'
        log_name += '_seed{}_epochs{}_bs{}_lr{}'.format(args.seed, args.epochs, args.batch_size, args.lr)

        if args.method == 'ckd' :
            if args.cp_lambf != 0.0:
                log_name += '_cp{}'.format(args.cp_lambf)
            log_name += '_ckd_f{}'.format(args.kd_lambf) if args.rep == 'feature' else '_ckd_l{}'.format(args.kd_lambf)
        
        if 'cp' in args.method:
            log_name += '_cp{}'.format(args.cp_lambf)

        if 'kd_mfd' in args.method:
            log_name += '_{}'.format(args.kernel)
            log_name += '_sigma{}'.format(args.sigma) if args.kernel == 'rbf' else ''
            log_name += '_mfd{}'.format(args.mfd_lambf)
        
        if args.method == 'kd_hinton':
            log_name += '_temp{}'.format(args.kd_temp)
            log_name += '_hinton{}'.format(args.lambh)
            
        if 'cov' in args.method: 
            log_name += '_cov{}'.format(args.cov_lambf)

        if 'lbc' in args.method: 
            log_name += '_iter{}_eta{}'.format(args.iter, args.eta)

        if 'sensei' in args.method:
            log_name += '_rho{}'.format(args.sensei_rho)
            log_name += '_eps{}'.format(args.sensei_eps)
            log_name += '_nsteps{}'.format(args.auditor_nsteps)
            log_name += '_auditorlr{}'.format(args.auditor_lr)

        if args.sampling != 'noBal':
            log_name += f'_{args.sampling}'

        if args.ce_aug:
            log_name += '_ce_aug'

        if 'celeba' in args.dataset or 'lfw' in args.dataset:
            log_name += '_{}_{}'.format(args.target, args.sensitive)

        if 'cifar10_b' in args.dataset:
            log_name += '_skewed{}'.format(args.skew_ratio)
            if args.editing_bias_alpha != 0.0:
                log_name += '_editbias_alpha{}'.format(args.editing_bias_alpha)

    return log_name

def save_anal(dataset='test', args=None, acc=0, deo_a=0, deo_m=0, pred_dist=0, pc=0, pc_mat=[], log_dir="", log_name=""):

    savepath = os.path.join(log_dir, log_name + '_{}_result'.format(dataset))
    result = {}
    result['acc'] = acc
    result['DEO_A'] = deo_a
    result['DEO_M'] = deo_m
    if dataset == 'test':
        result['pred_dist'] = pred_dist
        result['PC'] = pc.item()
        result['PC_mat'] = pc_mat
    result['args'] = args
    print('accuracy: {}'.format(100*acc),  'CD: {}'.format(100-100*pc), 'DEO_M: {}'.format(100*deo_m))
    print('success', savepath)
    # save result as pickle
    with open(savepath, 'wb') as f:
        pickle.dump(result, f)
