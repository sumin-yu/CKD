import torch
import numpy as np
import random
import os
import torch.nn.functional as F
import pickle

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


def get_metric(model, dataset, dataset_name, clip_filtering=False, is_pc_G=False):
    kwargs = {'num_workers': 4, 'pin_memory': True}

    bs = 1
    dataset.test_pair = True
    dataloader = torch.utils.data.DataLoader(dataset, bs, drop_last=False,
                                             shuffle=False, **kwargs)
    num_classes = dataset.num_classes
    model.eval()

    with torch.no_grad():
        num_as_pc = 0.
        num_as_pc_g = 0.
        pred_dist = 0.
        num_tot = 0.
        pc_mat = np.zeros((2, 2))
        num_tot_mat = np.zeros((2, 2))
        for data in dataloader:
            input, _, group, label , identity = data
            idx, img_name = identity
            group = group[0]
            num_tot += 1
            input = input.view(-1, *input.shape[2:])
            label = torch.reshape(label.permute((1,0)), (-1,))

            input = input.cuda()
            label = label.cuda()

            output = model(input, get_inter=True)
            logits = output[-1]
            features = output[-2]

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
        pc_g = num_as_pc_g / num_tot
        pc_mat /= num_tot_mat

    if is_pc_G:
        print('PC_G   = {}'.format(pc))
        return pred_dist, 0.0, pc, pc_mat
    else:
        print('PC     = {}'.format(pc))
        print('Sym-KL = {}'.format(pred_dist))
        return pred_dist, pc, 0.0, pc_mat

def make_log_name(args):
    log_name = args.model

    if args.mode == 'eval':
        log_name = args.modelpath.split('/')[-1]
        # remove .pt from name
        log_name = log_name[:-3]
        if args.test_alpha_pc:
            log_name += '_test_alpha_pc'
        if args.test_set != 'original':
            log_name += '_testset_{}'.format(args.test_set)
        if args.test_pc_G is not None:
            log_name += '_testpcG_{}'.format(args.test_pc_G)

    else:
        if args.pretrained:
            log_name += '_pretrained'
        log_name += '_seed{}_epochs{}_bs{}_lr{}'.format(args.seed, args.epochs, args.batch_size, args.lr)

        if args.method == 'logit_pairing_kd_logit_pairing' or args.method == 'logit_pairing_kd_feature_pairing':
            log_name += '_cp{}'.format(args.cp_lambf)
            log_name += '_ckd{}'.format(args.kd_lambf)
        elif 'logit_pairing' in args.method:
            log_name += '_cp{}'.format(args.cp_lambf)

        if 'kd_mfd' in args.method:
            log_name += '_{}'.format(args.kernel)
            log_name += '_sigma{}'.format(args.sigma) if args.kernel == 'rbf' else ''
            log_name += '_mfd{}'.format(args.mfd_lambf)

        # if 'feature_pairing' in args.method: 
        #     log_name += '_lambf{}'.format(args.cp_lambf)
            
        if 'cov' in args.method: 
            log_name += '_cov{}'.format(args.cov_lambf)

        if 'lbc' in args.method: 
            log_name += '_iter{}_eta{}'.format(args.iter, args.eta)

        if 'sensei' in args.method:
            log_name += '_rho{}'.format(args.sensei_rho)
            log_name += '_eps{}'.format(args.sensei_eps)
            log_name += '_nsteps{}'.format(args.auditor_nsteps)
            log_name += '_auditorlr{}'.format(args.auditor_lr)

        if args.teacher_path is not None:
            log_name += '_temp{}'.format(args.kd_temp)
            log_name += '_lambh{}'.format(args.lambh)

        if args.sampling != 'noBal':
            log_name += f'_{args.sampling}'

        if args.ce_aug:
            log_name += '_ce_aug'

        if 'celeba' in args.dataset or 'lfw' in args.dataset:
            log_name += '_{}_{}'.format(args.target, args.sensitive)

        if 'cifar10_b' in args.dataset:
            log_name += '_skewed{}'.format(args.skew_ratio)
            log_name += '_group{}_{}'.format(args.group_bias_type, args.group_bias_degree)
            if args.editing_bias_alpha == 0:
                log_name += '_domgap_{}_{}'.format(args.noise_type, args.domain_gap_degree)
            if args.editing_bias_alpha != 0.0:
                log_name += '_editbias_alpha{}'.format(args.editing_bias_alpha)
                log_name += '_beta{}'.format(args.editing_bias_beta)
                log_name += '_{}_{}'.format(args.noise_type, args.noise_degree)
                log_name += '_corr{}'.format(args.noise_corr)
            if args.test_alpha_pc:
                log_name += '_test_alpha_pc'

    return log_name

def save_anal(dataset='test', args=None, acc=0, pred_dist=0, pc=0, pc_g=0, pc_mat=[], deo_a=0, deo_m=0, log_dir="", log_name=""):

    savepath = os.path.join(log_dir, log_name + '_{}_result'.format(dataset))
    result = {}
    result['acc'] = acc
    result['pred_dist'] = pred_dist
    result['PC'] = pc
    result['DEO_A'] = deo_a
    result['DEO_M'] = deo_m
    result['test_pc'] = pc_g
    result['PC_mat'] = pc_mat
    result['args'] = args
    print('accuracy: {}'.format(acc), 'pred_dist: {}'.format(pred_dist), 'PC: {}'.format(pc), 'DEO_A: {}'.format(deo_a), 'DEO_M: {}'.format(deo_m), 'PC_G: {}'.format(pc_g))
    print('pc mat: {}'.format(pc_mat))
    print('success', savepath)
    # save result as pickle
    with open(savepath, 'wb') as f:
        pickle.dump(result, f)
