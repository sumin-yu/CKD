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


def get_accuracy(outputs, labels, binary=False):
    #if multi-label classification
    if len(labels.size())>1:
        outputs = (outputs>0.0).float()
        correct = ((outputs==labels)).float().sum()
        total = torch.tensor(labels.shape[0] * labels.shape[1], dtype=torch.float)
        avg = correct / total
        return avg.item()
    if binary:
        predictions = (torch.sigmoid(outputs) >= 0.5).float()
    else:
        predictions = torch.argmax(outputs, 1)
    c = (predictions == labels).float().squeeze()
    accuracy = torch.mean(c)
    return accuracy.item()


def check_log_dir(log_dir):
    try:
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
    except OSError:
        print("Failed to create directory!!")


def get_bmr(model, dataset):
    kwargs = {'num_workers': 4, 'pin_memory': True}

    bs = 2
    dataloader = torch.utils.data.DataLoader(dataset, bs, drop_last=False,
                                             shuffle=False, **kwargs)
    num_classes = dataset.num_classes
    model.eval()

    with torch.no_grad():
        num_as = 0.
        num_hits = 0.
        for data in dataloader:
            input, _, group, label , _ = data

            input = input.cuda()
            label = label.cuda()

            output = model(input)

            preds = torch.argmax(output, 1)
            # if label[0] < 5:
            #     if (preds[0] == label[0]):
            #         num_hits += 1
            #         if preds[1] != label[1]:
            #             num_as += 1
            # else:
            #     if (preds[1] == label[1]):
            #         num_hits += 1
            #         if preds[0] != label[0]:
            #             num_as += 1

            num_hits += (preds == label).sum()
            num_as += 1 if (preds == label).sum() == 1 else 0
            output = F.log_softmax(output, dim=1)

            # pred_dist += (F.kl_div(output[0], output[1], log_target=True, reduction='sum') +
            #               F.kl_div(output[1], output[0], log_target=True, reduction='sum')) / 2

        bmr = 0. if num_hits == 0. else num_as / num_hits
        # pred_dist /= len(dataloader)

    return bmr

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

        if args.method == 'adv_debiasing':
            log_name += '_advlamb{}_eta{}'.format(args.adv_lambda, args.eta)

        elif args.method == 'scratch_mmd':
            log_name += '_{}'.format(args.kernel)
            log_name += '_sigma{}'.format(args.sigma) if args.kernel == 'rbf' else ''
            log_name += '_lambf{}'.format(args.lambf)

        elif args.method == 'kd_mfd' or args.method=='kd_mfd_indiv':
            log_name += '_{}'.format(args.kernel)
            log_name += '_sigma{}'.format(args.sigma) if args.kernel == 'rbf' else ''
            log_name += '_lambf{}'.format(args.lambf)

        if args.teacher_path is not None and args.method != 'kd_Junyi':
            log_name += '_temp{}'.format(args.kd_temp)
            log_name += '_lambh{}'.format(args.lambh)

        if args.labelwise:
            log_name += '_labelwise'
        
        if args.method == 'kd_mfd_indiv' and args.with_perturbed:
            log_name+= '_wperturbed'

        # if args.dataset == 'celeba' and args.target != 'Attractive':
        #     log_name += '_{}'.format(args.target)
    return log_name

def save_anal(dataset='test', args=None, acc=0, bmr=0, deo_a=0, deo_m=0, log_dir="", log_name=""):

    savepath = os.path.join(log_dir, log_name + '_{}_result'.format(dataset))
    result = {}
    result['acc'] = acc
    result['BMR'] = bmr
    result['DEO_A'] = deo_a
    result['DEO_M'] = deo_m
    result['args'] = args
    print('success', savepath)
    # save result as pickle
    with open(savepath, 'wb') as f:
        pickle.dump(result, f)
