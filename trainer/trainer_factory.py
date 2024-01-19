import torch
import numpy as np
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from sklearn.metrics import confusion_matrix
from utils import make_log_name


class TrainerFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_trainer(method, **kwargs):
        if method == 'scratch':
            import trainer.vanilla_train as trainer
        elif method == 'scratch_aug':
            import trainer.vanilla_train_aug as trainer
        elif method == 'logit_pairing':
            import trainer.logit_pairing as trainer
        elif method == 'cov':
            import trainer.cov as trainer
        elif method == 'feature_pairing':
            import trainer.feature_pairing as trainer
        elif method == 'feature_pairing_mmd':
            import trainer.feature_pairing_mmd as trainer
        elif method == 'logit_pairing_ukn':
            import trainer.logit_pairing_ukn as trainer
        elif method == 'logit_pairing_aug':
            import trainer.logit_pairing_aug as trainer
        elif method == 'feature_pairing':
            import trainer.feature_pairing as trainer
        elif method == 'kd_logit_pairing':
            import trainer.kd_logit_pairing as trainer
        elif method == 'kd_logit_pairing_to_org':
            import trainer.kd_logit_pairing_to_org as trainer
        elif method == 'logit_pairing_kd_logit_pairing':
            import trainer.logit_pairing_kd_logit_pairing as trainer
        elif method == 'logit_pairing_kd_feature_pairing':
            import trainer.logit_pairing_kd_feature_pairing as trainer
        elif method == 'logit_pairing_kd_mfd':
            import trainer.logit_pairing_kd_mfd as trainer
        elif method == 'kd_feature_pairing':
            import trainer.kd_feature_pairing as trainer
        elif method == 'kd_feature_pairing_to_org':
            import trainer.kd_feature_pairing_to_org as trainer
        elif method == 'kd_hinton':
            import trainer.kd_hinton as trainer
        elif method == 'kd_hinton_logit':
            import trainer.kd_hinton_logit as trainer
        elif method == 'logit_pairing_kd_hinton':
            import trainer.logit_pairing_kd_hinton as trainer
        elif method == 'kd_fitnet':
            import trainer.kd_fitnet as trainer
        elif method == 'kd_hinton_aug':
            import trainer.kd_hinton_aug as trainer
        elif method == 'kd_fitnet_aug':
            import trainer.kd_fitnet_aug as trainer
        elif method == 'kd_mfd':
            import trainer.kd_mfd as trainer
        elif method == 'kd_mfd_balCE':
            import trainer.kd_mfd_balCE as trainer
        elif method == 'kd_mfd_aug':
            import trainer.kd_mfd_aug as trainer
        elif method == 'kd_mfd_ctf':
            import trainer.kd_mfd_ctf as trainer
        elif method == 'kd_mfd_ctf_ukn':
            import trainer.kd_mfd_ctf_ukn as trainer
        elif method == 'kd_mfd_ctf_ukn3':
            import trainer.kd_mfd_ctf_ukn3 as trainer
        elif method == 'kd_mfd_logit_pairing':
            import trainer.kd_mfd_logit_pairing as trainer
        elif method == 'kd_indiv':
            import trainer.kd_indiv as trainer
        elif method == 'kd_indiv_output':
            import trainer.kd_indiv_output as trainer
        elif method == 'kd_indiv_aug':
            import trainer.kd_indiv_aug as trainer
        elif method == 'kd_indiv_multi':
            import trainer.kd_indiv_multi as trainer
        elif method == 'kd_indiv_ukn1':
            import trainer.kd_indiv_ukn1 as trainer
        elif method == 'kd_indiv_ukn2':
            import trainer.kd_indiv_ukn2 as trainer
        elif method == 'kd_indiv_ukn3':
            import trainer.kd_indiv_ukn3 as trainer
        elif method == 'kd_indiv_logit_pairing':
            import trainer.kd_indiv_logit_pairing as trainer
        elif method == 'scratch_mmd':
            import trainer.scratch_mmd as trainer
        elif method == 'adv_debiasing':
            import trainer.adv_debiasing as trainer
        elif method == 'kd_at':
            import trainer.kd_at as trainer
        elif method == 'kd_nst':
            import trainer.kd_nst as trainer
        elif method == 'fairdro':
            import trainer.fairdro as trainer
        elif method == 'groupdro':
            import trainer.groupdro as trainer
        elif method == 'logit_pairing_groupdro':
            import trainer.logit_pairing_groupdro as trainer
        elif method == 'lbc':
            import trainer.lbc as trainer
        elif method == 'logit_pairing_lbc':
            import trainer.logit_pairing_lbc as trainer
        elif method == 'sensei':
            import trainer.sensei as trainer
        elif method == 'sensei_2':
            import trainer.sensei_2 as trainer
        elif method == 'group_predict':
            import trainer.group_predict as trainer
        elif method == 'ck_lp':
            import trainer.check_lp as trainer
        else:
            raise Exception('Not allowed method')
        return trainer.Trainer(**kwargs)


class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''
    def __init__(self, model, args, optimizer, teacher=None):
        self.get_inter = args.get_inter
        self.cuda = args.cuda
        self.device = args.device
        self.t_device = args.t_device
        self.term = args.term
        self.lr = args.lr
        self.parallel = args.parallel
        self.epochs = args.epochs
        self.method = args.method
        self.model = model
        self.teacher = teacher
        self.optimizer = optimizer
        self.bs = args.batch_size
        self.optim_type = args.optimizer
        self.img_size = args.img_size if not 'cifar10' in args.dataset else 32
        if 'sensei' in  args.method :
            self.criterion = nn.CrossEntropyLoss(reduction='none')
        else:
            self.criterion=nn.CrossEntropyLoss()
        self.scheduler = None

        self.log_name = make_log_name(args)
        self.log_dir = os.path.join(args.log_dir, args.date, args.dataset, args.method)
        self.save_dir = os.path.join(args.save_dir, args.date, args.dataset, args.method)

        if self.optim_type == 'Adam' and self.optimizer is not None:
            self.scheduler = ReduceLROnPlateau(self.optimizer)
        elif self.optim_type == 'AdamW' and self.optimizer is not None:
            self.scheduler = ReduceLROnPlateau(self.optimizer)
        else: 
            self.scheduler = ReduceLROnPlateau(self.optimizer)

    def evaluate(self, model, loader, criterion, device=None, groupwise=False):
        model.eval()
        num_groups = loader.dataset.num_groups
        num_classes = loader.dataset.num_classes
        device = self.device if device is None else device

        eval_acc = 0 if not groupwise else torch.zeros(num_groups, num_classes).cuda(device)
        eval_loss = 0 if not groupwise else torch.zeros(num_groups, num_classes).cuda(device)
        eval_eopp_list = torch.zeros(num_groups, num_classes).cuda(device)
        eval_data_count = torch.zeros(num_groups, num_classes).cuda(device)
        n_subgroups = num_classes * num_groups
        group_count = torch.zeros(n_subgroups).cuda()
        group_acc = torch.zeros(num_groups, num_classes).cuda(device)
        
        if 'Custom' in type(loader).__name__:
            loader = loader.generate()
        with torch.no_grad():
            for j, eval_data in enumerate(loader):
                # Get the inputs
                inputs, _, groups, classes, _ = eval_data
                #
                labels = classes 
                if self.cuda:
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)
                    groups = groups.cuda(device)

                outputs = model(inputs)    
                
                if groupwise:
                    if self.cuda:
                        groups = groups.cuda(device)
                    loss = nn.CrossEntropyLoss(reduction='none')(outputs, labels)
                    preds = torch.argmax(outputs, 1)
                    acc = (preds == labels).float().squeeze()
                    for g in range(num_groups):
                        for l in range(num_classes):
                            eval_loss[g, l] += loss[(groups == g) * (labels == l)].sum()
                            eval_acc[g, l] += acc[(groups == g) * (labels == l)].sum()
                            eval_data_count[g, l] += torch.sum((groups == g) * (labels == l))
                else:
                    criterion = nn.CrossEntropyLoss()
                    loss = criterion(outputs, labels)
                    eval_loss += loss.item() * len(labels)
                    preds = torch.argmax(outputs, 1)
                    acc = (preds == labels).float().squeeze()
                    eval_acc += acc.sum()

                    for g in range(num_groups):
                        for l in range(num_classes):
                            eval_eopp_list[g, l] += acc[(groups == g) * (labels == l)].sum()
                            eval_data_count[g, l] += torch.sum((groups == g) * (labels == l))

            eval_loss = eval_loss / eval_data_count.sum() if not groupwise else eval_loss / eval_data_count
            eval_acc = eval_acc / eval_data_count.sum() if not groupwise else eval_acc / eval_data_count
            eval_eopp_list = eval_eopp_list / eval_data_count
            eval_max_eopp = torch.max(eval_eopp_list, dim=0)[0] - torch.min(eval_eopp_list, dim=0)[0]
            eval_max_eopp = torch.max(eval_max_eopp).item()
        model.train()
        return eval_loss, eval_acc, eval_max_eopp
    
    def save_model(self, save_dir, log_name="", model=None):
        model_to_save = self.model if model is None else model
        model_savepath = os.path.join(save_dir, log_name + '.pt')
        torch.save(model_to_save.state_dict(), model_savepath)

        print('Model saved to %s' % model_savepath)

    def compute_confusion_matix(self, dataset='test', num_classes=2,
                                dataloader=None, log_dir="", log_name=""):
        from scipy.io import savemat
        from collections import defaultdict
        self.model.eval()
        confu_mat = defaultdict(lambda: np.zeros((num_classes, num_classes)))
        print('# of {} data : {}'.format(dataset, len(dataloader.dataset)))

        predict_mat = {}
        output_set = torch.tensor([])
        group_set = torch.tensor([], dtype=torch.long)
        target_set = torch.tensor([], dtype=torch.long)
        # g_group_set = torch.tensor([], dtype=torch.long)
        intermediate_feature_set = torch.tensor([])
        alpha_set = torch.tensor([])
        alpha_ctf_set = torch.tensor([])
        
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                # Get the inputs
                inputs, g_groups, groups, targets, _ = data
                labels = targets
                groups = groups.long()
                # g_groups, _ = g_groups
                # g_groups = g_groups.long()

                if self.cuda:
                    inputs = inputs.cuda(self.device)
                    labels = labels.cuda(self.device)

                # forward

                outputs = self.model(inputs)
                if self.get_inter:
                    intermediate_feature = self.model.forward(inputs, get_inter=True)[-2]

                group_set = torch.cat((group_set, groups))
                target_set = torch.cat((target_set, targets))
                # g_group_set = torch.cat((g_group_set, g_groups))
                output_set = torch.cat((output_set, outputs.cpu()))
                if self.get_inter:
                    intermediate_feature_set = torch.cat((intermediate_feature_set, intermediate_feature.cpu()))

                pred = torch.argmax(outputs, 1)
                group_element = list(torch.unique(groups).numpy())
                for i in group_element:
                    mask = groups == i
                    if len(labels[mask]) != 0:
                        confu_mat[str(i)] += confusion_matrix(
                            labels[mask].cpu().numpy(), pred[mask].cpu().numpy(),
                            labels=[i for i in range(num_classes)])

        predict_mat['group_set'] = group_set.numpy()
        predict_mat['target_set'] = target_set.numpy()
        predict_mat['output_set'] = output_set.numpy()
        # predict_mat['g_group_set'] = g_group_set.numpy()
        if self.get_inter:
            predict_mat['intermediate_feature_set'] = intermediate_feature_set.numpy()

        savepath = os.path.join(log_dir, log_name + '_{}_confu'.format(dataset))
        print('savepath', savepath)
        savemat(savepath, confu_mat, appendmat=True)

        savepath_pred = os.path.join(log_dir, log_name + '_{}_pred'.format(dataset))
        savemat(savepath_pred, predict_mat, appendmat=True)

        print('Computed confusion matrix for {} dataset successfully!'.format(dataset))

        group0 = confu_mat['0']
        group1 = confu_mat['1']
        all = group0 + group1
        acc = np.sum(np.diag(all)) / np.sum(all)

        group0_acc_classwise = np.diag(group0) / np.sum(group0, axis=1)
        group1_acc_classwise = np.diag(group1) / np.sum(group1, axis=1)

        DEO_A = np.sum(abs(group0_acc_classwise - group1_acc_classwise)) / num_classes
        DEO_M = np.max(abs(group0_acc_classwise - group1_acc_classwise))

        return acc, DEO_A, DEO_M
