from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
from utils import get_accuracy
from trainer.loss_utils import compute_hinton_loss
from sklearn.metrics import confusion_matrix
import trainer
import os


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.lambh = args.lambh
        self.lambf = args.lambf
        self.sigma = args.sigma
        self.kernel = args.kernel
        self.batch_size = args.batch_size
        self.jointfeature = args.jointfeature
        self.no_annealing = args.no_annealing

    def train(self, train_loader, val_loader, test_loader, epochs):

        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups
        num_aug = train_loader.dataset.num_aug

        distiller = MMDLoss(w_m=self.lambf, sigma=self.sigma, batch_size=self.batch_size,
                            num_classes=num_classes, num_groups=num_groups, num_aug=num_aug, kernel=self.kernel)

        for epoch in range(self.epochs):
            self._train_epoch(epoch, train_loader, self.model, self.teacher, distiller=distiller, num_aug=num_aug, num_groups=num_groups)

            val_loss, val_acc, val_deopp = self.evaluate(self.model, val_loader, self.criterion)
            print('[{}/{}] Method: {} '
                    'Val Loss: {:.3f} Val Acc: {:.2f} Val DEopp {:.2f}'.format
                    (epoch + 1, epochs, self.method,
                    val_loss, val_acc, val_deopp))
            
            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deopp = self.evaluate(self.model, test_loader, self.criterion)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEopp {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_deopp, (eval_end_time - eval_start_time)))

            if self.scheduler != None:
                self.scheduler.step(eval_loss)

        print('Training Finished!')

    def _train_epoch(self, epoch, train_loader, model, teacher, distiller=None, num_aug=1, num_groups=2):
        model.train()
        teacher.eval()

        running_acc = 0.0
        running_loss = 0.0
        batch_start_time = time.time()

        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs_, int_inputs_, _, groups_, targets_, index_ = data
            labels_ = targets_
            inputs_ = torch.stack(inputs_)
            int_inputs_ = torch.stack(int_inputs_)

            inputs = torch.reshape(inputs_, (-1, inputs_.shape[-3], inputs_.shape[-2], inputs_.shape[-1]))
            int_inputs = torch.reshape(int_inputs_, (-1, int_inputs_.shape[-3], int_inputs_.shape[-2], int_inputs_.shape[-1]))
            tot_inputs = torch.cat((inputs, int_inputs), dim=0)

            org_labels = labels_.repeat(num_aug)
            tot_labels = labels_.repeat(num_aug*num_groups)

            groups = groups_.repeat(num_aug)
            int_groups_ = torch.where(groups_ == 0, 1, 0)
            int_groups = int_groups_.repeat(num_aug)
            tot_groups = torch.cat((groups, int_groups), dim=0)

            tot_index = index_.repeat(num_aug*num_groups)

            if self.cuda:
                # inputs = inputs.cuda(self.device)
                # int_inputs = int_inputs.cuda(self.device)
                tot_inputs = tot_inputs.cuda(self.device)

                org_labels = org_labels.long().cuda(self.device)
                tot_labels = tot_labels.long().cuda(self.device)

                # groups = groups.long().cuda(self.device)
                # int_groups = int_groups.long().cuda(self.device)
                tot_groups = tot_groups.long().cuda(self.device)

            # t_inputs = inputs.to(self.t_device)
            t_inputs = tot_inputs.to(self.t_device)

            # outputs = model(inputs, get_inter=True)
            outputs_tot = model(tot_inputs, get_inter=True)
            logits_tot = outputs_tot[-1]

            t_outputs = teacher(t_inputs, get_inter=True)
            tea_logits = t_outputs[-1]

            kd_loss = compute_hinton_loss(logits_tot, t_outputs=tea_logits,
                                          kd_temp=self.kd_temp, device=self.device) if self.lambh != 0 else 0

            #outputs = model(inputs, get_inter=True)
            stu_logits = logits_tot[:len(logits_tot)//2]

            loss = self.criterion(stu_logits, org_labels)
            loss = loss + self.lambh * kd_loss


            f_s = outputs_tot[-2]
            f_t = t_outputs[-2]

            # mmd_loss = distiller.forward(f_s, f_t, groups=groups, labels=labels, jointfeature=self.jointfeature)
            mmd_loss = distiller.forward(f_s, f_t, groups=tot_groups, labels=tot_labels, jointfeature=self.jointfeature) if self.lambf != 0 else 0

            loss = loss + mmd_loss
            running_loss += loss.item()
            running_acc += get_accuracy(stu_logits, org_labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.term == self.term - 1:  # print every self.term mini-batches
                avg_batch_time = time.time() - batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i + 1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time / self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()

        if not self.no_annealing:
            self.lambh = self.lambh - 3 / (self.epochs - 1)
    

    def evaluate(self, model, loader, criterion, device=None, groupwise=False):
        model.eval()
        num_groups = loader.dataset.num_groups
        num_classes = loader.dataset.num_classes
        device = self.device if device is None else device

        eval_acc = 0 if not groupwise else torch.zeros(num_groups, num_classes).cuda(device)
        eval_loss = 0 if not groupwise else torch.zeros(num_groups, num_classes).cuda(device)
        eval_eopp_list = torch.zeros(num_groups, num_classes).cuda(device)
        eval_data_count = torch.zeros(num_groups, num_classes).cuda(device)
        
        if 'Custom' in type(loader).__name__:
            loader = loader.generate()
        with torch.no_grad():
            for j, eval_data in enumerate(loader):
                # Get the inputs
                inputs_, _, _, groups, classes, _ = eval_data
                #
                labels = classes 
                inputs = inputs_[0]

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
        intermediate_feature_set = torch.tensor([])
        
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                # Get the inputs
                inputs_, _,  _, groups, targets, _ = data
                labels = targets
                groups = groups.long()
                inputs = inputs_[0]
                
                if self.cuda:
                    inputs = inputs.cuda(self.device)
                    labels = labels.cuda(self.device)

                # forward

                outputs = self.model(inputs)
                if self.get_inter:
                    intermediate_feature = self.model.forward(inputs, get_inter=True)[-2]

                group_set = torch.cat((group_set, groups))
                target_set = torch.cat((target_set, targets))
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
        if self.get_inter:
            predict_mat['intermediate_feature_set'] = intermediate_feature_set.numpy()
            
        savepath = os.path.join(log_dir, log_name + '_{}_confu'.format(dataset))
        print('savepath', savepath)
        savemat(savepath, confu_mat, appendmat=True)

        savepath_pred = os.path.join(log_dir, log_name + '_{}_pred'.format(dataset))
        savemat(savepath_pred, predict_mat, appendmat=True)

        print('Computed confusion matrix for {} dataset successfully!'.format(dataset))
        return confu_mat

class MMDLoss(nn.Module):
    def __init__(self, w_m, batch_size, sigma, num_groups, num_classes, num_aug, kernel):
        super(MMDLoss, self).__init__()
        self.w_m = w_m
        self.sigma = sigma
        self.num_groups = num_groups
        self.batch_size = batch_size
        self.num_aug = num_aug
        self.num_classes = num_classes
        self.kernel = kernel

    def forward(self, f_s, f_t, groups, labels, jointfeature=False):
        if self.kernel == 'poly':
            student = F.normalize(f_s.view(f_s.shape[0], -1), dim=1)
            teacher = F.normalize(f_t.view(f_t.shape[0], -1), dim=1).detach()
        else:
            student = f_s.view(f_s.shape[0], -1)
            teacher = f_t.view(f_t.shape[0], -1).detach()

        mmd_loss = 0

        if jointfeature:
            K_TS, sigma_avg = self.pdist(teacher, student,
                              sigma_base=self.sigma, kernel=self.kernel)
            K_TT, _ = self.pdist(teacher, teacher, sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)
            K_SS, _ = self.pdist(student, student,
                              sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

            mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()

        else:
            with torch.no_grad():
                _, sigma_avg = self.pdist(teacher, student, sigma_base=self.sigma, kernel=self.kernel)

            t_order = torch.arange(self.batch_size*self.num_aug*self.num_groups)
            s_order = torch.arange(self.batch_size*self.num_aug)
            t_order = t_order.reshape(-1, self.batch_size).transpose(0,1).flatten()
            s_order = s_order.reshape(-1, self.batch_size).transpose(0,1).flatten()

            t_ref = torch.ones(self.batch_size, self.num_aug*self.num_groups,self.num_aug*self.num_groups, dtype=torch.int).cuda()
            s_ref = torch.ones(self.batch_size, self.num_aug,self.num_aug, dtype=torch.int).cuda()
            ts_ref = torch.ones(self.batch_size, self.num_aug*self.num_groups,self.num_aug, dtype=torch.int).cuda()
            t_ref = torch.block_diag(*t_ref)
            s_ref = torch.block_diag(*s_ref)
            ts_ref = torch.block_diag(*ts_ref)
            num_ind = (self.num_groups*self.num_aug)**2
            num_ind_s = self.num_aug**2
            num_ind_ts = self.num_groups*self.num_aug*self.num_aug

            for g in range(self.num_groups): # assume num_groups = 2 !!! # student 는 num_aug 개씩 같이 계산해야함 # teacher 는 num_aug * num_group 개씩 같이 계산해야함
                if len(student[(groups == g)]) == 0:
                    continue
                K_TS, _ = self.pdist(teacher, student[(groups == g)],
                                            sigma_base=self.sigma, sigma_avg=sigma_avg,  kernel=self.kernel)
                K_SS, _ = self.pdist(student[(groups == g)], student[(groups == g)],
                                    sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)
                K_TT, _ = self.pdist(teacher, teacher, sigma_base=self.sigma,
                                    sigma_avg=sigma_avg, kernel=self.kernel)
                K_TS = K_TS[t_order][:, s_order]
                K_SS = K_SS[s_order][:, s_order]
                K_TT = K_TT[t_order][:, t_order]
                K_TS = K_TS * ts_ref
                K_SS = K_SS * s_ref
                K_TT = K_TT * t_ref
                mmd_loss += K_TT.sum()/num_ind + K_SS.sum()/num_ind_s - 2 * K_TS.sum()/num_ind_ts

        loss = (1/2) * self.w_m * mmd_loss

        return loss

    @staticmethod
    def pdist(e1, e2, eps=1e-12, kernel='rbf', sigma_base=1.0, sigma_avg=None):
        if len(e1) == 0 or len(e2) == 0:
            res = torch.zeros(1)
        else:
            if kernel == 'rbf':
                e1_square = e1.pow(2).sum(dim=1)
                e2_square = e2.pow(2).sum(dim=1)
                prod = e1 @ e2.t()
                res = (e1_square.unsqueeze(1) + e2_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
                res = res.clone()
                sigma_avg = res.mean().detach() if sigma_avg is None else sigma_avg
                res = torch.exp(-res / (2*(sigma_base)*sigma_avg))
            elif kernel == 'poly':
                res = torch.matmul(e1, e2.t()).pow(2)

        return res, sigma_avg

