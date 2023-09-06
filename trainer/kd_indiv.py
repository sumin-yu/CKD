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
        self.num_aug = args.num_aug

    def train(self, train_loader, val_loader, test_loader, epochs):
        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups

        distiller = MMDLoss(w_m=self.lambf, sigma=self.sigma, batch_size=self.batch_size,
                            num_classes=num_classes, num_groups=num_groups, kernel=self.kernel, num_aug=self.num_aug)

        for epoch in range(self.epochs):
            self._train_epoch(epoch, train_loader, self.model, self.teacher, distiller=distiller, num_groups=num_groups)

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

    def _train_epoch(self, epoch, train_loader, model, teacher, distiller=None, num_groups=2):
        model.train()
        teacher.eval()

        running_acc = 0.0
        running_loss = 0.0
        batch_start_time = time.time()

        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data # 128 x num_aug * 2 x ...
            inputs = inputs.permute((1,0,2,3,4))  # num_aug x 128 x ...
            inputs = inputs.contiguous().view(-1, *inputs.shape[2:]) # (num_augx128) x ...
            
            groups = torch.reshape(groups.permute((1,0)), (-1,))
            targets = torch.reshape(targets.permute((1,0)), (-1,)).type(torch.LongTensor)

            labels = targets 

            if self.cuda:
                inputs = inputs.cuda(self.device)
                labels = labels.long().cuda(self.device)
                groups = groups.long().cuda(self.device)

            t_inputs = inputs.to(self.t_device)

            s_outputs = model(inputs, get_inter=True)
            logits_tot = s_outputs[-1]

            t_outputs = teacher(t_inputs, get_inter=True)
            tea_logits = t_outputs[-1]

            loss = self.criterion(logits_tot[:len(logits_tot)//(2*self.num_aug)], labels[:len(labels)//(2*self.num_aug)])

            f_s = s_outputs[-2]
            f_t = t_outputs[-2]
            mmd_loss = distiller.forward(f_s, f_t, groups=groups, labels=labels) if self.lambf != 0 else 0

            loss = loss + mmd_loss
            running_loss += loss.item()
            running_acc += get_accuracy(logits_tot[:len(logits_tot)//2], labels[:len(labels)//2])

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

class MMDLoss(nn.Module):
    def __init__(self, w_m, batch_size, sigma, num_groups, num_classes, kernel, num_aug=1):
        super(MMDLoss, self).__init__()
        self.w_m = w_m
        self.sigma = sigma
        self.num_groups = num_groups
        self.batch_size = batch_size
        self.num_aug = num_aug
        self.num_classes = num_classes
        self.kernel = kernel

    def forward(self, f_s, f_t, groups, labels):
        if self.kernel == 'poly':
            student = F.normalize(f_s.view(f_s.shape[0], -1), dim=1)
            teacher = F.normalize(f_t.view(f_t.shape[0], -1), dim=1).detach()
        else:
            student = f_s.view(f_s.shape[0], -1)
            teacher = f_t.view(f_t.shape[0], -1).detach()

        mmd_loss = 0

        with torch.no_grad():
            _, sigma_avg = self.pdist(teacher, student, sigma_base=self.sigma, kernel=self.kernel)

        t_order = torch.arange(self.batch_size*self.num_aug*self.num_groups)
        s_order = torch.arange(self.batch_size*self.num_aug)
        t_order = t_order.reshape(-1, self.batch_size).transpose(0,1).flatten() # 0,batch_size,batch_size*2, batch_size*3, 1, batch_size+1, batch_size+2, batch_size+3,...
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
            elif kernel == 'linear':
                res = torch.matmul(e1, e2.t())

        return res, sigma_avg

