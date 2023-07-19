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


class Trainer(trainer.kd_indiv_ukn1.Trainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)

    def _train_epoch(self, epoch, train_loader, model, teacher, distiller=None, num_groups=2):
        model.train()
        teacher.eval()

        running_acc = 0.0
        running_loss = 0.0
        batch_start_time = time.time()

        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data 
            inputs = inputs.permute((1,0,2,3,4))
            inputs = inputs.contiguous().view(-1, *inputs.shape[2:])

            # targets = torch.stack((targets,targets),dim=0).view(-1)
            # groups = torch.stack((groups,groups),dim=0).view(-1)
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

            loss = self.criterion(logits_tot[:self.batch_size], labels[:self.batch_size])

            # f_s = s_outputs[-2][self.batch_size+1:]
            # f_t = t_outputs[-2][self.batch_size+1:]
            # groups_aug = groups[self.batch_size+1:]
            # targets_aug = targets[self.batch_size+1:]
            f_s = s_outputs[-2][self.batch_size:]
            f_t = t_outputs[-2]
            groups_aug = groups[self.batch_size:]
            targets_aug = targets[self.batch_size:]
            mmd_loss = distiller.forward(f_s, f_t, groups=groups_aug, labels=targets_aug, n_set=2) if self.lambf != 0 else 0 # n_set = 2

            f_s = s_outputs[-2][:self.batch_size]
            f_t = t_outputs[-2]
            groups_aug = groups[:self.batch_size]
            targets_aug = targets[:self.batch_size]
            targets_aug = torch.zeros_like(targets_aug)
            mmd_loss2 = distiller.forward(f_s, f_t, groups=groups_aug, labels=targets_aug, n_set=3) if self.lambf != 0 else 0

            loss = loss + mmd_loss + mmd_loss2
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
