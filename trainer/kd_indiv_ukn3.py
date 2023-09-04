from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import time
from utils import get_accuracy
from trainer.loss_utils import compute_hinton_loss
from sklearn.metrics import confusion_matrix
from trainer.kd_indiv import Trainer as indiv_Trainer
from trainer.kd_indiv import MMDLoss 
import os


class Trainer(indiv_Trainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)

    def train(self, train_loader, val_loader, test_loader, epochs):

        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups

        distiller = MMDLoss(w_m=self.lambf, sigma=self.sigma, batch_size=self.batch_size,
                            num_classes=num_classes, num_groups=num_groups, kernel=self.kernel)

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

            loss = self.criterion(logits_tot, labels)

            f_s = s_outputs[-2]
            f_t = t_outputs[-2]
            mmd_loss = distiller.forward(f_s, f_t, groups=groups, labels=targets) if self.lambf != 0 else 0 # n_set = 2

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
