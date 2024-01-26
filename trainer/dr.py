from __future__ import print_function
from collections import defaultdict
import time
from utils import get_accuracy
import trainer
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import cvxpy as cvx
# import dccp
# from dccp.problem import is_dccp
import numpy as np
import trainer.vanilla_train

class Trainer(trainer.vanilla_train.Trainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.lamb = args.lambf
        # self.criterion_train = nn.CrossEntropyLoss(reduction='none')
        
    def _train_epoch(self, epoch, train_loader, model):
        model.train()
        
        n_classes = train_loader.dataset.num_classes
        n_groups = train_loader.dataset.num_groups
        n_subgroups = n_classes * n_groups
        
        running_acc = 0.0
        running_loss = 0.0
        total = 0
        batch_start_time = time.time()
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data if not self.aug_mode else self.dim_change(data)
            labels = targets
            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.cuda(device=self.device)
                
            outputs = model(inputs)

            loss = self.criterion(outputs, labels)

            def closure_eo(loss, groups, labels, n_classes, n_groups, n_subgroups):
                if self.aug_mode:
                    loss = loss[:self.bs]
                    groups = groups[:self.bs]
                    labels = labels[:self.bs]
                subgroups = groups * n_classes + labels
                group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
                group_count = group_map.sum(1)
                group_denom = group_count + (group_count==0).float() # avoid nans
                group_loss = (group_map @ loss.view(-1))/group_denom

                group_loss_matrix = group_loss.reshape(n_groups, n_classes)
                abs_group_loss_diff = torch.abs(group_loss_matrix[0,:] - group_loss_matrix[1,:])
                eo_reg = abs_group_loss_diff.mean()
                return eo_reg
                
            reg_loss = closure_eo(loss, groups, labels, n_classes, n_groups, n_subgroups)
            ce_loss = loss.mean()
            loss = ce_loss + self.lamb * reg_loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
                
            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)
            
            if i % self.term == self.term-1: # print every self.term mini-batches
                avg_batch_time = time.time()-batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i+1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time/self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()
    
    