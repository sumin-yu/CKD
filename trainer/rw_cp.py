from __future__ import print_function
import os
import numpy as np
import torch
import torch.nn as nn
import time
import torch.optim as optim
from utils import get_accuracy
from collections import defaultdict
import trainer
import pickle
from torch.utils.data import DataLoader
from trainer.rw import Trainer as rw_trainer

class Trainer(rw_trainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.cp_lambf = args.cp_lambf
                                                                          
    def _train_epoch(self, epoch, train_loader, model, weight_matrix, criterion=None):
        model.train()

        running_acc = 0.0
        running_loss = 0.0
        avg_batch_time = 0.0

        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups
        n_subgroups = num_classes * num_groups

        for i, data in enumerate(train_loader):
            batch_start_time = time.time()
            # Get the inputs
            inputs, _, groups, targets, _ = data if not self.aug_mode else self.dim_change(data)
            labels = targets
            groups = groups.long()
            labels = labels.long()

            weights = weight_matrix[groups, labels]
            
            if self.cuda:
                inputs = inputs.cuda()
                labels = labels.cuda()
                weights = weights.cuda()
                groups = groups.cuda()
                
            outputs = model(inputs)
                
            if self.balanced:
                subgroups = groups * num_classes + labels
                group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
                group_count = group_map.sum(1)
                group_denom = group_count + (group_count==0).float() # avoid nans
                loss = self.train_criterion(outputs, labels)
                group_loss = (group_map @ loss.view(-1))/group_denom
                loss = torch.mean(group_loss)
            else:
                if criterion is not None:
                    loss = criterion(outputs, labels).mean()
                else:
                    # loss = self.criterion(outputs, labels).mean()
                    loss = self.train_criterion(outputs, labels)
                    if self.aug_mode:
                        loss = (torch.mean(weights[:self.bs] * loss[:self.bs]) + torch.mean(loss[self.bs:])) / 2
                    else:
                        loss = torch.mean(weights * loss)

            ft_logit = outputs[:self.bs]
            ctf_logit = outputs[self.bs:]
            lp_loss = torch.mean((ft_logit-ctf_logit).pow(2))
            
            loss += self.cp_lambf * lp_loss if self.cp_lambf != 0.0 else 0.0

            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)

            batch_end_time = time.time()
            avg_batch_time += batch_end_time - batch_start_time

            if i % self.term == self.term - 1:  # print every self.term mini-batches
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i + 1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time / self.term))

                running_loss = 0.0
                running_acc = 0.0
                avg_batch_time = 0.0