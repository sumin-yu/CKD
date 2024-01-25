from __future__ import print_function

import time
import os
import torch
import torch.nn as nn
from utils import get_accuracy
from collections import defaultdict
import torch.optim as optim
import trainer
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from trainer.vanilla_train import Trainer as vanilla_trainer

class Trainer(vanilla_trainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)

    def _train_epoch(self, epoch, train_loader, model):
        dummy_loader = DataLoader(train_loader.dataset, batch_size=self.bs, shuffle=False,
                                    num_workers=2, 
                                    pin_memory=True, drop_last=False)

        model.train()
        
        running_acc = 0.0
        running_loss = 0.0

        batch_start_time = time.time()
        self.adjust_lambda(model, train_loader, dummy_loader)
        
        for i, data in enumerate(train_loader):
            inputs, _, groups, labels, _ = data if not self.aug_mode else self.dim_change(data)
            if self.cuda:
                inputs = inputs.cuda().squeeze()
                labels = labels.cuda().squeeze()
                groups = groups.cuda()

            labels = labels.long()

            outputs = model(inputs)

            loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            # binary = True if num_classes ==2 else False
            running_acc += get_accuracy(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.term == self.term-1: # print every self.term mini-batches
                avg_batch_time = time.time()-batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.3f} '
                      '[{:.3f} s/batch]'.format
                      (epoch + 1, self.epochs, i+1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time/self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()
                
        return running_acc / self.term, running_loss / self.term
    
    def adjust_lambda(self, model, train_loader, dummy_loader):
        """Adjusts the lambda values for FairBatch algorithm.
        
        The detailed algorithms are decribed in the paper.
        """
        
        model.train()
        
        logits = []
        labels = []
        n_classes = train_loader.dataset.num_classes
        n_groups = train_loader.dataset.num_groups
        
        sampler = train_loader.sampler
        with torch.no_grad():
            for i, data in enumerate(dummy_loader):
                inputs, _, groups, _labels, _ = data if not self.aug_mode else self.dim_change(data)
                if self.aug_mode:
                    inputs = inputs[:self.bs]
                    _groups = _groups[:self.bs]
                    _labels = _labels[:self.bs]

                if self.cuda:
                    inputs = inputs.cuda()
                    _labels = _labels.cuda()
                    groups = groups.cuda()
                
                outputs = model(inputs)

                logits.append(outputs)
                labels.append(_labels)

        logits = torch.cat(logits)
        labels = torch.cat(labels)
        labels = labels.long()
        # TO DO
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        
        yhat_yz = {}
        yhat_y = {}
                    
#             eo_loss = criterion ((F.tanh(logits)+1)/2, (labels+1)/2)

        eo_loss = criterion(logits, labels.long())
        
        for tmp_yz in sampler.yz_tuple:
            yhat_yz[tmp_yz] = float(torch.sum(eo_loss[sampler.yz_index[tmp_yz]])) / sampler.yz_len[tmp_yz]
            
        for tmp_y in sampler.y_item:
            yhat_y[tmp_y] = float(torch.sum(eo_loss[sampler.y_index[tmp_y]])) / sampler.y_len[tmp_y]

        max_diff = 0
        pos = (0, 0)

        for _l in range(n_classes):
            # max_diff = 0
            # pos = 0
            for _g in range(1,n_groups):
                tmp_diff = abs(yhat_yz[(_l, _g)] - yhat_yz[(_l, _g-1)])
                if max_diff < tmp_diff:
                    max_diff = tmp_diff
                    pos = (_l, _g) if yhat_yz[(_l, _g)] >= yhat_yz[(_l, _g-1)] else (_l, _g-1)

        pos_label = pos[0]
        pos_group = pos[1]
        for _g in range(n_groups):
            if _g == pos_group:
                sampler.lbs[pos_label][_g] += sampler.gamma
            else:
                sampler.lbs[pos_label][_g] -= sampler.gamma / (n_groups-1)
            if sampler.lbs[pos_label][_g] > 1:
                sampler.lbs[pos_label][_g] = 1
            elif sampler.lbs[pos_label][_g] < 0:
                sampler.lbs[pos_label][_g] = 0

        #normalize
        sampler.lbs[pos_label] = [i / sum(sampler.lbs[pos_label]) for i in sampler.lbs[pos_label]]
                
        model.train()
