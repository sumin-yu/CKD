from __future__ import print_function
from collections import defaultdict

import copy
import time
from utils import get_accuracy
import trainer
import torch
import numpy as np

from torch.utils.data import DataLoader

from collections import defaultdict

import copy
import time
from utils import get_accuracy
import trainer
import torch
import numpy as np

from torch.utils.data import DataLoader
from trainer.lbc import Trainer as lbc_trainer

class Trainer(lbc_trainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.test_criterion = torch.nn.CrossEntropyLoss()
        self.cp_lambf = args.cp_lambf

    def _train_epoch(self, epoch, train_loader, model):
        model.train()
        
        running_acc = 0.0
        running_loss = 0.0
        batch_start_time = time.time()
        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups

        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, labels, _ = self.dim_change(data)
            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.cuda(device=self.device)
            
            groups = groups.long()
            labels = labels.long()
            
            weights = self.weight_matrix[groups, labels].cuda()
            if not self.ce_aug:
                weights = weights[:self.bs]            
            
            outputs = model(inputs)

            loss = torch.mean(weights * self.criterion(outputs, labels))

            ft_logit = outputs[:self.bs]
            ctf_logit = outputs[self.bs:]
            lp_loss = torch.mean((ft_logit-ctf_logit).pow(2))

            loss += self.cp_lambf * lp_loss if self.cp_lambf != 0.0 else 0
            
            self.optimizer.zero_grad()
            loss.backward()                
            self.optimizer.step()

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
