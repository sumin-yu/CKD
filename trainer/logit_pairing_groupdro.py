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
from trainer.groupdro import Trainer as groupdro_trainer

class Trainer(groupdro_trainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.lambf = args.lambf

    def _train_epoch(self, epoch, train_loader, model):

        model.train()
        
        running_acc = 0.0
        running_loss = 0.0
        batch_start_time = time.time()

        n_classes = train_loader.dataset.num_classes
        n_groups = train_loader.dataset.num_groups
        n_subgroups = n_classes * n_groups
        
        idxs = np.array([i * n_classes for i in range(n_groups)])            

        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = self.dim_change(data)
            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                targets = targets.cuda(device=self.device)
                groups = groups.cuda(device=self.device)

            outputs = model(inputs)
            loss = self.criterion(outputs, targets)

            if not self.ce_aug:
                group = group[:self.bs]
                target = target[:self.bs]

            # calculate the groupwise losses
            subgroups = groups * n_classes + targets
            group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
            group_count = group_map.sum(1)
            group_denom = group_count + (group_count==0).float() # avoid nans
            group_loss = (group_map @ loss.view(-1))/group_denom

            # update q
            self.adv_probs = self.adv_probs * torch.exp(self.gamma*group_loss.data)
            self.adv_probs = self.adv_probs/(self.adv_probs.sum()) # proj

            loss = group_loss @ self.adv_probs

            ft_logit = outputs[:self.bs]
            ctf_logit = outputs[self.bs:]
            lp_loss = torch.mean((ft_logit-ctf_logit).pow(2))
            loss += lp_loss * self.lambf

            self.optimizer.zero_grad()
            loss.backward()                
            self.optimizer.step()

            running_loss += loss.item()
            running_acc += get_accuracy(outputs, targets)
            if i % self.term == self.term-1: # print every self.term mini-batches
                avg_batch_time = time.time()-batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i+1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time/self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()
        
