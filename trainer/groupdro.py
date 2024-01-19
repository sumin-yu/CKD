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

class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.test_criterion = torch.nn.CrossEntropyLoss()
        self.gamma = args.gamma
                        
    def train(self, train_loader, val_loader, test_loader, epochs):        
        
        global loss_set
        model = self.model
        model.train()
        
        n_classes = train_loader.dataset.num_classes
        n_groups = train_loader.dataset.num_groups
        
        self.normal_loader = DataLoader(train_loader.dataset, 
                                        batch_size=128, 
                                        shuffle=False, 
                                        num_workers=2, 
                                        pin_memory=True, 
                                        drop_last=False)
        
        self.adv_probs = torch.ones(n_groups*n_classes).cuda() / (n_groups*n_classes)

        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, model)            

            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deom,  = self.evaluate(self.model, 
                                                                test_loader, 
                                                                self.test_criterion,
                                                            )
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEOM {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method,
                   eval_loss, eval_acc, eval_deom, (eval_end_time - eval_start_time)))

            if self.scheduler != None and 'Reduce' in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()
                  
        print('Training Finished!')        

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
            inputs, _, groups, targets, _ = data if not self.aug_mode else self.dim_change(data)
            labels = targets
            
            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.cuda(device=self.device)

            outputs = model(inputs)
            loss = self.criterion(outputs, labels)

            # calculate the groupwise losses
            subgroups = groups * n_classes + labels
            group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
            group_count = group_map.sum(1)
            group_denom = group_count + (group_count==0).float() # avoid nans
            group_loss = (group_map @ loss.view(-1))/group_denom

            # update q
            self.adv_probs = self.adv_probs * torch.exp(self.gamma*group_loss.data)
            self.adv_probs = self.adv_probs/(self.adv_probs.sum()) # proj

            loss = group_loss @ self.adv_probs
                
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
        