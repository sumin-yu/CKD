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
        # self.gamma = args.gamma # learning rate of adv_probs
        self.rho = args.rho
        self.train_criterion = torch.nn.CrossEntropyLoss(reduction='none')
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
        
        self.adv_probs_dict = {}
        for l in range(n_classes):
            n_data = train_loader.dataset.n_data[:,l]
            self.adv_probs_dict[l] = torch.ones(n_groups).cuda(device=self.device) / n_groups
        
        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, model)            

            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deom,  = self.evaluate(self.model, 
                                                                test_loader, 
                                                                self.criterion,
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
            inputs, _, groups, targets, idx = data
            labels = targets
            
            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.cuda(device=self.device)
                
            def closure():
                subgroups = groups * n_classes + labels
                outputs = model(inputs)

                loss = self.train_criterion(outputs, labels)
            
                # calculate the losses for each subgroup
                group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
                group_count = group_map.sum(1)
                group_denom = group_count + (group_count==0).float() # avoid nans
                group_loss = (group_map @ loss.view(-1))/group_denom
            
                robust_loss = 0
                for l in range(n_classes):
                    label_group_loss = group_loss[idxs+l]
                    robust_loss += label_group_loss @ self.adv_probs_dict[l]
                robust_loss /= n_classes        
                return outputs, robust_loss
            
            outputs, robust_loss = closure()
            
            self.optimizer.zero_grad()
            robust_loss.backward()                
            self.optimizer.step()

            running_loss += robust_loss.item()
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
        