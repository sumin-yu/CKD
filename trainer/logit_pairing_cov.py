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
from trainer.cov import Trainer as cov_trainer

class Trainer(cov_trainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.lambf = args.cp_lambf
        self.cov_lambf = args.cov_lambf
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


                
            loss = self.criterion(outputs, labels).mean()
        
            def closure_FPR(inputs, groups, labels, outputs):
                if self.aug_mode:
                    inputs = inputs[:self.bs]
                    groups = groups[:self.bs]
                    labels = labels[:self.bs]
                    outputs = outputs[:self.bs]
                groups_onehot = torch.nn.functional.one_hot(groups.long(), num_classes=n_groups)
                groups_onehot = groups_onehot.float() # n by g

                d_theta = torch.diff(outputs, dim=1) # w1Tx - w0Tx + b1-b0  # n by 1
                d_theta_new = -(labels.view(-1,1)-1)*(2*labels.view(-1,1)-1)*d_theta
                g_theta = torch.minimum(d_theta_new, torch.tensor(0)) # n by 1
                z_bar = torch.mean(groups_onehot, dim=0) # 1 by g
                loss_groupwise = torch.abs(torch.mean((groups_onehot - z_bar)*g_theta, dim=0))
                loss = torch.sum(loss_groupwise)
                return loss

            def closure_FNR(inputs, groups, labels, outputs):
                if self.aug_mode:
                    inputs = inputs[:self.bs]
                    groups = groups[:self.bs]
                    labels = labels[:self.bs]
                    outputs = outputs[:self.bs]
                groups_onehot = torch.nn.functional.one_hot(groups.long(), num_classes=n_groups)
                groups_onehot = groups_onehot.float() # n by g

                d_theta = torch.diff(outputs, dim=1) # w1Tx - w0Tx + b1-b0  # n by 1
                d_theta_new = (labels.view(-1,1))*(2*labels.view(-1,1)-1)*d_theta
                g_theta = torch.minimum(d_theta_new, torch.tensor(0)) # n by 1
                z_bar = torch.mean(groups_onehot, dim=0) # 1 by g
                loss_groupwise = torch.abs(torch.mean((groups_onehot - z_bar)*g_theta, dim=0))
                loss = torch.sum(loss_groupwise)
                return loss
        
            loss += self.cov_lambf*closure_FPR(inputs, groups, labels, outputs)
            loss += self.cov_lambf*closure_FNR(inputs, groups, labels, outputs)

            ft_logit = outputs[:self.bs]
            ctf_logit = outputs[self.bs:]
            lp_loss = torch.mean((ft_logit-ctf_logit).pow(2))

            loss += self.lambf * lp_loss
            
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
    
    
    def calculate_covariance(self, model, train_loader):
        model.train()
        
        n_groups = train_loader.dataset.num_groups
        
        FPR_total = 0
        FNR_total = 0
        for i, data in enumerate(train_loader):
            inputs, _, groups, targets, idx = data
            labels = targets
            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.cuda(device=self.device)

            def closure_FPR(inputs, groups, labels, model):
                groups_onehot = torch.nn.functional.one_hot(groups.long(), num_classes=n_groups)
                groups_onehot = groups_onehot.float() # n by g

                outputs = model(inputs) # n by 2

                d_theta = torch.diff(outputs, dim=1) # w1Tx - w0Tx + b1-b0  # n by 1
                d_theta_new = -(labels.view(-1,1)-1)*(2*labels.view(-1,1)-1)*d_theta
                g_theta = torch.minimum(d_theta_new, torch.tensor(0)) # n by 1
                z_bar = torch.mean(groups_onehot, dim=0) # 1 by g
                loss_groupwise = torch.abs(torch.mean((groups_onehot - z_bar)*g_theta, dim=0))
                loss = torch.sum(loss_groupwise)
                return loss

            def closure_FNR(inputs, groups, labels, model):
                groups_onehot = torch.nn.functional.one_hot(groups.long(), num_classes=n_groups)
                groups_onehot = groups_onehot.float() # n by g
                outputs = model(inputs) # n by 2

                d_theta = torch.diff(outputs, dim=1) # w1Tx - w0Tx + b1-b0  # n by 1
                d_theta_new = (labels.view(-1,1))*(2*labels.view(-1,1)-1)*d_theta
                g_theta = torch.minimum(d_theta_new, torch.tensor(0)) # n by 1
                z_bar = torch.mean(groups_onehot, dim=0) # 1 by g
                loss_groupwise = torch.abs(torch.mean((groups_onehot - z_bar)*g_theta, dim=0))
                loss = torch.sum(loss_groupwise)
                return loss
            
            FPR_total  += closure_FPR(inputs, groups, labels, model)
            FNR_total += closure_FNR(inputs, groups, labels, model)
        return [torch.abs(FPR_total).item(), torch.abs(FNR_total).item()]