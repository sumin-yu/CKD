from __future__ import print_function
from collections import defaultdict

import copy
import time
from utils import get_accuracy
import trainer
import torch
import numpy as np

from torch.utils.data import DataLoader
from trainer.vanilla_train import Trainer as vanilla_trainer
class Trainer(vanilla_trainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        # self.gamma = args.gamma # learning rate of adv_probs
        self.rho = args.rho
        self.test_criterion = torch.nn.CrossEntropyLoss()
        
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
            self.adv_probs_dict[l] = torch.ones(n_groups).cuda(device=self.device) / n_groups
        
        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, model)            
            _, _, _, _, train_subgroup_acc, train_subgroup_loss = self.evaluate_train(self.model, 
                                                                                    self.normal_loader,
                                                                                    epoch,
                                                                                    train=True,
                                                                                    )
            train_subgroup_loss = 1-train_subgroup_acc

            # q update
            self.adv_probs_dict = self._q_update_ibr_linear_interpolation( 
                                                                        train_subgroup_loss, 
                                                                        n_classes,
                                                                        n_groups, 
                                                                        epoch, 
                                                                        epochs
                                                                        )


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
            inputs, _, groups, labels, _ = data if not self.aug_mode else self.dim_change(data)
            
            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.cuda(device=self.device)
                
            def closure():
                subgroups = groups * n_classes + labels
                outputs = model(inputs)

                loss = self.criterion(outputs, labels)
                if self.aug_mode and self.ce_aug:
                    ctf_loss = loss[self.bs:].mean()
                    loss = loss[:self.bs]
                    subgroups = subgroups[:self.bs]
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

                if self.aug_mode and self.ce_aug: 
                    robust_loss = (robust_loss + ctf_loss)/2
                    
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


    def _q_update_ibr_linear_interpolation(self, train_subgroup_loss, n_classes, n_groups, epoch, epochs):
        train_subgroup_loss = torch.flatten(train_subgroup_loss)
        assert len(train_subgroup_loss) == (n_classes * n_groups)

        idxs = np.array([i * n_classes for i in range(n_groups)]) 
        q_start = copy.deepcopy(self.adv_probs_dict)
        q_ibr = copy.deepcopy(self.adv_probs_dict)
        cur_step_size = 1 - epoch/epochs
        for l in range(n_classes):
            label_group_loss = train_subgroup_loss[idxs+l]
            q_ibr[l] = self._update_mw_margin(label_group_loss)#, self.group_dist[l])
            # self.adv_probs_dict[l] = q_start[l] + cur_step_size*(q_ibr[l] - q_start[l])
            q_ibr[l] = q_start[l] + cur_step_size*(q_ibr[l] - q_start[l])
            print(f'{l} label loss : {train_subgroup_loss[idxs+l]}')
            # print(f'{l} label q_ibr values : {q_ibr[l]}')
            # print(f'{l} label q values : {self.adv_probs_dict[l]}')
            print(f'{l} label q values : {q_ibr[l]}')
        return q_ibr


    def _update_mw_margin(self, losses, p_train=None):

        if losses.min() < 0:
            raise ValueError

        rho = self.rho

        n_groups = len(losses)
        mean = losses.mean()
        denom = (losses - mean).norm(2)
        if denom == 0:
            q = torch.zeros_like(losses) + 1/n_groups
        else:
            q = 1/n_groups + np.sqrt(2 * self.rho / n_groups)* (1/denom) * (losses - mean)
        return q

    def evaluate_train(self, model, loader, epoch=0, train=False):
        if not train:
            model.eval()
        else:
            model.train()
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        n_groups = loader.dataset.num_groups
        n_classes = loader.dataset.num_classes
        n_subgroups = n_groups * n_classes        

        group_count = torch.zeros(n_subgroups).cuda(device=self.device)
        group_loss = torch.zeros(n_subgroups).cuda(device=self.device)        
        group_acc = torch.zeros(n_subgroups).cuda(device=self.device) 

        with torch.no_grad():
            for j, eval_data in enumerate(loader):
                inputs, _, groups, labels, _ = eval_data if not self.aug_mode else self.dim_change(eval_data)
                if self.aug_mode:
                    inputs = inputs[:self.bs]
                    groups = groups[:self.bs]
                    labels = labels[:self.bs]

                if self.cuda:
                    inputs = inputs.cuda(self.device)
                    labels = labels.cuda(self.device)
                    groups = groups.cuda(self.device)

                outputs = model(inputs)

                loss = criterion(outputs, labels)
                preds = torch.argmax(outputs, 1)
                acc = (preds == labels).float().squeeze()

                subgroups = groups * n_classes + labels                
                group_map = (subgroups == torch.arange(n_subgroups).unsqueeze(1).long().cuda()).float()
                group_count += group_map.sum(1)

                group_loss += (group_map @ loss.view(-1))
                group_acc += group_map @ acc

            loss = group_loss.sum() / group_count.sum() 
            acc = group_acc.sum() / group_count.sum() 

            group_loss /= group_count
            group_acc /= group_count

            group_loss = group_loss.reshape((n_groups, n_classes))            
            group_acc = group_acc.reshape((n_groups, n_classes))
            labelwise_acc_gap = torch.max(group_acc, dim=0)[0] - torch.min(group_acc, dim=0)[0]
            deoa = torch.mean(labelwise_acc_gap).item()
            deom = torch.max(labelwise_acc_gap).item()

        model.train()
        return loss, acc, deom, deoa, group_acc, group_loss 
