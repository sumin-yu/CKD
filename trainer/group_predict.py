from __future__ import print_function

import time
from utils import get_accuracy
import trainer
import torch
import torch.nn as nn


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)

    def train(self, train_loader, val_loader, test_loader, epochs):
        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, self.model)

            val_loss, val_acc = self.evaluate(self.model, val_loader, self.criterion)
            print('[{}/{}] Method: {} '
                    'Val Loss: {:.3f} Val Acc: {:.2f}'.format
                    (epoch + 1, epochs, self.method,
                    val_loss, val_acc))
                    
            eval_start_time = time.time()
            eval_loss, eval_acc = self.evaluate(self.model, test_loader, self.criterion)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method, 
                   eval_loss, eval_acc, (eval_end_time - eval_start_time)))

            if self.scheduler != None and 'Multi' not in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()
                  
        print('Training Finished!')        

    def _train_epoch(self, epoch, train_loader, model):
        model.train()
        
        running_acc = 0.0
        running_loss = 0.0

        batch_start_time = time.time()
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data
            
            labels = groups

            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
            outputs = model(inputs)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if i % self.term == self.term-1: # print every self.term mini-batches
                avg_batch_time = time.time()-batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i+1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time/self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()


    def evaluate(self, model, loader, criterion, device=None):
        model.eval()
        num_groups = loader.dataset.num_groups
        num_classes = loader.dataset.num_classes
        device = self.device if device is None else device

        eval_acc = 0 
        eval_loss = 0
        eval_data_count = 0
        n_subgroups = num_classes * num_groups
        
        if 'Custom' in type(loader).__name__:
            loader = loader.generate()
        with torch.no_grad():
            for j, eval_data in enumerate(loader):
                # Get the inputs
                inputs, _, groups, classes, _ = eval_data
                #
                labels = groups 
                if self.cuda:
                    inputs = inputs.cuda(device)
                    labels = labels.cuda(device)

                outputs = model(inputs)    
                
                loss = criterion(outputs, labels)
                eval_loss += loss.item() * len(labels)
                preds = torch.argmax(outputs, 1)
                acc = (preds == labels).float().squeeze()
                eval_acc += acc.sum()
                eval_data_count += len(labels)

            eval_loss = eval_loss / eval_data_count.sum() 
            eval_acc = eval_acc / eval_data_count.sum()
        model.train()
        return eval_loss, eval_acc