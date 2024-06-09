from __future__ import print_function

import time
from utils import get_accuracy
import trainer.vanilla_train
import torch

class Trainer(trainer.vanilla_train.Trainer):
    def __init__(self, args, **kwargs):
        if 'aug' not in args.dataset:
            raise ValueError
        
        self.lamb = args.cp_lambf
        
        super().__init__(args=args, **kwargs)
        self.clip_filtering = args.clip_filtering

    def _train_epoch(self, epoch, train_loader, model):
        model.train()
        
        running_acc = 0.0
        running_loss = 0.0

        batch_start_time = time.time()
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, filter_indicator = self.dim_change(data)

            labels = targets 

            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
            
            outputs = model(inputs)
            celoss = self.criterion(outputs, labels)

            ft_logit = outputs[:self.bs]
            ctf_logit = outputs[self.bs:]
            pairing_loss = torch.mean((ft_logit-ctf_logit).pow(2))

            loss = celoss + self.lamb * pairing_loss

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

