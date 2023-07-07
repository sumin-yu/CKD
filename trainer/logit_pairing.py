from __future__ import print_function

import time
from utils import get_accuracy
import trainer.vanilla_train
import torch

class Trainer(trainer.vanilla_train.Trainer):
    def __init__(self, args, **kwargs):
        if 'aug' not in args.dataset:
            raise ValueError
        
        self.lamb = args.lambf
        
        super().__init__(args=args, **kwargs)

    def _train_epoch(self, epoch, train_loader, model):
        model.train()
        
        running_acc = 0.0
        running_loss = 0.0

        batch_start_time = time.time()
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data
            inputs = inputs.permute((1,0,2,3,4))
            inputs = inputs.contiguous().view(-1, *inputs.shape[2:])
            targets = torch.stack((targets,targets),dim=0).view(-1)
            
            labels = targets

            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
            
            outputs = model(inputs)
            celoss = self.criterion(outputs[:int(outputs.shape[0]/2)], labels[:int(labels.shape[0]/2)])

            ft_batch_size = int(inputs.shape[0] / 2)
            ft_logit = outputs[:ft_batch_size]
            ctf_logit = outputs[ft_batch_size:]
            pairing_loss = (ft_logit-ctf_logit).norm(2).pow(2)

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

