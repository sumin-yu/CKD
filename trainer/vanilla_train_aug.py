from __future__ import print_function

import time
from utils import get_accuracy
import trainer.vanilla_train
import torch

class Trainer(trainer.vanilla_train.Trainer):
    def __init__(self, args, **kwargs):
        if 'aug' not in args.dataset:
            raise ValueError
        
        super().__init__(args=args, **kwargs)
        self.clip_filtering = args.clip_filtering


    def _train_epoch(self, epoch, train_loader, model):
        model.train()
        
        running_acc = 0.0
        running_loss = 0.0

        batch_start_time = time.time()
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, filter_indicator = data
            batch_size = inputs.shape[0]
            inputs = inputs.permute((1,0,2,3,4))
            inputs = inputs.contiguous().view(-1, *inputs.shape[2:])
            
            groups = torch.reshape(groups.permute((1,0)), (-1,))
            targets = torch.reshape(targets.permute((1,0)), (-1,)).type(torch.LongTensor)

            if self.clip_filtering :
                org_idx = torch.arange(batch_size)
                ctf_idx_ = (filter_indicator == 1).nonzero(as_tuple=True)[0]
                filtered_idx = torch.cat((org_idx , (ctf_idx_+torch.ones(ctf_idx_.shape[0])*batch_size))).type(torch.LongTensor)
                inputs = inputs[filtered_idx,:,:,:]
                groups = groups[filtered_idx]
                targets = targets[filtered_idx]

            labels = targets

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

