from __future__ import print_function

import time
from utils import get_accuracy
import torch
from trainer.kd_hinton import Trainer as hinton_Trainer

class Trainer(hinton_Trainer):
    def __init__(self, args, **kwargs):
        if 'aug' not in args.dataset:
            raise ValueError
        
        self.lamb = args.lambf
        self.kd_lamb = args.kd_lambf
        
        super().__init__(args=args, **kwargs)
        self.clip_filtering = args.clip_filtering

    def _train_epoch(self, epoch, train_loader, model, teacher):
        model.train()
        teacher.eval()
        
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
            
            org_filtered_idx = torch.arange(batch_size)
            if self.clip_filtering:
                
                # ctf_filtered_idx = torch.arange(batch_size, batch_size*2)
                ctf_idx_ = (filter_indicator == 1).nonzero(as_tuple=True)[0]
                filtered_idx = torch.cat((torch.arange(batch_size) , (ctf_idx_+torch.ones(ctf_idx_.shape[0])*batch_size))).type(torch.LongTensor)
                inputs = inputs[filtered_idx,:,:,:]
                groups = groups[filtered_idx]
                targets = targets[filtered_idx]

                org_filtered_idx = ctf_idx_.type(torch.LongTensor)
                # ctf_filtered_idx = (ctf_idx_ + torch.ones(ctf_idx_.shape[0])*batch_size).type(torch.LongTensor)

            labels = targets 

            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.long().cuda(self.device)
                t_inputs = inputs.to(self.t_device)
            
            outputs = model(inputs, get_inter=True)
            stu_logits = outputs[-1]
            stu_features = outputs[-2]

            t_outputs = teacher(t_inputs, get_inter=True)
            t_features = t_outputs[-2]

            celoss = self.criterion(stu_logits[:batch_size], labels[:batch_size])

            # logit pairing loss
            ft_logit = stu_logits[org_filtered_idx]
            ctf_logit = stu_logits[batch_size:]
            lp_loss = torch.mean((ft_logit-ctf_logit).pow(2))

            # kd feature pairing loss
            t_target_features = (t_features[:batch_size] + t_features[batch_size:])/2

            ft_features = stu_features[org_filtered_idx]
            ctf_features = stu_features[batch_size:]
            pairing_loss1 = torch.mean((ft_features-t_target_features).pow(2))
            pairing_loss2 = torch.mean((ctf_features-t_target_features).pow(2))

            kd_fp_loss = (pairing_loss1 + pairing_loss2) /2

            loss = celoss + self.lamb * lp_loss + self.kd_lamb * kd_fp_loss

            running_loss += loss.item()
            running_acc += get_accuracy(stu_logits, labels)

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

