from __future__ import print_function

import time
from utils import get_accuracy
import torch
from trainer.kd_hinton import Trainer as hinton_Trainer

class Trainer(hinton_Trainer):
    def __init__(self, args, **kwargs):
        if 'aug' not in args.dataset:
            raise ValueError
        
        self.cp_lamb = args.cp_lambf
        self.kd_lamb = args.kd_lambf
        self.rep = args.rep
        
        super().__init__(args=args, **kwargs)

    def _train_epoch(self, epoch, train_loader, model, teacher):
        model.train()
        teacher.eval()
        
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
                groups = groups.long().cuda(self.device)
                t_inputs = inputs.to(self.t_device)
            
            outputs = model(inputs, get_inter=True)
            stu_logits = outputs[-1]
            stu_rep_vectors = outputs[-2] if self.rep == 'feature' else outputs[-1]

            t_outputs = teacher(t_inputs, get_inter=True)
            t_rep_vectors = t_outputs[-2] if self.rep == 'feature' else t_outputs[-1]

            celoss = self.criterion(stu_logits, labels, t_outputs[-1])

            if self.cp_lamb != 0:
                # logit pairing loss
                ft_logit = stu_logits[:self.bs]
                ctf_logit = stu_logits[self.bs:]
                lp_loss = torch.mean((ft_logit-ctf_logit).pow(2))

            # kd feature pairing loss
            t_target_vectors = (t_rep_vectors[:self.bs] + t_rep_vectors[self.bs:])/2

            ft_rep_vectors = stu_rep_vectors[:self.bs]
            ctf_rep_vectors = stu_rep_vectors[self.bs:]
            pairing_loss1 = torch.mean((ft_rep_vectors-t_target_vectors).pow(2))
            pairing_loss2 = torch.mean((ctf_rep_vectors-t_target_vectors).pow(2))
            kd_fp_loss = (pairing_loss1 + pairing_loss2) /2

            loss = celoss + self.cp_lamb * lp_loss + self.kd_lamb * kd_fp_loss if self.cp_lamb != 0 else celoss + self.kd_lamb * kd_fp_loss

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

