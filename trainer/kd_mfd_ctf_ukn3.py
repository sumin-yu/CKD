from __future__ import print_function

import torch
import torch.nn.functional as F
import torch.nn as nn
import time
from utils import get_accuracy
from trainer.kd_mfd_ctf import Trainer as mfd_ctf_Trainer
from trainer.loss_utils import compute_hinton_loss

class Trainer(mfd_ctf_Trainer):
    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        if 'wo_org' not in args.dataset:
            raise ValueError
        
    def _train_epoch(self, epoch, train_loader, model, teacher, distiller=None):
        model.train()
        teacher.eval()

        running_acc = 0.0
        running_loss = 0.0
        batch_start_time = time.time()

        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, _ = data
            inputs = inputs.permute((1,0,2,3,4))
            inputs = inputs.contiguous().view(-1, *inputs.shape[2:])
            
            groups = torch.reshape(groups.permute((1,0)), (-1,))
            targets = torch.reshape(targets.permute((1,0)), (-1,)).type(torch.LongTensor)

            labels = targets 

            if self.cuda:
                inputs = inputs.cuda(self.device)
                labels = labels.cuda(self.device)
                groups = groups.long().cuda(self.device)
            t_inputs = inputs.to(self.t_device)

            outputs = model(inputs, get_inter=True)
            stu_logits = outputs[-1]

            t_outputs = teacher(t_inputs, get_inter=True)
            tea_logits = t_outputs[-1]

            kd_loss = compute_hinton_loss(stu_logits, t_outputs=tea_logits,
                                          kd_temp=self.kd_temp, device=self.device) if self.lambh != 0 else 0

            # loss = self.criterion(stu_logits, labels)
            loss = self.criterion(stu_logits, labels)
            loss = loss + self.lambh * kd_loss

            f_s = outputs[-2]
            f_t = t_outputs[-2]
            mmd_loss = distiller.forward(f_s, f_t, groups=groups, labels=labels)

            loss = loss + mmd_loss
            running_loss += loss.item()
            running_acc += get_accuracy(stu_logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.term == self.term - 1:  # print every self.term mini-batches
                avg_batch_time = time.time() - batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i + 1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time / self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()
