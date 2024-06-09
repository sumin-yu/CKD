from __future__ import print_function
import time
from utils import get_accuracy
from trainer.loss_utils import compute_hinton_loss
import trainer
import torch

from trainer.kd_hinton import Trainer as hinton_trainer
class Trainer(hinton_trainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.lamb = args.cp_lambf

    def _train_epoch(self, epoch, train_loader, model, teacher, distiller=None, num_aug=1, num_groups=2):

        model.train()
        teacher.eval()

        running_acc = 0.0
        running_loss = 0.0

        batch_start_time = time.time()
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, labels, filter_indicator = self.dim_change(data)
            if self.cuda:
                inputs = inputs.cuda(self.device)
                labels = labels.cuda(self.device)
            
            t_inputs = inputs.to(self.t_device)

            outputs = model(inputs)
            t_outputs = teacher(t_inputs)
            kd_loss = compute_hinton_loss(outputs, t_outputs, kd_temp=self.kd_temp, device=self.device)
            ft_logit = outputs[:self.bs]
            ctf_logit = outputs[self.bs:]
            lp_loss = torch.mean((ft_logit-ctf_logit).pow(2))

            loss = self.criterion(outputs, labels) + self.lambh * kd_loss +  self.lamb * lp_loss 

            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if i % self.term == self.term-1: # print every self.term mini-batches
                avg_batch_time = time.time() - batch_start_time

                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i+1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time/self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()

        # if not self.no_annealing:
        #     self.lambh = self.lambh - 3/(self.epochs-1)
