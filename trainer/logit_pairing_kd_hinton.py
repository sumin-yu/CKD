from __future__ import print_function
import time
from utils import get_accuracy
from trainer.loss_utils import compute_hinton_loss
import trainer
import torch


class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.lambh = args.lambh
        self.kd_temp = args.kd_temp
        self.seed = args.seed

        self.lamb = args.lambf
        # self.no_annealing = args.no_annealing

    def train(self, train_loader, val_loader, test_loader, epochs):

        self.model.train()
        self.teacher.eval()

        for epoch in range(self.epochs):
            self._train_epoch(epoch, train_loader, self.model, self.teacher)
            
            val_loss, val_acc, val_deopp = self.evaluate(self.model, val_loader, self.criterion)
            print('[{}/{}] Method: {} '
                    'Val Loss: {:.3f} Val Acc: {:.2f} Val DEopp {:.2f}'.format
                    (epoch + 1, epochs, self.method,
                    val_loss, val_acc, val_deopp))
            
            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deopp = self.evaluate(self.model, test_loader, self.criterion)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEopp {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method, 
                   eval_loss, eval_acc, eval_deopp, (eval_end_time - eval_start_time)))

            if self.scheduler != None:
                self.scheduler.step(eval_loss)

        print('Training Finished!')

    def _train_epoch(self, epoch, train_loader, model, teacher, distiller=None, num_aug=1, num_groups=2):

        model.train()
        teacher.eval()

        running_acc = 0.0
        running_loss = 0.0

        batch_start_time = time.time()
        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs_, int_inputs_, _, groups_, targets, _ = data
            inputs = inputs.permute((1,0,2,3,4))
            batch_size = inputs.shape[0]
            inputs = inputs.contiguous().view(-1, *inputs.shape[2:])
            
            groups = torch.reshape(groups.permute((1,0)), (-1,))
            targets = torch.reshape(targets.permute((1,0)), (-1,)).type(torch.LongTensor)
            
            labels = targets

            org_filtered_idx = torch.arange(batch_size)
            if self.cuda:
                inputs = inputs.cuda(self.device)
                labels = labels.cuda(self.device)
            
            t_inputs = inputs.to(self.t_device)

            outputs = model(inputs)
            t_outputs = teacher(t_inputs[org_filtered_idx])
            kd_loss = compute_hinton_loss(outputs[org_filtered_idx], t_outputs, kd_temp=self.kd_temp, device=self.device)

            ft_logit = stu_logits[org_filtered_idx]
            ctf_logit = stu_logits[batch_size:]
            lp_loss = torch.mean((ft_logit-ctf_logit).pow(2))

            t_target_logit = (tea_logits[:batch_size] + tea_logits[batch_size:])/2

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
