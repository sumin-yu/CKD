from __future__ import print_function

import time
from utils import get_accuracy
import trainer.vanilla_train
import torch
import torch.nn as nn

class Trainer(trainer.vanilla_train.Trainer):
    def __init__(self, args, **kwargs):
        if 'aug' not in args.dataset:
            raise ValueError
        
        self.lambf = args.lambf
        self.sigma = args.sigma
        self.kernel = args.kernel

        super().__init__(args=args, **kwargs)
        self.clip_filtering = args.clip_filtering

    def train(self, train_loader, val_loader, test_loader, epochs):
        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups

        mmd_loss = MMDLoss(w_m=self.lambf, sigma=self.sigma,
                            num_classes=num_classes, num_groups=num_groups, kernel=self.kernel)

        for epoch in range(epochs):
            self._train_epoch(epoch, train_loader, self.model, mmd_loss)

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

            if self.scheduler != None and 'Multi' not in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()
                  
        print('Training Finished!')        


    def _train_epoch(self, epoch, train_loader, model, mmd):
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

            org_filtered_idx = torch.arange(batch_size)
            # ctf_filtered_idx = torch.arange(batch_size, batch_size*2)
            if self.clip_filtering:
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
                groups = groups.cuda(device=self.device)
            
            outputs = model(inputs, get_inter=True)
            logits = outputs[-1]
            
            features = outputs[-2]
            celoss = self.criterion(logits[:batch_size], labels[:batch_size])

            f_s = outputs[-2]
            f_t = f_s.clone().detach()
            mmd_loss = mmd.forward(f_s, f_t, groups=groups, labels=labels)

            loss = celoss + mmd_loss

            running_loss += loss.item()
            running_acc += get_accuracy(logits, labels)

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



class MMDLoss(nn.Module):
    def __init__(self, w_m, sigma, num_groups, num_classes, kernel):
        super(MMDLoss, self).__init__()
        self.w_m = w_m
        self.sigma = sigma
        self.num_groups = num_groups
        self.num_classes = num_classes
        self.kernel = kernel

    def forward(self, f_s, f_t, groups, labels, jointfeature=False):
        if self.kernel == 'poly':
            student = F.normalize(f_s.view(f_s.shape[0], -1), dim=1)
            teacher = F.normalize(f_t.view(f_t.shape[0], -1), dim=1).detach()
        else:
            student = f_s.view(f_s.shape[0], -1)
            teacher = f_t.view(f_t.shape[0], -1).detach()

        mmd_loss = 0

        if jointfeature:
            K_TS, sigma_avg = self.pdist(teacher, student,
                              sigma_base=self.sigma, kernel=self.kernel)
            K_TT, _ = self.pdist(teacher, teacher, sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)
            K_SS, _ = self.pdist(student, student,
                              sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

            mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()

        else:
            with torch.no_grad():
                _, sigma_avg = self.pdist(teacher, student, sigma_base=self.sigma, kernel=self.kernel)

            for c in range(self.num_classes):
                if len(teacher[labels==c]) == 0:
                    continue
                for g in range(self.num_groups):
                    if len(student[(labels==c) * (groups == g)]) == 0:
                        continue
                    K_TS, _ = self.pdist(teacher[labels == c], student[(labels == c) * (groups == g)],
                                                 sigma_base=self.sigma, sigma_avg=sigma_avg,  kernel=self.kernel)
                    K_SS, _ = self.pdist(student[(labels == c) * (groups == g)], student[(labels == c) * (groups == g)],
                                         sigma_base=self.sigma, sigma_avg=sigma_avg, kernel=self.kernel)

                    K_TT, _ = self.pdist(teacher[labels == c], teacher[labels == c], sigma_base=self.sigma,
                                         sigma_avg=sigma_avg, kernel=self.kernel)

                    mmd_loss += K_TT.mean() + K_SS.mean() - 2 * K_TS.mean()

        loss = (1/2) * self.w_m * mmd_loss

        return loss

    @staticmethod
    def pdist(e1, e2, eps=1e-12, kernel='rbf', sigma_base=1.0, sigma_avg=None):
        if len(e1) == 0 or len(e2) == 0:
            res = torch.zeros(1)
        else:
            if kernel == 'rbf':
                e1_square = e1.pow(2).sum(dim=1)
                e2_square = e2.pow(2).sum(dim=1)
                prod = e1 @ e2.t()
                res = (e1_square.unsqueeze(1) + e2_square.unsqueeze(0) - 2 * prod).clamp(min=eps)
                res = res.clone()
                sigma_avg = res.mean().detach() if sigma_avg is None else sigma_avg
                res = torch.exp(-res / (2*(sigma_base)*sigma_avg))
            elif kernel == 'poly':
                res = torch.matmul(e1, e2.t()).pow(2)

        return res, sigma_avg
