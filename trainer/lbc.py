from __future__ import print_function
from collections import defaultdict

import copy
import time
from utils import get_accuracy
import trainer
import torch
import numpy as np

from torch.utils.data import DataLoader

from collections import defaultdict

import copy
import time
from utils import get_accuracy
import trainer
import torch
import numpy as np

from torch.utils.data import DataLoader

class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.train_criterion = torch.nn.CrossEntropyLoss(reduction='none')
        self.eta = args.eta
        self.iteration = args.iter
        self.n_workers = args.num_workers

    def train(self, train_loader, val_loader, test_loader, epochs):        
        
        global loss_set
        model = self.model
        model.train()
        
        self.num_classes = train_loader.dataset.num_classes
        self.num_groups = train_loader.dataset.num_groups
        
        self.extended_multipliers = torch.zeros((self.num_groups, self.num_classes))
        self.weight_matrix = self.get_weight_matrix(self.extended_multipliers)
        
        print('eta_learning_rate : ', self.eta)
        n_iters = self.iteration
        print('n_iters : ', n_iters)
        violations = 0
        for iter_ in range(n_iters):

            for epoch in range(epochs):
                self._train_epoch(epoch, train_loader, model)            

                eval_start_time = time.time()
                eval_loss, eval_acc, eval_deom,  = self.evaluate(self.model, 
                                                                    test_loader, 
                                                                    self.criterion,
                                                                )
                eval_end_time = time.time()
                print('[{}/{}] Method: {} '
                    'Test Loss: {:.3f} Test Acc: {:.2f} Test DEOM {:.2f} [{:.2f} s]'.format
                    (epoch + 1, epochs, self.method,
                    eval_loss, eval_acc, eval_deom, (eval_end_time - eval_start_time)))

                if self.scheduler != None and 'Reduce' in type(self.scheduler).__name__:
                    self.scheduler.step(eval_loss)
                else:
                    self.scheduler.step()

            pred_set, y_set, s_set = self.get_statistics(train_loader.dataset, bs=self.bs,
                                                                n_workers=self.n_workers, model=model)

            # calculate violation
            acc, violations = self.get_error_and_violations_DCA(pred_set, y_set, s_set, self.num_groups, self.num_classes)

            self.extended_multipliers -= self.eta * violations 
            self.weight_matrix = self.get_weight_matrix(self.extended_multipliers) 
        print('Training Finished!')        

    def _train_epoch(self, epoch, train_loader, model):
        model.train()
        
        running_acc = 0.0
        running_loss = 0.0
        batch_start_time = time.time()
        num_classes = train_loader.dataset.num_classes
        num_groups = train_loader.dataset.num_groups
        n_subgroups = num_classes * num_groups
        
        
        idxs = np.array([i * num_classes for i in range(num_groups)])            

        for i, data in enumerate(train_loader):
            # Get the inputs
            inputs, _, groups, targets, idx = data
            labels = targets
            
            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.cuda(device=self.device)
            
            groups = groups.long()
            labels = labels.long()

            weights = self.weight_matrix[groups, labels].cuda()

            
            outputs = model(inputs)

            loss = torch.mean(weights * self.train_criterion(outputs, labels))
            
            self.optimizer.zero_grad()
            loss.backward()                
            self.optimizer.step()

            running_loss += loss.item()
            running_acc += get_accuracy(outputs, labels)
            if i % self.term == self.term-1: # print every self.term mini-batches
                avg_batch_time = time.time()-batch_start_time
                print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
                      '[{:.2f} s/batch]'.format
                      (epoch + 1, self.epochs, i+1, self.method, running_loss / self.term, running_acc / self.term,
                       avg_batch_time/self.term))

                running_loss = 0.0
                running_acc = 0.0
                batch_start_time = time.time()
        
    def get_statistics(self, dataset, bs=128, n_workers=2, model=None):

        dataloader = DataLoader(dataset, batch_size=bs, shuffle=False,
                                num_workers=n_workers, pin_memory=True, drop_last=False)

        if model != None:
            model.eval()

        pred_set = []
        y_set = []
        s_set = []
        total = 0
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                inputs, _, sen_attrs, targets, _ = data
                y_set.append(targets) # sen_attrs = -1 means no supervision for sensitive group
                s_set.append(sen_attrs)

                if self.cuda:
                    inputs = inputs.cuda()
                    targets = targets.cuda()

                if model != None:
                    if self.data == 'jigsaw':
                        input_ids = inputs[:, :, 0]
                        input_masks = inputs[:, :, 1]
                        segment_ids = inputs[:, :, 2]
                        outputs = model(
                            input_ids=input_ids,
                            attention_mask=input_masks,
                            token_type_ids=segment_ids,
                            labels=targets,
                        )[1] 
                    else:
                        outputs = model(inputs)
                    pred_set.append(torch.argmax(outputs, dim=1))
                total+= inputs.shape[0]

        y_set = torch.cat(y_set)
        s_set = torch.cat(s_set)
        pred_set = torch.cat(pred_set) if len(pred_set) != 0 else torch.zeros(0)
        return pred_set.long(), y_set.long().cuda(), s_set.long().cuda()
    

    # Vectorized version for DCA & multi-class
    def get_error_and_violations_DCA(self, y_pred, label, sen_attrs, num_groups, num_classes):
        acc = torch.mean((y_pred == label).float())
        violations = torch.zeros((num_groups, num_classes)) 
        for g in range(num_groups):
            for c in range(num_classes):
                class_idxs = torch.where(label==c)[0]
                pred_class_idxs = torch.where(torch.logical_and(y_pred == c, label == c))[0]
                pivot = len(pred_class_idxs)/len(class_idxs)
                group_class_idxs=torch.where(torch.logical_and(sen_attrs == g, label == c))[0]
                group_pred_class_idxs = torch.where(torch.logical_and(torch.logical_and(sen_attrs == g, y_pred == c), label == c))[0]
                violations[g, c] = len(group_pred_class_idxs)/len(group_class_idxs) - pivot
        print('violations',violations)
        return acc, violations
    
    def get_weight_matrix(self, extended_multipliers):  
        w_matrix = torch.sigmoid(extended_multipliers) # g by c
        return w_matrix
    
