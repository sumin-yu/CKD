from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.linear_model import LogisticRegression

from utils import get_accuracy
import networks
import trainer

class Trainer(trainer.GenericTrainer):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        # rho, eps, auditor_nsteps, auditor_lr, f_extractor_path
        self.epochs_dist = args.epochs_dist
        self.rho = args.sensei_rho # 5.0
        self.eps = args.sensei_eps # 0.1
        self.lamb = None
        self.minlambda = torch.tensor(1e-5, device=self.device)
        self.auditor_nsteps = args.auditor_nsteps # 100
        self.auditor_lr = args.auditor_lr # 1e-3
        self.distance_x = LogisticRegSensitiveSubspace(self.model, self.device)
        self.distance_y = SquaredEuclideanDistance(self.device)
        self.auditor = SenSeIAuditor(self.distance_x, self.distance_y, self.auditor_nsteps, self.auditor_lr)

    def train(self, train_loader, val_loader, test_lodaer, epochs):
        self.model.train()

        # # distance_x learning
        # for epoch in range(self.epochs_dist):
        #     self._train_epoch_dist(epoch, train_loader, self.model)

        self.distance_x.fit(train_loader)
        # freeze model body
        for name, param in self.model.named_parameters():
            if 'fc' not in name:
                param.requires_grad = False

        # distance_y learning
        self.distance_y.fit(num_dims=train_loader.dataset.num_classes)

        for epoch in range(epochs):
            self._train_epoch_fair(epoch, train_loader, self.model, auditor=self.auditor)

            val_loss, val_acc, val_deopp = self.evaluate(self.model, val_loader, self.criterion)
            print('[{}/{}] Method: {} '
                    'Val Loss: {:.3f} Val Acc: {:.2f} Val DEopp {:.2f}'.format
                    (epoch + 1, epochs, self.method,
                    val_loss, val_acc, val_deopp))
            
            eval_start_time = time.time()
            eval_loss, eval_acc, eval_deopp = self.evaluate(self.model, test_lodaer, self.criterion)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} Test DEopp {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method, 
                   eval_loss, eval_acc, eval_deopp, (eval_end_time - eval_start_time)))

            if self.scheduler != None and 'Multi' not in type(self.scheduler).__name__:
                self.scheduler.step(eval_loss)
            else:
                self.scheduler.step()

        print("Training Finished!")

    def _train_epoch_fair(self, epoch, train_loader, model, auditor=None):

        running_acc = 0.0
        running_loss = 0.0

        batch_start_time = time.time()
        for i, data in enumerate(train_loader):
            inputs, _, groups, targets, _ = data
            labels = targets

            if self.cuda:
                inputs = inputs.cuda(device=self.device)
                labels = labels.cuda(device=self.device)
                groups = groups.cuda(device=self.device)
                self.eps = torch.tensor(self.eps, device=self.device)
                if self.lamb is None:
                    self.lamb = torch.tensor(1.0, device=self.device)
            
            outputs = model(inputs, get_inter=True)
            
            input_worst, feature_worst = auditor.generate_worst_case_examples(model, inputs, lambda_param=self.lamb)

            dist_x = self.distance_x(outputs[-2], feature_worst)
            mean_dist_x = dist_x.mean()
            lr_factor = torch.maximum(mean_dist_x, self.eps) / torch.minimum(mean_dist_x, self.eps)

            self.lamb = torch.max(torch.stack([self.minlambda, self.lamb + lr_factor * (mean_dist_x - self.eps)]))

            output_worst = model(input_worst)
            loss = self.criterion(outputs[-1], labels)

            dist_y = self.distance_y(outputs[-1], output_worst)
            # print('loss: ', torch.mean(loss), ' dist_y: ', torch.mean(dist_y))
            loss = torch.mean(loss + self.rho * dist_y)
            running_loss += loss.item()
            running_acc += get_accuracy(outputs[-1], labels)
            
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

    # def _train_epoch_dist(self, epoch, train_loader, model):
        
    #     running_acc = 0.0
    #     running_loss = 0.0

    #     batch_start_time = time.time()
    #     for i, data in enumerate(train_loader):
    #         # Get the inputs
    #         inputs, _, groups, targets, _ = data
            
    #         labels = targets

    #         if self.cuda:
    #             inputs = inputs.cuda(device=self.device)
    #             labels = labels.cuda(device=self.device)
    #         outputs = model(inputs)
    #         criterion = nn.CrossEntropyLoss()
    #         loss = criterion(outputs, labels)

    #         running_loss += loss.item()
    #         running_acc += get_accuracy(outputs, labels)

    #         self.optimizer.zero_grad()
    #         loss.backward()
    #         self.optimizer.step()
            
    #         if i % self.term == self.term-1: # print every self.term mini-batches
    #             avg_batch_time = time.time()-batch_start_time
    #             print('[{}/{}, {:5d}] Method: {} Train Loss: {:.3f} Train Acc: {:.2f} '
    #                   '[{:.2f} s/batch]'.format
    #                   (epoch + 1, self.epochs_dist, i+1, 'before metric learning', running_loss / self.term, running_acc / self.term,
    #                    avg_batch_time/self.term))

    #             running_loss = 0.0
    #             running_acc = 0.0
    #             batch_start_time = time.time()

class SenSeIAuditor(nn.Module):
    def __init__(self, distance_x, distance_y, num_steps, lr):
        super().__init__()
        self.distance_x = distance_x
        self.distance_y = distance_y
        self.num_steps = num_steps
        self.lr = lr
        self.max_noise = 0.1
        self.min_noise = -0.1
    
    def generate_worst_case_examples(self, model, x, lambda_param):
        # print('generate worst case examples...')

        self.freeze_network(model)
        lambda_param = lambda_param.detach()

        delta = nn.Parameter(torch.rand_like(x) * (self.max_noise - self.min_noise) + self.min_noise)

        optimizer = torch.optim.Adam([delta], lr=self.lr)
        for i in range(self.num_steps):
            optimizer.zero_grad()
            x_worst = x + delta
            # input_dist = self.distance_x(x, x_worst)
            out_x = model(x, get_inter=True)
            out_x_worst = model(x_worst, get_inter=True)
            input_dist = self.distance_x(out_x[-2], out_x_worst[-2])
            out_dist = self.distance_y(out_x[-1], out_x_worst[-1])

            loss = -(out_dist - lambda_param * input_dist)
            loss.sum().backward()
            optimizer.step()
        
        self.unfreeze_network(model)

        return  (x + delta).detach(), out_x_worst[-2].detach()
    
    def freeze_network(self, network):
        for p in network.parameters():
            p.requires_grad = False
    
    def unfreeze_network(self, network):
        # for p in network.parameters():
        #     p.requires_grad = True
        for name, param in network.named_parameters():
            if 'fc' in name:
                param.requires_grad = True

class SquaredEuclideanDistance(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.num_dims = 2
        self.device = device

    def fit(self, num_dims):
        self.num_dims = num_dims
        self.sigma = torch.eye(num_dims).detach()
        self.sigma = self.sigma.cuda(device=self.device)

        print('')

    def forward(self, x1, x2):
        dist = self.compute_dist(x1, x2, self.sigma)
        return dist
    
    def compute_dist(self, x1, x2, sigma):
        if len(x1.shape) == 1:
            x1 = x1.unsqueeze(0)
        if len(x2.shape) == 1:
            x2 = x2.unsqueeze(0)

        X_diff = x1 - x2
        dist = torch.sum((X_diff @ sigma) * X_diff, dim=-1, keepdim=True)
        return dist

class LogisticRegSensitiveSubspace(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.basis_vectors = None
        self._logreg_models = None
        self.device = device
        self.model = model
        # # load pre-trained fixed scratch model predicting gender attribute
        # self.f_extractor = networks.ModelFactory.get_model(target_model='resnet', num_classes=2, img_size=256)
        # self.f_extractor.load_state_dict(torch.load(model_path))
        # self.f_extractor.cuda('cuda:{}'.format(device))
        # self.f_extractor.eval()
    
    def fit(self, data_loader): 
        # 1. get feature vectors from self.f_extractor
        data_feature = []
        data_SensitiveAttrs = []
        with torch.no_grad():
            for i, data in enumerate(data_loader):
                inputs, _, groups, _, _ = data

                inputs = inputs.cuda(device=self.device)
                outputs = self.model(inputs, get_inter=True)[-2]
                outputs = outputs.view(outputs.shape[0], -1)
                data_feature += outputs
                data_SensitiveAttrs += groups
                if (i==100): break
        data_feature = torch.stack(data_feature)
        data_SensitiveAttrs = torch.stack(data_SensitiveAttrs).reshape(-1,1)
        # 2. then, using the inter-feature vectors, train logistic regression model
        basis_vectors_ = self.compute_basis_vectors_data(X_train=data_feature, y_train=data_SensitiveAttrs)

        self.sigma = self.compute_projection_complement(basis_vectors_)
        self.sigma = self.sigma.cuda(device=self.device)
        self.basis_vectors = basis_vectors_

        print('distance_X metric learning finished.')

    def compute_basis_vectors_data(self, X_train, y_train): 
        dtype = X_train.dtype

        X_train = self.convert_tensor_to_numpy(X_train)
        y_train = self.convert_tensor_to_numpy(y_train)

        basis_vectors_ = []
        outdim = y_train.shape[-1]

        self._logreg_models = [
            LogisticRegression(solver="liblinear", penalty="l1")
            .fit(X_train, y_train[:, idx])
            for idx in range(outdim)
        ]

        basis_vectors_ = np.array(
            [
                self._logreg_models[idx].coef_.squeeze()
                for idx in range(outdim)
            ]
        )

        basis_vectors_ = torch.tensor(basis_vectors_, dtype=dtype).T
        basis_vectors_ = basis_vectors_.detach()
        return basis_vectors_

    def convert_tensor_to_numpy(self, tensor):
        if torch.is_tensor(tensor):
            array_np = tensor.detach().cpu().numpy()
            return array_np

    def compute_projection_complement(self, basis_vectors):
        projection = torch.linalg.inv(torch.matmul(basis_vectors.T, basis_vectors))
        projection = torch.matmul(basis_vectors, projection)
        # Shape: (n_features, n_features)
        projection = torch.matmul(projection, basis_vectors.T)

        # Complement the projection as: (I - Proj)
        projection_complement_ = torch.eye(projection.shape[0]) - projection
        projection_complement_ = projection_complement_.detach()

        return projection_complement_

    def forward(self, x1_feature, x2_feature):
        # with torch.no_grad():
        #     x1_feature = self.model(x1, get_inter=True)
        #     x2_feature = self.model(x2, get_inter=True)
        dist = self.compute_dist(x1_feature, x2_feature, self.sigma)
        return dist

    def compute_dist(self, x1, x2, sigma):
        if len(x1.shape) == 1:
            x1 = x1.unsqueeze(0)
        if len(x2.shape) == 1:
            x2 = x2.unsqueeze(0)
        
        x1 = x1.view(x1.shape[0], -1)
        x2 = x2.view(x2.shape[0], -1)
        X_diff = x1 - x2
        # X_diff = X_diff.cuda()
        # sigma = sigma.cuda()
        dist = torch.sum((X_diff @ sigma) * X_diff, dim=-1, keepdim=True)
        return dist