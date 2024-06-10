from __future__ import print_function
import os
import torch
import torch.nn as nn
import numpy as np
import time
from sklearn.linear_model import LogisticRegression

from utils import get_accuracy
import networks
import trainer
import pickle
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import SigmoidTransform
from torch.distributions.uniform import Uniform

from torch.utils.data import DataLoader

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
        self.dist_path = args.sensei_dist_path
        self.distance_x = LogisticRegSensitiveSubspace(self.model, self.device, self.dist_path)
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

            val_loss, val_acc = self.evaluate(self.model, val_loader, self.criterion)
            print('[{}/{}] Method: {} '
                    'Val Loss: {:.3f} Val Acc: {:.2f}}'.format
                    (epoch + 1, epochs, self.method,
                    val_loss, val_acc))
            
            eval_start_time = time.time()
            eval_loss, eval_acc = self.evaluate(self.model, test_lodaer, self.criterion)
            eval_end_time = time.time()
            print('[{}/{}] Method: {} '
                  'Test Loss: {:.3f} Test Acc: {:.2f} [{:.2f} s]'.format
                  (epoch + 1, epochs, self.method, 
                   eval_loss, eval_acc, (eval_end_time - eval_start_time)))

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
    def __init__(self, model, device, dist_path):
        super().__init__()
        self.basis_vectors = None
        self._logreg_models = None
        self.device = device
        self.model = model
        self.dist_path = dist_path
        self.iters = 8
        self.dist_batch = 4
        self.load_num = 30
    
    def fit(self, data_loader): 
        file_name = self.dist_path
        if os.path.isfile(file_name):
            with open(file_name,'rb') as f:
                self.sigma = pickle.load(f)
                print('sigma load complete!')
        else:
            print('sigma learning start!')
            X1 = []
            X2 = []
            Y = []
            data_loader = DataLoader(data_loader.dataset, batch_size=2, shuffle=True, drop_last=True)
            with torch.no_grad():
                for i, data in enumerate(data_loader):
                    inputs, _, groups, _, _ = data

                    inputs = inputs.cuda(device=self.device)
                    groups = groups.cuda(device=self.device)
                    outputs = self.model(inputs, get_inter=True)[-2]
                    outputs = outputs.view(outputs.shape[0], -1)
                    X1 += outputs[0].unsqueeze(0)
                    X2 += outputs[1].unsqueeze(0)
                    Y += [1] if groups[0] == groups[1] else [0]
                    # if i % 10 == 0:
                        # print('data loading..., ',i)
                    if (i==self.load_num-1): break
            X1 = torch.stack(X1)
            X2 = torch.stack(X2)
            # Y = torch.stack(Y)
            Y = torch.tensor(Y)

            assert (
                X1.shape[0] == X2.shape[0] == Y.shape[0]
            ), "Number of elements in X1, X2, and Y do not match"
            X = X1- X2
            # X = self.convert_tensor_to_numpy(X1 - X2)
            # Y = self.convert_tensor_to_numpy(Y)
            self.sigma = self.compute_sigma(X, Y, self.iters, self.dist_batch)
            with open(file_name, 'wb') as f:
                pickle.dump(self.sigma, f)
            print('sigma load complete!')

    def __grad_likelihood__(self, X, Y, sigma):
        """Computes the gradient of the likelihood function using sigmoidal link"""

        # diag = np.einsum("ij,ij->i", np.matmul(X, sigma), X)
        # diag = np.maximum(diag, 1e-10)
        # prVec = logistic.cdf(diag)
        # sclVec = 2.0 / (np.exp(diag) - 1)
        # vec = (Y * prVec) - ((1 - Y) * prVec * sclVec)
        # grad = np.matmul(X.T * vec, X) / X.shape[0]
        # return grad 

        diag = torch.einsum("ij,ij->i", torch.matmul(X, sigma), X)
        diag = torch.maximum(diag, torch.tensor(1e-9))

        base_distribution = Uniform(0, 1)
        transforms = [SigmoidTransform().inv]
        logistic = TransformedDistribution(base_distribution, transforms)
        prVec = logistic.cdf(diag)
        sclVec = 2.0 / (torch.exp(diag) - 1)
        sclVec = torch.nan_to_num(sclVec)
        vec = (Y * prVec) - ((1 - Y) * prVec * sclVec)
        grad = torch.matmul(X.T * vec, X) / X.shape[0]
        return grad.clone().detach()

    def __projPSD__(self, sigma):
        """Computes the projection onto the PSD cone"""

        # try:
        #     L = np.linalg.cholesky(sigma)
        #     sigma_hat = np.dot(L, L.T)
        # except np.linalg.LinAlgError:
        #     d, V = np.linalg.eigh(sigma)
        #     sigma_hat = np.dot(
        #         V[:, d >= 1e-8], d[d >= 1e-8].reshape(-1, 1) * V[:, d >= 1e-8].T
        #     )
        # return sigma_hat
        try:
            L = torch.linalg.cholesky(sigma)
            sigma_hat = torch.matmul(L, L.T)
        except torch.linalg.LinAlgError:
            d, V = torch.linalg.eigh(sigma)
            sigma_hat = torch.matmul(
                V[:, d >= 1e-8], d[d >= 1e-8].reshape(-1, 1) * V[:, d >= 1e-8].T
            )
        return sigma_hat.clone().detach()

    def compute_sigma(self, X, Y, iters, batchsize):
        N = X.shape[0]
        P = X.shape[1]

        sigma_t = torch.normal(0.0, 1.0, size=(P**2,)).reshape(P, P)
        sigma_t = torch.matmul(sigma_t, sigma_t.T)
        sigma_t = sigma_t / torch.linalg.norm(sigma_t)
        sigma_t = sigma_t.cuda(device=self.device)

        curriter = 0

        while curriter < iters:
            # print('iter num: ',curriter)
            batch_idxs = torch.randperm(N)[:batchsize]
            # batch_idxs = torch.random.choice(N, size=batchsize, replace=False)
            X_batch = X[batch_idxs].cuda(device=self.device)
            Y_batch = Y[batch_idxs].cuda(device=self.device)

            grad_t = self.__grad_likelihood__(X_batch, Y_batch, sigma_t)
            # print('grad: \n',grad_t)
            t = 1.0 / (1 + curriter // 1000)
            # t = 1.0 / (100 * (1+curriter))
            sigma_t = self.__projPSD__(sigma_t - t * grad_t)

            curriter += 1
            # print('sigma: \n',sigma_t)
        sigma = sigma_t.detach()
        return sigma
    
    def convert_tensor_to_numpy(self, tensor):
        if torch.is_tensor(tensor):
            array_np = tensor.detach().cpu().numpy()
            return array_np

    def forward(self, x1_feature, x2_feature):
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