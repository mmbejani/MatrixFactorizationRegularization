import torch
from torch import nn
import numpy as np
import numpy.linalg as linalg
from sklearn.decomposition import NMF
from sklearn.neighbors import NearestNeighbors
from typing import Callable


class DSR(nn.Module):

    def __init__(self, net:nn.Module, M=2.0, device='cuda'):
        super().__init__()
        self.M = M
        self.net = net
        self.device = device

    @staticmethod
    def optimal_d(s):
        variance = np.std(s)
        mean = np.average(s)
        for i in range(s.shape[0] - 1):
            if s[i] < mean + variance:
                return i
        return s.shape[0] - 1

    def approximate_svd_tensor(self, w: np.ndarray) -> np.ndarray:
        w_shape = w.shape
        n1 = w_shape[0]
        n2 = w_shape[1]
        ds = []
        if w_shape[2] == 1 or w_shape[3] == 1:
            return w
        u, s, v = linalg.svd(w)
        for i in range(n1):
            for j in range(n2):
                ds.append(DSR.optimal_d(s[i, j]))
        d = int(np.mean(ds))
        w = np.matmul(u[..., 0:d], s[..., 0:d, None] * v[..., 0:d, :])
        return w
		
    def approximate_svd_matrix(self, w):
        u, s, v = linalg.svd(w)
        d = DSR.optimal_d(s)
        w = np.matmul(u[:, 0:d], np.matmul(np.diag(s[0:d]), v[:,0:d]))
        return w

    def regularize(self, train_loss, test_loss):
        v = test_loss/train_loss
        if v > self.M:
            for p in list(self.net.parameters()):
                if len(p.size()) > 1:
                    if len(p.size()) == 2 and np.prod(p.size()) < 10000:
                        w = p.data.detach().cpu().numpy()
                        w = self.approximate_svd_matrix(w)
                        if self.device == 'cuda':
                            p.data = torch.tensor(w,requires_grad=True).cuda()
                        else:
                            p.data = torch.tensor(w,requires_grad=True)

                    elif len(p.size()) == 4:
                        w = p.data.detach().cpu().numpy()
                        w = self.approximate_svd_tensor(w)
                        if self.device == 'cuda':
                            p.data = torch.tensor(w,requires_grad=True).cuda()
                        else:
                            p.data = torch.tensor(w,requires_grad=True)
                    
                    


class DLRF(nn.Module):

    def __init__(self, net: nn.Module, damped: Callable = None, device='cuda'):
        super().__init__()
        self.net = net
        self.damped = damped
        self.k = 1
        self.device = device
        self.hist = list()

    def approximate_lrf_tensor_kernel_filter_wise(self, w):
        for i in range(w.shape[0]):
            for j in range(w.shape[1]):
                m = np.min(w[i, j, :, :])
                w[i, j, :, :] -= m
                mdl = NMF(n_components=self.k, max_iter=2, tol=1.0)
                W = mdl.fit_transform(np.reshape(w[i, j, :, :], [w.shape[2], w.shape[3]]))
                H = mdl.components_
                w[i, j, :, :] = np.matmul(W, H) + m
        return w

    def approximation_nmf_matrix(self, w):
        m = np.min(w)
        w -= m
        mdl = NMF(n_components=self.k, max_iter=20, tol=1.0)
        W = mdl.fit_transform(w)
        H = mdl.components_
        return np.matmul(W, H) + m

    def compute_condition_number(self, loss_value: torch.Tensor, verbose=False):
        params = list(self.net.parameters())
        print(len(params))
        condition_number_list = list()
        for p in params:
            if len(p.size()) > 1:
                j_theta_norm = torch.norm(p.grad)
                theta_norm = torch.norm(p)
                condition_number = j_theta_norm * theta_norm / loss_value
                condition_number_list.append(condition_number)

        return condition_number_list

    def regularize(self, train_loss, test_loss, epoch):
        v = test_loss / train_loss
		self.hist.append(v)
        if len(self.hist) > 3 and (self.hist[-1] - self.hist[-2]) > 0 and (self.hist[-2] - self.hist[-3]) > 0:
            condition_number_list = self.compute_condition_number(train_loss)
            max_condition_number = max(condition_number_list)
            counter = 0
            for p in list(self.net.parameters()):
                if len(p.size()) > 1:
                    c = condition_number_list[counter] / max_condition_number
                    r = np.random.rand()
                    if self.damped is not None:
                        r *= 1/(self.damped(epoch) + 1)
                    w = p.detach().cpu().numpy()
                    if r < c:
                        if len(w.shape) == 2 and np.prod(w.shape) < 1000:
                                w = self.approximation_nmf_matrix(w)
                        if len(w.shape) == 4:
                                w = self.approximate_lrf_tensor_kernel_filter_wise(w)
                        if self.device == 'cuda':
                            p.data = torch.tensor(w,requires_grad=True).cuda()
                        else:
                            p.data = torch.tensor(w,requires_grad=True)
                    counter += 1
