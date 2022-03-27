import os
import numpy as np
import torch
import torch.distributions as D

torch.set_default_tensor_type(torch.DoubleTensor)


class Regression:
    def __init__(self, device):
        self.device = device
        self.sigma0 = 3.0
        self.sigma = 0.5
        self.NI = 25000
        self.NJ = 5
        self.defbeta = torch.Tensor([0.2, -0.8, 1.0, -1.2, 0.8, -1.2, -0.6, 0.6, 1.2]).to(device)
        self.X_dir = "source/data/data_reg_X.txt"
        self.Y_dir = "source/data/data_reg_Y.txt"
        self.X = None if not os.path.exists(self.X_dir) else torch.Tensor(np.loadtxt(self.X_dir)).to(device)
        self.Y = None if not os.path.exists(self.Y_dir) else torch.Tensor(np.loadtxt(self.Y_dir)).to(device)
        self.defout = self.solve(self.defbeta.unsqueeze(0)) if self.X is not None else None
        self.Cov = (torch.Tensor(self.NJ, self.NJ).fill_(self.sigma0 ** 2) + torch.eye(self.NJ) * self.sigma ** 2).to(device)
        self.dist = D.MultivariateNormal(loc=torch.zeros(self.NJ).to(device), covariance_matrix=self.Cov)


    def genX(self, store=True):
        if os.path.exists(self.X_dir):
            print("X already exists.")
            return
        res = torch.Tensor(self.NI, 4).uniform_(0, 1) * torch.Tensor([1.0, 3.0, 0.5, 2.0])
        if store: np.savetxt(self.X_dir, res)
        self.X = res

    def genY(self, store=True):
        if os.path.exists(self.Y_dir):
            print("Y already exists.")
            return
        res = self.defout.reshape(-1, 1) \
              + torch.normal(0, self.sigma0, size=(self.NI, 1)) \
              + torch.normal(0, self.sigma, size=(self.NI, self.NJ))
        if store: np.savetxt(self.Y_dir, res)
        self.Y = res

    def solve(self, param, ind=None):
        X = self.X if ind is None else self.X[ind, :]
        b1, b2, b3, b4, b5, b6, b7, b8, b9 = torch.split(param, 1, dim=1)
        x1, x2, x3, x4 = torch.split(X.t(), 1, dim=0)
        Ey = b1 \
             + b2 * torch.exp(x1 + b3) \
             + b4 * torch.log(x2 + b5 ** 2) \
             + b6 * torch.exp(x3 + b7) \
             + b8 * torch.log(x4 + b9 ** 2)
        return Ey

    def den_t(self, param, ind=None):
        if self.X is None or self.Y is None: raise ValueError('Absence of X and Y data.')
        if ind is None:
            diff = (self.solve(param, ind=ind).unsqueeze(0) - self.Y.t().unsqueeze(1)).reshape(self.NJ, -1).t()
            res = self.dist.log_prob(diff).reshape(-1, self.Y.size(0)).sum(1).reshape(-1, 1)
        else:
            Y = self.Y[ind, :]
            diff = (self.solve(param, ind=ind).unsqueeze(0) - Y.t().unsqueeze(1)).reshape(self.NJ, -1).t()
            res = self.dist.log_prob(diff).reshape(-1, Y.size(0))
        return res

    def den_tt(self, param, ind=None):
        if self.X is None or self.Y is None: raise ValueError('Absence of X and Y data.')
        if ind is None:
            diff = (self.solve(param, ind=ind).unsqueeze(0) - self.Y.t().unsqueeze(1)).reshape(self.NJ, -1).t()
            res = self.dist.log_prob(diff).reshape(-1, self.Y.size(0))
        else:
            Y = self.Y[ind, :]
            diff = (self.solve(param, ind=ind).unsqueeze(0) - Y.t().unsqueeze(1)).reshape(self.NJ, -1).t()
            res = self.dist.log_prob(diff).reshape(-1, Y.size(0))
        return res

def Test_Regression():
    # Define Class and parameters
    device = torch.device('cpu')
    rg = Regression(device)

    # Define Data
    if False:
        rg.genX()
        rg.genY()

    # Test Model Solution and Neg_Log_Likelihood
    if False:
        # params = torch.Tensor(5, 9).uniform_(0, 1) * torch.Tensor([1, 2, 3, 4, 5, 4, 3, 2, 1])
        params = rg.defbeta.unsqueeze(0)
        print('parameters: \n', params)
        out = rg.solve(params)
        print('out: \n', out)
        print(list(out.size()))

    # Test LL
    if False:
        params = torch.Tensor([[1.0, 2.0, 0.5, -1.5, 1.5, -4.0, -3.5, -0.5, 0.5],
                               [0.0, 2.0, 0.5, -1.5, 1.5, -4.0, -3.5, -0.5, 0.5],
                               [1.0, 0.0, 0.5, -1.5, 1.5, -4.0, -3.5, -0.5, 0.5],
                               [1.0, 2.0, 0.0, -1.5, 1.5, -4.0, -3.5, -0.5, 0.5],
                               [1.0, 2.0, 0.5, 0.0, 1.5, -4.0, -3.5, -0.5, 0.5],
                               [1.0, 2.0, 0.5, -1.5, 0.0, -4.0, -3.5, -0.5, 0.5],
                               [1.0, 2.0, 0.5, -1.5, 1.5, -0.0, -3.5, -0.5, 0.5],
                               [1.0, 2.0, 0.5, -1.5, 1.5, -4.0, -0.0, -0.5, 0.5],
                               [1.0, 2.0, 0.5, -1.5, 1.5, -4.0, -3.5, -0.0, 0.5],
                               [1.0, 2.0, 0.5, -1.5, 1.5, -4.0, -3.5, -0.5, 0.0]])
        ind = np.array([1, 2, 3, 4, 45])
        ind = torch.as_tensor(ind)
        ll = rg.den_t(params, ind=ind)
        print(ll)

    # Test computational efficiency
    if False:
        params = torch.Tensor(250, 9).uniform_(-1, 1) + rg.defbeta
        ll = rg.den_t(params)
        print(ll)

    # Evaluation
    if True:
        params = torch.Tensor(np.loadtxt('results/DP-NF-Exp1/samples6000'))
        out = rg.solve(params)
        print(rg.defout)
        res = out-rg.defout
        res = torch.cat([torch.mean(res, dim=1, keepdim=True), torch.std(res, dim=1, keepdim=True)], dim=1)
        np.savetxt('results/DP-NF-Exp1/MAF_out_diff_info.txt', res.detach().numpy())




#         ll = - rt.den_t(params, torch.Tensor(rt.data))
#         print('Model log-likelihood: \n', ll)
# TEST MODELS
if __name__ == "__main__":
    # Test_Regression()
    device = torch.device('cpu')
    rg = Regression(device)
    for i in range(1, 10):
        folder = 'reg/' + str(i)
        params = torch.Tensor(np.loadtxt(folder + '/samples6000'))
        out = rg.solve(params)
        print(rg.defout)
        res = out - rg.defout
        res = torch.cat([torch.mean(res, dim=1, keepdim=True), torch.std(res, dim=1, keepdim=True)], dim=1)
        np.savetxt(folder + '/MAF_out_diff_info.txt', res.detach().numpy())

    # folder = 'reg/'
    # params = torch.Tensor(np.loadtxt(folder + '/samples6000'))
    # out = rg.solve(params)
    # print(rg.defout)
    # res = out - rg.defout
    # res = torch.cat([torch.mean(res, dim=1, keepdim=True), torch.std(res, dim=1, keepdim=True)], dim=1)
    # np.savetxt(folder + '/MAF_out_diff_info.txt', res.detach().numpy())
