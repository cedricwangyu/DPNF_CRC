import numpy as np
import torch

torch.set_default_tensor_type(torch.DoubleTensor)


class Hierarchy:
    def __init__(self):
        self.sigma0 = 0.5
        self.sigma = 1
        self.NI = 2
        self.NJ = 50
        self.mu0 = 0
        self.data = None

    def genDataFile(self, dataFileName="data_hierarchy.txt", store=True):
        mu = torch.normal(self.mu0, self.sigma0, size=(self.NI, 1))
        self.data = mu + torch.normal(0, self.sigma, size=(self.NI, self.NJ))
        if store: np.savetxt(dataFileName, self.data)
        return self.data

    def den_t(self, params):
        mu0, mu = torch.split(params, [1, self.NI], dim=1)
        negll = torch.sum((mu - mu0) ** 2 / (2 * self.sigma0 ** 2), dim=1, keepdim=True) \
             + torch.sum(torch.sum((mu.unsqueeze(0) - self.data.t().unsqueeze(1)) ** 2 / (2 * self.sigma ** 2), dim=0),
                         dim=1, keepdim=True)
        return -negll

    def den_tt(self, params, ind=None):
        if ind is None:
            mu0, mu = torch.split(params, [1, self.NI], dim=1)
            negll = (mu - mu0) ** 2 / (2 * self.sigma0 ** 2) + torch.sum((mu.unsqueeze(0) - self.data.t().unsqueeze(1)) ** 2 / (2 * self.sigma ** 2), dim=0)
            return -negll
        else:
            mu0, mu = params[:, 0].reshape(-1, 1), params[:, ind+1].reshape(-1, 1)
            data = self.data[ind, :].unsqueeze(0)
            negll = (mu - mu0) ** 2 / (2 * self.sigma0 ** 2) + torch.sum(
                (mu.unsqueeze(0) - data.t().unsqueeze(1)) ** 2 / (2 * self.sigma ** 2), dim=0)
            return -negll

def Test_Hierarchy():
    # Define Class and parameters
    rt = Hierarchy()
    dataName = 'source/data/data_hierarchy_v.txt'
    rt.sigma0 = 5
    rt.NI = 100

    # Define Data
    if False:
        rt.genDataFile(dataName)

    # Load Data
    rt.data = np.loadtxt(dataName)

    # Test Model Solution and Neg_Log_Likelihood
    if False:
        params = torch.Tensor([[0, 0, 0],
                               [0, 0, 1],
                               [0, 1, 0],
                               [1, 0, 0]])

        print('parameters: \n', params)

        ll = - rt.den_t(params, torch.Tensor(rt.data))
        print('Model log-likelihood: \n', ll)
# TEST MODELS
if __name__ == "__main__":
    Test_Hierarchy()
