import torch
import numpy as np
from torch import nn
# from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

torch.set_default_tensor_type(torch.DoubleTensor)


class Data(Dataset):
    def __init__(self, n):
        self.x = [i for i in range(n)]
        self.len = len(self.x)

    def __getitem__(self, index):
        return self.x[index]

    def __len__(self):
        return self.len


def train(nf, model, optimizer, iteration, exp, log, ind=None, sampling=True):
    nf.train()
    x0 = nf.base_dist.sample([exp.batch_size])
    xk, sum_log_abs_det_jacobians = nf(x0)
    if sampling and iteration % 200 == 0:
        x00 = nf.base_dist.sample([exp.n_sample])
        xkk, _ = nf(x00)
        np.savetxt(exp.output_dir + '/samples' + str(iteration), xkk.cpu().data.numpy(), newline="\n")

    if torch.any(torch.isnan(xk)):
        print("Error Iteration " + str(iteration))
        print(xk)
        np.savetxt(exp.output_dir + '/' + exp.log_file, np.array(log), newline="\n")
        return False
    optimizer.zero_grad()
    if ind is None:
        loss = (- torch.sum(sum_log_abs_det_jacobians, dim=1, keepdim=True) - model.den_t(xk)).mean()
    else:
        loss = (- torch.sum(sum_log_abs_det_jacobians, dim=1, keepdim=True) * exp.opt_batch_size / model.NI - torch.sum(model.den_t(xk, ind=ind), dim=1, keepdim=True)).mean()
    loss.backward()
    optimizer.step()
    print("{}\t{}".format(iteration, loss.item()), end='\r')
    if iteration % exp.log_interval == 0:
        print("{}\t{}".format(iteration, loss.item()))
        log.append([iteration, loss.item()])
    return True


def train_noisysgd_2(nf, model, optimizer, device, iteration, exp, log, ind=None, sampling=True):
    grads = [0 for _ in nf.parameters()]
    nf.train()
    x0 = nf.base_dist.sample([exp.batch_size])
    xk, sum_log_abs_det_jacobians = nf(x0)
    if sampling and iteration % 200 == 0:
        x00 = nf.base_dist.sample([exp.n_sample])
        xkk, _ = nf(x00)
        np.savetxt(exp.output_dir + '/samples' + str(iteration), xkk.cpu().data.numpy(), newline="\n")
    if torch.any(torch.isnan(xk)):
        print("Error Iteration " + str(iteration))
        print(xk)
        np.savetxt(exp.output_dir + '/' + exp.log_file, np.array(log), newline="\n")
        return False
    Jacobian = (- torch.sum(sum_log_abs_det_jacobians, dim=1, keepdim=True)).mean() * exp.opt_batch_size / model.NI
    LogJoint = (- model.den_tt(xk, ind=ind)).mean(0)
    for lj in LogJoint:
        optimizer.zero_grad()
        lj.backward(retain_graph=True)
        if exp.noisy:
            nn.utils.clip_grad_norm_(nf.parameters(), max_norm=exp.C, norm_type=2)
        for i, p in enumerate(nf.parameters()):
            grads[i] += p.grad.clone()
    optimizer.zero_grad()
    Jacobian.backward()
    if exp.noisy:
        nn.utils.clip_grad_norm_(nf.parameters(), max_norm=exp.C, norm_type=2)
    for i, p in enumerate(nf.parameters()):
        grads[i] += p.grad.clone()

    optimizer.zero_grad()
    if exp.noisy:
        for i, p in enumerate(nf.parameters()):
            p.grad = grads[i] + (torch.randn(p.grad.size()) * 2 * exp.C * exp.sigma).to(device)
            # p.grad = grads[i] + (torch.randn(p.grad.size()) * exp.C * exp.sigma).to(device)
    else:
        for i, p in enumerate(nf.parameters()):
            p.grad = grads[i]
    optimizer.step()
    loss = Jacobian + LogJoint.sum()
    print("{}\t{}".format(iteration, loss), end='\r')
    if iteration % exp.log_interval == 0:
        print("{}\t{}".format(iteration, loss))
        log.append([iteration, loss.item()])
    return True


if __name__ == "__main__":
    dataset = Data(100)
    trainloader = DataLoader(dataset=dataset, batch_size=5, shuffle=True)
    for x in trainloader: print(x)
    print("--------------")
    for x in trainloader: print(x)
