from experiment_setting import experiment
import os
import argparse
import torch
from torch import nn
import numpy as np
from maf import MADE, MAF, RealNVP
import random
import copy

# print('--- Numpy Version: ', np.__version__)
# print('--- Scipy Version: ', sp.__version__)
# print('--- Torch Version: ', torch.__version__)

torch.set_default_tensor_type(torch.DoubleTensor)
parser = argparse.ArgumentParser(description='EHR')
parser.add_argument('--data', type=int, default=1, metavar='N', help='data id: 1 - 10')
parser.add_argument('--sigma', type=float, default=5.0, metavar='G', help='noise added')
parser.add_argument('--start', type=int, default=1, metavar='G', help='start trial')
parser.add_argument('--end', type=int, default=10, metavar='G', help='end trial')
args = parser.parse_args()

def _del_nested_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        _del_nested_attr(getattr(obj, names[0]), names[1:])


def _set_nested_attr(obj, names, value):
    if len(names) == 1:
        setattr(obj, names[0], value)
    else:
        _set_nested_attr(getattr(obj, names[0]), names[1:], value)


def extract_weights(mod):
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        _del_nested_attr(mod, name.split("."))
        names.append(name)

    params = tuple(p.detach().requires_grad_() for p in orig_params)
    return params, names


def load_weights(mod, names, params):
    for name, p in zip(names, params):
        _set_nested_attr(mod, name.split("."), p)


def train(model, x, optimizer, i, exp, loglist, device=None):
    model.train()
    if device is not None: x = x.to(device)
    loss = - model.log_prob(x).mean(0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % exp.log_interval == 0:
        loglist.append(loss.item())
        # print('epoch {}: loss {:.4f}'.format(i, loss.item()))


def train_noisy(model, x, optimizer, i, exp, loglist, device=None):
    model.double()
    model.train()
    if device is not None: x = x.to(device)
    grads = [0 for _ in model.parameters()]
    log_probs = - model.log_prob(x)
    n = log_probs.size(0)
    for j, lp in enumerate(log_probs):
        optimizer.zero_grad()
        if j < n - 1:
            lp.backward(retain_graph=True)
        else:
            lp.backward()
        if exp.noisy:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=exp.C, norm_type=2)
        for jj, p in enumerate(model.parameters()):
            grads[jj] += p.grad.clone()

    optimizer.zero_grad()
    if exp.noisy:
        for j, p in enumerate(model.parameters()):
            # p.grad = (grads[j] + torch.randn(p.grad.size(), device=device) * 2 * exp.C * exp.sigma) / n
            p.grad = (grads[j] + torch.randn(p.grad.size(), device=device) * exp.C * exp.sigma) / n
    else:
        for j, p in enumerate(model.parameters()):
            p.grad = grads[j] / n

    optimizer.step()
    loss = log_probs.mean()
    if i % exp.log_interval == 0:
        loglist.append(loss.item())
        # print('epoch {}: loss {:.4f}'.format(i, loss.item()))


def train_noisy_boost(model, model_copy, names, x, optimizer, i, exp, loglist, device=None):
    model.double()
    model.train()
    if device is not None: x = x.to(device)

    def model_func(*new_params):
        load_weights(model_copy, names, new_params)
        return -model_copy.log_prob(x)

    jacobians = torch.autograd.functional.jacobian(model_func, tuple(model.parameters()))
    optimizer.zero_grad()
    if exp.noisy:
        for j in range(len(jacobians[0])):
            nn.utils.clip_grad_norm_((jacobians[i][j] for i in range(len(jacobians))), max_norm=exp.C, norm_type=2)
        for g, p in zip(jacobians, model.parameters()):
            p.grad = g.mean(0) + torch.randn(p.size(), device=device) * exp.C * exp.sigma
    else:
        for g, p in zip(jacobians, model.parameters()):
            p.grad = g.mean(0)
    optimizer.step()
    if i % exp.log_interval == 0:
        loss = - model.log_prob(x).mean()
        loglist.append(loss.item())
        # print('epoch {}: loss {:.4f}'.format(i, loss.item()))


@torch.no_grad()
def generate(model, sample_size, directory=None):
    model.eval()
    u = model.base_dist.sample((sample_size,))
    samples, _ = model.inverse(u)
    if directory:
        np.savetxt(directory, samples.cpu().numpy())
    else:
        np.savetxt(exp.output_dir + "/generated_EHR.txt", samples.cpu().numpy())


def execute(exp):
    # setup file ops
    if not os.path.isdir(exp.output_dir):
        os.makedirs(exp.output_dir)

    # setup device
    # print("Cuda Availability: ", torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')
    torch.manual_seed(exp.seed)
    if device.type == 'cuda': torch.cuda.manual_seed(exp.seed)

    data = torch.Tensor(np.loadtxt(exp.data_dir))[:64, ]

    # model
    if exp.flow_type == 'made':
        model = MADE(exp.input_size, exp.hidden_size, exp.n_hidden, None, exp.activation_fn, exp.input_order)
    elif exp.flow_type == 'maf':
        model = MAF(exp.n_blocks, exp.input_size, exp.hidden_size, exp.n_hidden, None,
                    exp.activation_fn, exp.input_order, batch_norm=exp.batch_norm_order)
    elif exp.flow_type == 'realnvp':  # Under construction
        model = RealNVP(exp.n_blocks, exp.input_size, exp.hidden_size, exp.n_hidden, None,
                        batch_norm=exp.batch_norm_order)
    else:
        raise ValueError('Unrecognized model.')

    model = model.to(device)
    # optimizer = torch.optim.RMSprop(model.parameters(), lr=exp.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=exp.lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, exp.lr_decay)

    loglist = []

    mode = 'noise'
    if mode == 'no_noise':
        for i in range(exp.n_iter):
            # scheduler.step()
            ind = torch.Tensor(data.size(0)).uniform_(0, 1) < 0.5
            train(model, data[ind, :], optimizer, i, exp, loglist, device)
    elif mode == 'noise':
        for i in range(exp.n_iter):
            ind = torch.Tensor(data.size(0)).uniform_(0, 1) < 0.5
            train_noisy(model, data[ind, :], optimizer, i, exp, loglist, device)
    elif mode == 'noise_boost':
        model_copy = copy.deepcopy(model)
        model_copy = model_copy.to(device)
        _, names = extract_weights(model_copy)
        for i in range(exp.n_iter):
            ind = torch.Tensor(data.size(0)).uniform_(0, 1) < 0.5
            train_noisy_boost(model, model_copy, names, data[ind, ], optimizer, i, exp, loglist, device)

    # generate(model, 200)
    torch.save(model.state_dict(), exp.output_dir + '/' + 'MAF_params_' + str(exp.id))
    np.savetxt(exp.output_dir + '/' + exp.log_file, np.array(loglist), newline="\n")


if __name__ == "__main__":
    # Settings
    exp = experiment()
    exp.name = "EHR-1"  # str: Name of experiment                           default 'DP-NF-Exp1'
    exp.flow_type = 'maf'  # str: Type of flow                                 default 'realnvp'
    exp.n_blocks = 15  # int: Number of layers                             default 5
    exp.hidden_size = 200  # int: Hidden layer size for MADE in each layer     default 100
    exp.n_hidden = 1  # int: Number of hidden layers in each MADE         default 1
    exp.activation_fn = 'relu'  # str: Actication function used                     default 'relu'
    exp.input_order = 'sequential'  # str: Input order for create_mask                  default 'sequential'
    exp.batch_norm_order = True  # boo: Order to decide if batch_norm is used        default True
    exp.data_dir = "source/data/EHR.txt"

    exp.input_size = 19  # int: Dimensionality of input                      default 5
    # exp.batch_size = 175  # int: Number of samples generated                  default 250
    exp.n_iter = 8001  # int: Number of iterations                         default 25000
    exp.lr = 0.00002  # float: Learning rate                              default 0.03
    exp.lr_decay = 0.9999  # float: Learning rate decay                        default 0.9995
    exp.log_interval = 100  # int: How often to show loss stat                  default 100

    exp.output_dir = './results/' + exp.name
    exp.seed = 1123
    # exp.seed = random.randint(0, 10 ** 9)
    exp.n_sample = 5000
    exp.no_cuda = True

    # DP parameters
    exp.noisy = True
    exp.C = 7
    exp.sigma = 15
    exp.opt_batch_size = 12

    # Experiment Setting
    exp.id = 0
    # execute(exp)

    # for sigma in [5, 7, 9, 11, 13, 15]:
    #     exp.id += 1
    #     exp.sigma = sigma
    #     execute(exp)



    # Bulk Experiments
    # for data_id in range(1, 6):
    #     exp.data_dir = "source/imputed_EHR/EHR_" + str(data_id) + ".txt"
    #     for sigma in [0]:
    #         for trial in [1, 3, 4, 5, 6, 7]:
    #             print("Data: ", data_id, "sigma: ", sigma, "trial: ", trial)
    #             exp.seed = random.randint(0, 10 ** 9)
    #             exp.id = trial
    #             exp.sigma = sigma
    #             exp.log_file = "log_" + str(trial) + ".txt"
    #             exp.output_dir = "./results/impute_" + str(data_id) + "/sigma_" + str(sigma)
    #             execute(exp)

    # Reformat directory
    # import shutil
    # for data_id in range(1, 6):
    #     for sigma in [0, 5, 10, 15]:
    #         curr_dir = "results/impute_" + str(data_id) + "/sigma_" + str(sigma)
    #         for trial in [1, 3, 4, 5, 6, 7]:
    #             os.makedirs(curr_dir + "/trial_" + str(trial))
    #             shutil.move(curr_dir + "/log_" + str(trial) + ".txt", curr_dir + "/trial_" + str(trial) + "/log.txt")
    #             shutil.move(curr_dir + "/MAF_params_" + str(trial), curr_dir + "/trial_" + str(trial) + "/MAF_params")

    # Generate
    # for data_id in range(1, 6):
    #     exp.data_dir = "source/imputed_EHR/EHR_" + str(data_id) + ".txt"
    #     for sigma in [0, 5, 10, 15]:
    #         curr_dir = "results/impute_" + str(data_id) + "/sigma_" + str(sigma)
    #         for trial in [1, 3, 4, 5, 6, 7]:
    #             device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')
    #             exp.output_dir = curr_dir + "/trial_" + str(trial)
    #             torch.manual_seed(exp.seed)
    #             if device.type == 'cuda': torch.cuda.manual_seed(exp.seed)
    #             data = torch.Tensor(np.loadtxt(exp.data_dir))
    #             model = MAF(exp.n_blocks, exp.input_size, exp.hidden_size, exp.n_hidden, None, exp.activation_fn,
    #                         exp.input_order, batch_norm=exp.batch_norm_order)
    #             model = model.to(device)
    #             model.load_state_dict(torch.load(exp.output_dir + "/MAF_params"))
    #             for i in range(1, 51):
    #                 exp.seed = random.randint(0, 10 ** 9)
    #                 generate(model, 300, exp.output_dir + "/synthetic_" + str(i) + ".txt")

    # for data_id in range(1, 6):
    #     for sigma in [0, 5, 10, 15]:
    #         for trial in range(1, 7):
    #             curr_dir = "results/impute_" + str(data_id) + "/sigma_" + str(sigma) + "/trial_" + str(trial)
    #             os.remove(curr_dir + "/log.txt")
    #             os.remove(curr_dir + "/MAF_params")

    # exp.n_iter = 5
    random.seed(args.sigma * 1e5 + args.data * 1e3)
    for trial in range(args.start, args.end+1):
        exp.data_dir = "source/imputed_EHR/EHR_" + str(args.data) + ".txt"
        exp.id = trial
        exp.sigma = args.sigma
        exp.seed = random.randint(0, 10 ** 9)
        exp.log_file = "log_" + str(trial) + ".txt"
        exp.output_dir = "./results/impute_" + str(args.data) + "/sigma_" + str(args.sigma)
        execute(exp)
