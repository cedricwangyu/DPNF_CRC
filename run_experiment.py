from experiment_setting import experiment
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy as sp
from maf import MADE, MAF, RealNVP
from train import Data, train, train_noisysgd_2
import time
from HierarchyModels import Hierarchy
from RegressionModels import Regression

start_time = time.time()
print('--- Numpy Version: ', np.__version__)
print('--- Scipy Version: ', sp.__version__)
print('--- Torch Version: ', torch.__version__)

torch.set_default_tensor_type(torch.DoubleTensor)


def execute(exp):
    # setup file ops
    if not os.path.isdir(exp.output_dir):
        os.makedirs(exp.output_dir)

    # setup device
    print("Cuda Availability: ", torch.cuda.is_available())
    device = torch.device('cuda:0' if torch.cuda.is_available() and not exp.no_cuda else 'cpu')
    torch.manual_seed(exp.seed)
    if device.type == 'cuda': torch.cuda.manual_seed(exp.seed)

    test_mode = 'reg'

    if test_mode == 'hierarchy2':
        rt = Hierarchy()
        rt.NI = 2
        exp.input_size = 3
        exp.n_blocks = 5
        exp.hidden_size = 100
        exp.lr = 0.0005
        exp.lr_decay = 0.9999
        exp.noisy = True
        exp.opt_batch_size = 2
        # exp.C = 0.03
        # exp.sigma = 50
        rt.data = torch.Tensor(np.loadtxt('source/data/data_hierarchy.txt'))
    elif test_mode == 'hierarchy100':
        rt = Hierarchy()
        rt.NI = 100
        exp.n_blocks = 5
        exp.hidden_size = 100
        exp.input_size = 101
        exp.lr = 0.003
        exp.lr_decay = 0.9999
        exp.noisy = True
        # exp.C = 0.02
        # exp.sigma = 7
        # exp.opt_batch_size = 25
        rt.data = torch.Tensor(np.loadtxt('source/data/data_hierarchy_100.txt'))
    elif test_mode == 'hierarchy100v':
        rt = Hierarchy()
        rt.NI = 100
        rt.sigma0 = 5
        exp.n_blocks = 5
        exp.hidden_size = 100
        exp.input_size = 101
        exp.lr = 0.003
        exp.lr_decay = 0.9999
        # exp.noisy = False
        exp.C = 0.04
        exp.sigma = 7
        # exp.opt_batch_size = 50
        rt.data = torch.Tensor(np.loadtxt('source/data/data_hierarchy_100v.txt'))
    elif test_mode == 'reg':
        exp.flow_type = 'maf'
        rt = Regression(device)
        rt.NI = 25000
        rt.sigma0 = 3.0
        rt.sigma = 0.5
        exp.n_blocks = 5
        exp.hidden_size = 175
        exp.input_size = 9
        exp.lr = 0.0001
        exp.lr_decay = 0.9995
        exp.batch_size = 500
        exp.n_iter = 6001
        exp.noisy = True
        exp.C = 1.0
        # exp.sigma = 1.0
        exp.opt_batch_size = 100

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

    SGD_mode = 'V4n'
    if SGD_mode == 'V1':
        for i in range(exp.n_iter):
            # scheduler.step()
            ind = torch.randint(rt.NI, (exp.opt_batch_size,)).to(device)
            if train(model, rt, optimizer, i, exp, loglist, ind=ind, sampling=True):
                pass
            else:
                return
    elif SGD_mode == 'V4n':
        i = 0
        while i < exp.n_iter:
            ind = torch.randint(rt.NI, (exp.opt_batch_size,)).to(device)
            if train_noisysgd_2(model, rt, optimizer, device, i, exp, loglist, sampling=True, ind=ind):
                pass
            else:
                return
            i += 1

    np.savetxt(exp.output_dir + '/' + exp.log_file, np.array(loglist), newline="\n")
    print("%s seconds" % (time.time() - start_time))


if __name__ == "__main__":
    # Settings
    exp = experiment()
    exp.name = "DP-NF-Exp1"  # str: Name of experiment                           default 'DP-NF-Exp1'
    exp.flow_type = 'maf'  # str: Type of flow                                 default 'realnvp'
    exp.n_blocks = 5  # int: Number of layers                             default 5
    exp.hidden_size = 100  # int: Hidden layer size for MADE in each layer     default 100
    exp.n_hidden = 1  # int: Number of hidden layers in each MADE         default 1
    exp.activation_fn = 'relu'  # str: Actication function used                     default 'relu'
    exp.input_order = 'sequential'  # str: Input order for create_mask                  default 'sequential'
    exp.batch_norm_order = True  # boo: Order to decide if batch_norm is used        default True

    exp.input_size = 3  # int: Dimensionality of input                      default 5
    exp.batch_size = 175  # int: Number of samples generated                  default 250
    exp.n_iter = 6001  # int: Number of iterations                         default 25000
    exp.lr = 0.0001  # float: Learning rate                              default 0.03
    exp.lr_decay = 0.9999  # float: Learning rate decay                        default 0.9995
    exp.log_interval = 100  # int: How often to show loss stat                  default 100

    exp.output_dir = './results/' + exp.name
    exp.seed = 12321
    # exp.seed = random.randint(0, 10 ** 9)
    exp.n_sample = 5000
    exp.no_cuda = False

    # DP parameters
    exp.noisy = False
    exp.C = 1.0
    exp.sigma = 1.0
    exp.opt_batch_size = 50

    # Experiment Setting
    exp.id = 1
    # execute(exp)
    sigmas = [5.100585702922621, 2.763723731926735, 1.9689303546651278, 1.5728252095039805, 1.3378229895544393,
              1.1832428373933588, 1.074244982227546, 0.9934104125117782, 0.9311122172152788, 0.8816230358493715,
              0.8413356872281041, 0.8078721900735089, 0.7796049169760273, 0.755384180803052, 0.7343758074014414]
    for sigma in sigmas:
        exp.name = str(exp.id)
        exp.output_dir = './results/' + exp.name
        exp.sigma = sigma
        print(exp.name, exp.sigma)
        execute(exp)
        exp.id += 1
