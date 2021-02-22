from utils.optimize_matrices import get_matrices
import torch
import torch.nn as nn
import numpy as np
import os
import pandas as pd
from utils.get_data import Synthetic, ComplexVectorDataset
import utils.algorithms as algo_norm
import utils.algorithms_comm as algo_comm
from time import time

import utils.conf as conf

device = conf.device

non_learned_algos = [algo_norm.ISTA, algo_norm.FISTA, algo_comm.ISTA, algo_comm.FISTA]


def train_model(m, n, s, k, p, model_fn, noise_fn, epochs, initial_lr, name, model_dir='res/models/',
                matrix_dir='res/matrices/'):
    if not os.path.exists(model_dir + name):
        os.makedirs(model_dir + name)

    if os.path.isfile(model_dir + name + "/train_log"):
        print("Results for " + name + " are already available. Skipping computation...")
        return None

    phi, W_frob = get_matrices(m, n, matrix_dir=matrix_dir)
    data = Synthetic(m, n, s, s)
    # put W_frob = reverse tv norm operator ...
    model = model_fn(m, n, s, k, p, phi, W_frob=W_frob, ).to(device)

    if type(model) not in non_learned_algos:
        opt = torch.optim.Adam(model.parameters(), lr=initial_lr)

    train_losses = []
    train_dbs = []
    test_losses = []
    test_dbs = []

    if type(model) in non_learned_algos:
        epochs = 1
    for i in range(epochs):
        if type(model) not in non_learned_algos:
            train_loss, train_db = train_one_epoch(model, data.train_loader, noise_fn, opt)
        else:
            train_loss = 0
            train_db = 0
        test_loss, test_db = test_one_epoch(model, data.test_loader, noise_fn)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_dbs.append(train_db)
        test_dbs.append(test_db)

        if test_dbs[-1] == min(test_dbs) and type(model) not in non_learned_algos:
            print("saving!")
            model.save(model_dir + name + "/checkpoint")

        data.train_data.reset()

        print(i, train_db, test_db)

    print("saving results to " + model_dir + name + "/train_log")
    pd.DataFrame(
        {
            "epoch": range(epochs),
            "train_loss": train_losses,
            "test_loss": test_losses,
            "train_dbs": train_dbs,
            "test_dbs": test_dbs,
        }
    ).to_csv(model_dir + name + "/train_log")


def train_model_communication(m, n, s, k, p, model_fn, noise_fn, epochs, initial_lr, name, model_dir='res/models/',
                              matrix_dir='res/matrices/'):
    if not os.path.exists(model_dir + name):
        os.makedirs(model_dir + name)

    if os.path.isfile(model_dir + name + "/train_log"):
        print("Results for " + name + " are already available. Skipping computation...")
        return None

    torch.manual_seed(3)

    L = 1

    ## ours
    dset = ComplexVectorDataset
    data = Synthetic(m, n, s, s, dataset=dset)
    train_loader = data.train_loader
    test_loader = data.test_loader

    # forward op: P_omega * fft_1d(x)
    # backward op: ifft(P_omega * y)
    overall_length = 1024
    padding = 212

    ii = (overall_length - n)
    P_omega = torch.zeros(n + ii)
    rs = np.random.RandomState(322)
    non_zero_m = rs.choice(list(range(padding, overall_length - padding)), m)
    print('m')
    print(m)
    print(non_zero_m)
    P_omega[non_zero_m] = 1.0  # take 10 measurements
    P_omega = torch.stack([P_omega, P_omega])
    P_omega = P_omega.to(device)

    def forward_op(x):
        other = torch.zeros(x.size()[0], ii, 2, device=device)
        _x = torch.cat([x, other], axis=1)
        return torch.fft(_x, 1, True) * P_omega.T

    wavelet = None
    backward_op = lambda y: torch.ifft(y * P_omega.T, 1, True)[:, :-ii, :]

    model = model_fn(m, n, s, k, p, forward_op, backward_op, L).to(device)

    if type(model) not in non_learned_algos:
        opt = torch.optim.Adam(model.parameters(), lr=initial_lr)

    train_losses = []
    train_dbs = []
    test_losses = []
    test_dbs = []

    if type(model) in non_learned_algos:
        epochs = 1
    for i in range(epochs):
        if type(model) not in non_learned_algos:
            train_loss, train_db = train_one_epoch_comm(model, train_loader, noise_fn, opt, transform=wavelet)
        else:
            train_loss = 0
            train_db = 0
        test_loss, test_db = test_one_epoch_comm(model, test_loader, noise_fn, transform=wavelet)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_dbs.append(train_db)
        test_dbs.append(test_db)

        if test_dbs[-1] == min(test_dbs) and type(model) not in non_learned_algos:
            print("saving!")
            model.save(model_dir + name + "/checkpoint")

        print(i, train_db, test_db)

    print("saving results to " + model_dir + name + "/train_log")
    pd.DataFrame(
        {
            "epoch": range(epochs),
            "train_loss": train_losses,
            "test_loss": test_losses,
            "train_dbs": train_dbs,
            "test_dbs": test_dbs,
        }
    ).to_csv(model_dir + name + "/train_log")

def train_one_epoch_comm(model, loader, noise_fn, opt, transform=None):
    train_loss = 0
    train_normalizer = 0
    for i, (X, info) in enumerate(loader):
        X = X.to(device)

        if transform is not None:
            # if i == 0:
            #  import matplotlib.pyplot as plt
            #  fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            #  ax[0, 0].imshow(X[0, 0].detach().cpu().numpy())
            X = transform.wt(X)
            # Xnorm = torch.norm(X.reshape(X.shape[0], -1), dim=1).reshape(X.shape[0], 1, 1, 1)
            # X = X/Xnorm

            # if i == 0:
            #  ax[0,1].imshow(transform.iwt(model.backward_op(model.forward_op(X)))[0,0].detach().cpu().numpy())

        info = info.to(device)
        opt.zero_grad()
        y = noise_fn(model.forward_op(X))  # (l,n,2)
        #print(y.size())
        #print((torch.norm(y[:, :, 0], dim=-1) ** 2 + torch.norm(y[:, :, 1], dim=-1) ** 2).mean() / 0.2 / 1024)
        #print((torch.norm(X[:, :, 0], dim=-1) ** 2 + torch.norm(X[:, :, 1], dim=-1) ** 2).mean() / 1024)
        # print(torch,)
        X_hat, gammas, thetas = model(y, info)

        # Xhatnorm = torch.norm(X.reshape(X.shape[0], -1), dim=1).reshape(X.shape[0], 1, 1, 1)
        if transform is not None:
            loss = ((transform.iwt(X_hat) - transform.iwt(X)) ** 2).mean()
        else:
            loss = ((X_hat - X) ** 2).mean()
        loss.backward()

        # if transform is not None:
        # if i == 0:
        #  ax[1, 1].imshow(X_hat[0][0].detach().cpu().numpy())
        #  ax[1,0].imshow(transform.iwt(X_hat)[0][0].detach().cpu().numpy())
        #  plt.show()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        train_normalizer += (X ** 2).mean().item()
        train_loss += loss.item()
    return train_loss / len(loader), 10 * np.log10(train_loss / train_normalizer)

def train_one_epoch(model, loader, noise_fn, opt):
    train_loss = 0
    train_normalizer = 0
    for i, (X, info) in enumerate(loader):
        X = X.to(device)
        info = info.to(device)
        opt.zero_grad()
        y = torch.matmul(X, model.phi.T)
        X_hat, gammas, thetas = model(noise_fn(y), info)
        loss = ((X_hat - X) ** 2).mean()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        train_normalizer += (X ** 2).mean().item()
        train_loss += loss.item()
    return train_loss / len(loader), 10 * np.log10(train_loss / train_normalizer)


def test_one_epoch(model, loader, noise_fn):
    test_loss = 0
    test_normalizer = 0
    with torch.no_grad():
        for i, (X, info) in enumerate(loader):
            X = X.to(device)
            info = info.to(device)
            y = torch.matmul(X, model.phi.T)
            X_hat, gammas, thetas = model(noise_fn(y), info)
            test_loss += ((X_hat - X) ** 2).mean().item()
            test_normalizer += (X ** 2).mean().item()
    return test_loss / len(loader), 10 * np.log10(test_loss / test_normalizer)


def test_one_epoch_comm(model, loader, noise_fn, transform=None):
    test_loss = 0
    test_loss_no_recon = 0
    test_normalizer = 0
    with torch.no_grad():
        for i, (X, info) in enumerate(loader):
            X = X.to(device)

            #if i == 0:
            #    import matplotlib.pyplot as plt
            #    fig, ax = plt.subplots(3, 2, figsize=(10, 10))
            #    show(X.detach().cpu().numpy(), ax[0, 0])

                # Xnorm = torch.norm(X.reshape(X.shape[0], -1), dim=1).reshape(X.shape[0], 1, 1, 1)
                # X = X/Xnorm
            info = info.to(device)
            y = noise_fn(model.forward_op(X))
            #if i == 0:
            #    import matplotlib.pyplot as plt
            #    show(y.detach().cpu().numpy(), ax[1, 0])
            #    show(model.backward_op(y).detach().cpu().numpy(), ax[1, 1])
            X_hat, gammas, thetas = model(y, info)

            Xbackward = model.backward_op(y)
            # Xbackwardnorm = torch.norm(Xbackward.reshape(X.shape[0], -1), dim=1).reshape(X.shape[0], 1, 1, 1) / 0.01
            # Xhatnorm = torch.norm(X.reshape(X.shape[0], -1), dim=1).reshape(X.shape[0], 1, 1, 1) / 0.01

            #if i == 0:
            #    show(X_hat.detach().cpu().numpy(), ax[2, 1])
            #    plt.show()

            if transform is not None:
                test_loss += ((transform.iwt(X_hat) - transform.iwt(X)) ** 2).mean().item()
                test_loss_no_recon += ((transform.iwt(X) - transform.iwt(Xbackward)) ** 2).mean().item()
            else:
                test_loss += ((X_hat - X) ** 2).mean().item()
                test_loss_no_recon += ((X - Xbackward) ** 2).mean().item()

            test_normalizer += (X ** 2).mean().item()
    print("NO RECON:", 10 * np.log10(test_loss_no_recon / test_normalizer))
    return test_loss / len(loader), 10 * np.log10(test_loss / test_normalizer)


def evaluate_model(m, n, s, k, p, model_fn, noise_fn, name, model_dir='res/models/'):
    phi, W_soft_gen, W_frob = get_matrices(m, n)
    data = Synthetic(m, n, s, s)
    model = model_fn(m, n, s, k, p, phi, W_soft_gen, W_frob).to(device)
    model.load(model_dir + name + "/checkpoint")

    test_loss = []
    test_normalizer = []
    sparsities = []
    t1 = time()
    with torch.no_grad():
        for epoch in range(1):
            for i, (X, info) in enumerate(data.train_loader):
                sparsities.extend(list((X != 0).int().sum(dim=1).detach().numpy()))
                X = X.to(device)
                info = info.to(device)
                y = torch.matmul(X, model.phi.T)
                X_hat, gammas, thetas = model(noise_fn(y), info)
                test_loss.extend(list(((X_hat - X) ** 2).cpu().detach().numpy()))
                test_normalizer.extend(list((X ** 2).cpu().detach().numpy()))
            data.train_data.reset()
    t2 = time()
    runtime_evaluation = t2 - t1

    test_loss = np.array(test_loss)
    test_normalizer = np.array(test_normalizer)
    sparsities = np.array(sparsities)

    keys = []
    counts = []
    values = []
    for s in sorted(np.unique(sparsities)):
        count = (sparsities == s).mean()
        if count > 10e-5:
            keys.append(s)
            counts.append(count)
            values.append(
                10
                * np.log10(
                    np.sum(test_loss[sparsities == s]) / np.sum(test_normalizer[sparsities == s])
                )
            )

    return keys, counts, values, 10 * np.log10(np.sum(test_loss) / np.sum(test_normalizer))
