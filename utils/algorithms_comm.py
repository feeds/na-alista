import torch
import numpy as np
import torch.nn as nn
from utils.optimize_matrices import generalized_coherence
import utils.conf as conf

device = conf.device


def soft_threshold(x, theta, p):
    shape = x.shape
    x = x.reshape(x.shape[0], -1)
    if p == 0:
        return (torch.sign(x) * torch.relu(torch.abs(x) - theta)).reshape(shape)

    abs_ = torch.abs(x)
    topk, _ = torch.topk(abs_, int(p), dim=1)
    topk, _ = topk.min(dim=1)
    index = (abs_ > topk.unsqueeze(1)).float()
    return (index * x + (1 - index) * torch.sign(x) * torch.relu(torch.abs(x) - theta)).reshape(shape)


class ISTA(nn.Module):
    def __init__(self, m, n, k, forward_op, backward_op, L, lmbda):
        super(ISTA, self).__init__()

        self.m = m
        self.n = n
        self.k = k
        self.forward_op = forward_op
        self.backward_op = backward_op
        self.lmbda = lmbda
        self.L = L

    def forward(self, y, info):
        x = torch.zeros((y.shape[0], self.n), device=device)
        x = torch.zeros(self.backward_op(y).shape, device=device)

        for i in range(self.k):
            a = y - self.forward_op(x)
            b = self.backward_op(a)
            x = soft_threshold(x + 1 / self.L * b, self.lmbda / self.L, 0)
        return x, 0, 0


class FISTA(nn.Module):
    def __init__(self, m, n, k, forward_op, backward_op, L, lmbda):
        super(FISTA, self).__init__()

        self.m = m
        self.n = n
        self.k = k
        self.forward_op = forward_op
        self.backward_op = backward_op
        self.lmbda = lmbda
        self.L = 1

    def forward(self, y, info):
        x = torch.zeros((y.shape[0], self.n), device=device)
        x = torch.zeros(self.backward_op(y).shape, device=device)

        t = 1
        z = x
        for i in range(self.k):
            zold = z
            a = y - self.forward_op(x)
            b = self.backward_op(a)
            z = soft_threshold(x + 1 / self.L * b, self.lmbda / self.L, 0)

            t0 = t
            t = (1.0 + np.sqrt(1.0 + 4.0 * t ** 2)) / 2.0
            x = z + ((t0 - 1.0) / t) * (z - zold)
        return x, 0, 0


class ALISTA(nn.Module):
    def __init__(self, m, n, k, forward_op, backward_op, s, p):

        super(ALISTA, self).__init__()
        self.m = m
        self.n = n
        self.k = k
        self.forward_op = forward_op
        self.backward_op = backward_op
        self.p = p
        self.s = s

        self.gamma = nn.ParameterList([nn.Parameter(torch.ones(1) * 3) for i in range(k)])
        self.theta = nn.ParameterList([nn.Parameter(torch.ones(1) * 0.02) for i in range(k)])

    def forward(self, y, info, include_cs=False):
        # x = torch.zeros((y.shape[0],self.n), device=device)
        x = torch.zeros(self.backward_op(y).shape, device=device)

        cs = []
        for i in range(self.k):
            a = self.forward_op(x)
            b = a - y
            c = self.backward_op(b)
            x = soft_threshold(x - self.gamma[i][0] * c, self.theta[i][0], self.p[i])

            cs.append(torch.norm(c, dim=0, p=1).reshape(-1, 1))

        if include_cs:
            return x, torch.zeros(self.k, 1), torch.zeros(self.k, 1), torch.cat(cs, dim=1)
        return x, torch.zeros(self.k, 1), torch.zeros(self.k, 1)

    def save(self, name):
        torch.save(self.state_dict(), name)

    def load(self, name):
        self.load_state_dict(torch.load(name, map_location=device))


def softsign(x):
    return torch.log(1 + torch.exp(x))


class NA_ALISTA(nn.Module):
    def __init__(self, m, n, k, forward_op, backward_op, s, p, lstm_input="c_b", lstm_hidden=128):

        super(NA_ALISTA, self).__init__()
        self.m = m
        self.n = n
        self.k = k
        self.forward_op = forward_op
        self.backward_op = backward_op
        self.s = s
        self.p = p

        if lstm_input == "c_b":
            self.regressor = NormLSTMCB(n, s, k, dim=lstm_hidden)
        elif lstm_input == "c":
            self.regressor = NormLSTMC(n, s, k, dim=lstm_hidden)
        elif lstm_input == 'b':
            self.regressor = NormLSTMB(n, s, k, dim=lstm_hidden)
        else:
            raise ValueError()

        self.alpha = 0.99

    def forward(self, y, info, include_cs=False):
        # x = torch.zeros(y.shape[0], self.n, device=device)
        x = torch.zeros(self.backward_op(y).shape, device=device)

        gammas = []
        thetas = []
        cs = []
        xk = []

        cellstate, hidden = self.regressor.get_initial(y.shape[0])

        for i in range(self.k):
            a = self.forward_op(x)
            b = a - y
            c = self.backward_op(b)
            pred, hidden, cellstate = self.regressor(b.reshape(x.shape[0], -1).T, c.reshape(x.shape[0], -1).T, hidden,
                                                     cellstate)
            gamma = pred[:, :1]
            theta = pred[:, 1:]
            gammas.append(gamma)
            thetas.append(theta)
            cs.append(torch.norm(c.reshape(x.shape[0], -1), dim=0, p=1).reshape(-1, 1))

            if len(c.shape) == 4:
                gamma = gamma.unsqueeze(2).unsqueeze(2)
            elif len(c.shape) == 3:
                gamma = gamma.unsqueeze(2)
            d = x - (gamma *3* c)
            x = soft_threshold(d, theta/20, self.p[i])
            xk.append(x.T)
        if include_cs:
            return x, torch.cat(gammas, dim=1), torch.cat(thetas, dim=1), torch.cat(cs, dim=1), xk
        return x, torch.cat(gammas, dim=1), torch.cat(thetas, dim=1)

    def save(self, name):
        torch.save(self.state_dict(), name)

    def load(self, name):
        self.load_state_dict(torch.load(name, map_location=device))


class NormLSTMC(nn.Module):
    def __init__(self, n, s, k, dim=128):
        super(NormLSTMC, self).__init__()
        self.n = n
        self.k = k
        self.s = s
        self.dim = dim
        self.lstm = nn.LSTMCell(1, dim)
        self.lll = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, 2)
        self.softplus = nn.Softplus()

        self.hidden = nn.Parameter(torch.Tensor(np.random.normal(size=self.dim)))
        self.cellstate = nn.Parameter(torch.Tensor(np.random.normal(size=self.dim)))

        self.cl1_mean = 0
        self.cl1_std = 0
        self.initialized_normalizers = False

    def get_initial(self, batch_size):
        return (
            self.cellstate.unsqueeze(0).repeat(batch_size, 1),
            self.hidden.unsqueeze(0).repeat(batch_size, 1),
        )

    def forward(self, b, c, hidden, cellstate):
        cl1 = torch.norm(c, dim=0, p=1)
        if not self.initialized_normalizers:
            self.cl1_mean = cl1.mean().item()
            self.cl1_std = cl1.std().item()
            self.initialized_normalizers = True

        stack = torch.stack([(cl1 - self.cl1_mean) / self.cl1_std]).T
        hidden, cellstate = self.lstm(stack, (hidden, cellstate))
        out = self.softplus(self.linear(torch.relu(self.lll(cellstate))))
        return out, hidden, cellstate


class NormLSTMCB(nn.Module):
    def __init__(self, n, s, k, dim=128):
        super(NormLSTMCB, self).__init__()
        self.n = n
        self.k = k
        self.s = s
        self.dim = dim
        self.lstm = nn.LSTMCell(2, dim)
        self.lll = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, 2)
        self.softplus = nn.Softplus()

        self.hidden = nn.Parameter(torch.Tensor(np.random.normal(size=self.dim)))
        self.cellstate = nn.Parameter(torch.Tensor(np.random.normal(size=self.dim)))

        self.bl1_mean = 0
        self.cl1_mean = 0
        self.bl1_std = 0
        self.cl1_std = 0
        self.initialized_normalizers = False

    def get_initial(self, batch_size):
        return (
            self.cellstate.unsqueeze(0).repeat(batch_size, 1),
            self.hidden.unsqueeze(0).repeat(batch_size, 1),
        )

    def forward(self, b, c, hidden, cellstate):
        bl1 = torch.norm(b, dim=0, p=1)
        cl1 = torch.norm(c, dim=0, p=1)
        if not self.initialized_normalizers:
            self.bl1_mean = bl1.mean().item()
            self.bl1_std = bl1.std().item()
            self.cl1_mean = cl1.mean().item()
            self.cl1_std = cl1.std().item()
            self.initialized_normalizers = True

        stack = torch.stack([(bl1 - self.bl1_mean) / self.bl1_std, (cl1 - self.cl1_mean) / self.cl1_std]).T

        hidden, cellstate = self.lstm(stack, (hidden, cellstate))
        out = self.softplus(self.linear(torch.relu(self.lll(cellstate))))
        return out, hidden, cellstate


def g(x, epsilon=0.1):
    return 1 / (x / epsilon + 1)


class NormLSTMB(nn.Module):
    def __init__(self, n, s, k, dim=128):
        super(NormLSTMB, self).__init__()
        self.n = n
        self.k = k
        self.s = s
        self.dim = dim
        self.lstm = nn.LSTMCell(1, dim)
        self.lll = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, 2)
        self.softplus = nn.Softplus()

        self.hidden = nn.Parameter(torch.Tensor(np.random.normal(size=self.dim)))
        self.cellstate = nn.Parameter(torch.Tensor(np.random.normal(size=self.dim)))

        self.bl1_mean = 0
        self.bl1_std = 0
        self.initialized_normalizers = False

    def get_initial(self, batch_size):
        return (
            self.cellstate.unsqueeze(0).repeat(batch_size, 1),
            self.hidden.unsqueeze(0).repeat(batch_size, 1),
        )

    def forward(self, b, c, hidden, cellstate):
        bl1 = torch.norm(b, dim=0, p=1)
        if not self.initialized_normalizers:
            self.bl1_mean = bl1.mean().item()
            self.bl1_std = bl1.std().item()
            self.initialized_normalizers = True

        stack = torch.stack([(bl1 - self.bl1_mean) / self.bl1_std]).T

        hidden, cellstate = self.lstm(stack, (hidden, cellstate))
        out = self.softplus(self.linear(torch.relu(self.lll(cellstate))))
        return out, hidden, cellstate


class ALISTA_AT(nn.Module):
    def __init__(self, m, n, k, forward_op, backward_op, s, p):
        super(ALISTA_AT, self).__init__()
        self.m = m
        self.n = n
        self.k = k
        self.forward_op = forward_op
        self.backward_op = backward_op
        self.p = p
        self.s = s

        self.gamma = nn.ParameterList([nn.Parameter(torch.ones(1) * 3) for i in range(k)])
        self.theta = nn.ParameterList([nn.Parameter(torch.ones(1) * 0.02) for i in range(k)])

    def forward(self, y, info):
        # x = torch.zeros((y.shape[0], self.n), device=device)
        x = torch.zeros(self.backward_op(y).shape, device=device)

        for i in range(self.k):
            a = self.forward_op(x)
            b = a - y
            c = self.backward_op(b)

            theta = self.theta[i][0] * g(torch.abs(x))
            if len(theta.shape) >= 3:
                theta = theta.reshape(x.shape[0], -1)
            x = soft_threshold(x -  self.gamma[i][0] * c, theta, self.p[i])
        return x, torch.zeros(self.k, 1), torch.zeros(self.k, 1)

    def save(self, name):
        torch.save(self.state_dict(), name)

    def load(self, name):
        self.load_state_dict(torch.load(name, map_location=device))


class AGLISTA(nn.Module):
    def __init__(self, m, n, k, forward_op, backward_op, s, p):

        super(AGLISTA, self).__init__()
        self.m = m
        self.n = n
        self.k = k
        self.forward_op = forward_op
        self.backward_op = backward_op
        self.p = p
        self.s = s

        self.gamma = nn.ParameterList([nn.Parameter(torch.ones(1) * 3) for _ in range(k)])
        self.theta = nn.ParameterList([nn.Parameter(torch.ones(1) * 0.02) for _ in range(k)])
        self.a = nn.ParameterList([nn.Parameter(torch.ones(1) * 0.01) for _ in range(k)])
        self.v = nn.ParameterList([nn.Parameter(torch.ones(1)) for _ in range(k)])
        self.vu = nn.ParameterList([nn.Parameter(torch.ones(1) * 0.05) for _ in range(k)])
        self.eps = 0.01
        self.theta_initial = nn.ParameterList([nn.Parameter(torch.ones(1))])

    def forward(self, y, info):
        x = torch.zeros(self.backward_op(y).shape, device=device)
        for i in range(self.k):

            if i > 0:
                t = self.theta[i][0]
            else:
                t = self.theta_initial[0][0]

            gain = 1 + t * self.vu[i][0] * torch.exp(-self.v[i] * torch.abs(x))
            a = self.forward_op(gain * x)
            b = a - y

            c = self.backward_op(b)
            x_ = soft_threshold(x - self.gamma[i][0] * c, self.theta[i][0], self.p[i])

            overshoot = 1 + self.a[i][0] / (torch.abs(x_ - x) + self.eps)
            x = overshoot * x_ + (1 - overshoot) * x

        return x, torch.zeros(self.k, 1), torch.zeros(self.k, 1)

    def save(self, name):
        torch.save(self.state_dict(), name)

    def load(self, name):
        self.load_state_dict(torch.load(name, map_location=device))
