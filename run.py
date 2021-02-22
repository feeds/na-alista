from utils.noise import GaussianNoise
from utils.train_utils import *
from utils.all_models import *

import torch

# try to use gpu
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
print("Using: " + device)

# set randomness
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

"""
Exemplary file to run experiments
=================================

The example runs some experiments that were evaluated for the paper.
Executing all runs required for complete reproduction takes quite some time. We provide results for seed=0 in the 
res/model folder.

You can define your own parameterizations of the models as in the utils.all_models file and use them here, 
e.g. changing the hidden layer size of NA-ALISTA.

All experiments are executed on a GPU if available.
Expected duration using a GPU for one run of NA_ALISTA, n=1000, k=16, noise=40db with the default settings below
will be approximately one hour (400 epochs).

"""

model_dir = 'res/models/'

# Default settings for reproducing our experiments.
m = 250  # measurements
s = 50  # sparsity
lr = 0.2 * 10e-4  # learning rate

for model_func in [NA_ALISTA_UR_128, ALISTA_AT, ALISTA, FISTA, ISTA, AGLISTA, NA_ALISTA_U_128, NA_ALISTA_R_128]:

    for k in [10, 12, 14, 16]: # number of iterations that the ISTA-style method is executed

        epoch = 100 + 20 * k

        for n in [750, 500, 1000]: # input size

            for noisename, noisefn in [["GaussianNoise40", GaussianNoise(40)], ["GaussianNoise20", GaussianNoise(20)]]:

                # apply the p-trick
                p = (np.linspace((s * 1 * 1.2) // k, s * 1 * 1.2, k)).astype(int)

                params = {
                    'model': model_func.__name__,
                    'm': m,
                    's': s,
                    'k': k,
                    'noise': noisename,
                    'n': n,
                }

                # filename for saving, do not change if you intend to use plot_results.ipynb
                name = '__'.join([f"{k}={v}" for k, v in params.items()])

                print(f"Running: {name}")

                # trains and saves model along with some training metrics
                train_model(m=m, n=n, s=s, k=k, p=p,
                            model_fn=model_func,
                            noise_fn=noisefn,
                            epochs=epoch,
                            initial_lr=lr,
                            name=name,
                            model_dir=model_dir)

                print("Done.")
