# Neurally-Augmented ALISTA

Code to reproduce the results from [Neurally Augmented ALISTA, ICLR 2021](https://openreview.net/forum?id=q_S44KLQ_Aa).
Freya Behrens, Jonathan Sauder, Peter Jung.

### Synthetic Data Experiments

Experiment demo:
```
python run.py
```
for parameterization, refer to the comments in the file.

Reproduce the figures from the paper:
```
jupyter notebook plot_results.ipynb
```

### Communication Experiments

Experiment demo:
```
python run-communication.py
```
for parameterization, refer to the comments in the file.

Reproduce the figures from the paper:
```
jupyter notebook plot_results_communication.ipynb
```


### Remarks

- Without a GPU the experiments take a lot of time.
- We provide some precomputed results in the ```res/``` folder.





