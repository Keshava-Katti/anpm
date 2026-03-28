import numpy as np
from math import floor
from torch.utils.data import Dataset, DataLoader
import torch

'''
    Generates chaotic signal using Mackey-Glass equation.
'''

# Estimate of Mackey-Glass using RK4
def Df(x, n, beta):
	y = (beta * x) / (1 + x**n)
	return y

def Mackey_Glass(N, tau, n, gamma, beta, dt=1):
	x = np.zeros((N,))
	t = np.zeros((N,))
	
	x[0] = 1.2

	for k in range(N - 1):	
		t[k + 1] = t[k] + dt
		if k < tau:
			k1 = -gamma * x[k]
			k2 = -gamma * (x[k] + dt * k1 / 2) 
			k3 = -gamma * (x[k] + k2 * dt / 2) 
			k4 = -gamma * (x[k] + k3 * dt)
			x[k + 1] = x[k] + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6
		else:
			j = floor((t[k] - tau - t[0]) / dt + 1)
			k1 = Df(x[j], n, beta) - gamma * x[k]
			k2 = Df(x[j], n, beta) - gamma * (x[k] + dt * k1 / 2)
			k3 = Df(x[j], n, beta) - gamma * (x[k] + 2 * k2 * dt / 2) 
			k4 = Df(x[j], n, beta) - gamma * (x[k] + k3 * dt)
			x[k + 1] = x[k] + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6; 

	return t, x

class MG_Dataset(Dataset):
    def __init__(self, data, H, D, P, data_range=[201, 3200], train=True):
        super(MG_Dataset, self).__init__()
        self.data = data
        self.train = train
        self.data_range = data_range
        self.samples = [id for id in range(data_range[0], data_range[1])]

        self.H = H
        self.D = D 
        self.P = P

        self.nSamples = len(self.samples)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index < self.nSamples, 'index range error' 
        t = self.samples[index]

        features = []
        for i in range(self.D):
            features.append(self.data[t - self.P * i])
        target = self.data[t + self.H]

        # Convert to torch.float32 tensors
        features = torch.tensor(features, dtype=torch.float32)   # (D,)
        target = torch.tensor(target, dtype=torch.float32)       # scalar

        return np.array(features), target

def get_dataloader(data, H, D, P, train=False, batch_size=64):
    if train:
        data_range=[201, 3200]
        loader = DataLoader(
            MG_Dataset(data, H, D, P, data_range=data_range),
            batch_size=batch_size
        )
    else:
        data_range = [5001, 5500]
        loader = DataLoader(
            MG_Dataset(data, H, D, P, data_range=data_range, train=False),
            batch_size=batch_size
        )
    return loader