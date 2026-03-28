from scipy.signal import chirp, square, sawtooth
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader

'''
    Generates periodic and quasi-periodic signals.
'''

def get_signal(signal_type, t, dt, noise=False):
    T = int(len(t) * dt)
    if signal_type == 'sine':
        freq = random.uniform(1, 5)
        signal = np.sin(2 * np.pi * freq * t)

    elif signal_type == 'chirp':
        f0, f1 = random.uniform(0.5, 1), random.uniform(5, 6)
        signal = chirp(t, f0=f0, f1=f1, t1=T, method='linear')

    elif signal_type == 'square':
        freq = random.uniform(1, 5)
        signal = square(2 * np.pi * freq * t)

    elif signal_type == 'sawtooth':
        freq = random.uniform(1, 5)
        signal = sawtooth(2 * np.pi * freq * t)

    elif signal_type == 'am_sine':
        carrier = random.uniform(2, 6)
        mod = random.uniform(0.3, 1.0)
        temp = (1 + mod * np.sin(0.5 * np.pi * t)) * np.sin(2 * np.pi * carrier * t)
        signal = temp / np.max(temp)

    elif signal_type == 'composite':
        f1, f2, f3 = random.uniform(1, 3), random.uniform(1, 3), random.uniform(1, 3)
        n = len(t)
        s1 = np.sin(2 * np.pi * f1 * t[:n//3])
        s2 = square(2 * np.pi * f2 * t[n//3:2*n//3])
        s3 = sawtooth(2 * np.pi * f3 * t[2*n//3:])
        signal = np.concatenate([s1, s2, s3])

    elif signal_type == 'drift':
        a = random.uniform(0.3, 1.0)
        f_mod = random.uniform(0.05, 0.2)
        f_t = a * np.sin(f_mod * np.pi * t)
        signal = np.sin(2 * np.pi * f_t * t)

    elif signal_type == 'gaussian_pulse':
        num_pulses = random.randint(2, 5)
        signal = np.zeros_like(t)
        for _ in range(num_pulses):
            A = random.uniform(0.5, 1.0)
            mu = random.uniform(0, T)
            sigma = random.uniform(0.1, 0.5)
            signal += A * np.exp(-((t - mu) ** 2) / (2 * sigma ** 2))

    elif signal_type == 'sigmoid_step':
        t0 = random.uniform(2, 5)
        k = random.uniform(5, 20)
        signal = 1 / (1 + np.exp(-k * (t - t0)))
        
    elif signal_type == 'env_sine':
        carrier_freq = random.uniform(2, 5)
        env = np.exp(-((t - T / 2) ** 2) / (2 * (T / 5) ** 2))
        signal = env * np.sin(2 * np.pi * carrier_freq * t)

    elif signal_type == 'freq_drift_sine':
        base_freq = random.uniform(1, 3)
        drift_rate = random.uniform(0.1, 0.5)
        f_t = base_freq + drift_rate * np.sin(0.2 * np.pi * t)
        signal = np.sin(2 * np.pi * f_t * t)

    if noise:
        signal += np.random.normal(scale=0.05, size=signal.shape)

    return signal

class PeriodicDataset(Dataset):
    def __init__(self, data, H, D, P, data_range, train=True):
        super(PeriodicDataset, self).__init__()
        self.data = data
        self.train = train
        self.data_range = [data_range]
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

        features = torch.tensor(features, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return np.array(features), target

def get_periodic_dataloader(data, H, D, P, train=False, batch_size=64):
    if train:
        data_range=[P * D, len(data) // 2 - H - 1]
        loader = DataLoader(
            PeriodicDataset(data, H, D, P, data_range=data_range),
            batch_size=batch_size
        )
    else:
        data_range = [len(data) // 2 + P * D, len(data) - H - 1]
        loader = DataLoader(
            PeriodicDataset(data, H, D, P, data_range=data_range, train=False),
            batch_size=batch_size
        )
    return loader