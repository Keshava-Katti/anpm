import torch
import torch.nn as nn
import numpy as np

'''
    Bandpass oscillator layer.
'''

class BandpassOscillatorLayer(nn.Module):
    def __init__(self, num_oscillators, dt=1e-3, freq_range=(1, 64), zeta=0.3, epsilon=1, coupling_type='linear'):
        super().__init__()
        self.num = num_oscillators
        self.dt = dt
        self.zeta = zeta
        self.epsilon = epsilon
        self.coupling_type = coupling_type

        self.omega = nn.Parameter(torch.tensor(2 * np.pi * np.linspace(freq_range[0], freq_range[1], num_oscillators), dtype=torch.float32), requires_grad=False)
        self.y1 = torch.zeros(num_oscillators)
        self.y2 = torch.zeros_like(self.y1)
        self.W = nn.Parameter(torch.randn(num_oscillators, num_oscillators))
        self.output_gain = nn.Parameter(torch.ones(num_oscillators))
        self.gate = nn.Linear(1, num_oscillators)

    def forward(self, input_drive):
        omega = self.omega
        dy1 = self.y2
        dy2 = -2 * self.zeta * omega * self.y2 - omega ** 2 * self.y1 + input_drive

        if self.coupling_type == 'linear':
            f_yj = self.y1
        elif self.coupling_type == 'phase':
            theta = torch.atan2(self.y2, self.y1)
            phase_diff = theta.unsqueeze(1) - theta.unsqueeze(0)
            f_yj = torch.sum(torch.cos(phase_diff), dim=1)
        elif self.coupling_type == 'tanh':
            f_yj = torch.tanh(self.y1)
        elif self.coupling_type == 'sigmoid':
            f_yj = torch.sigmoid(self.y1)
        else:
            raise ValueError(f"Unknown coupling type: {self.coupling_type}")
        
        gate_signal = torch.sigmoid(self.gate(input_drive.mean().unsqueeze(0)))
        gated_input = gate_signal * input_drive

        dy2 += (1 + gated_input) * (input_drive + self.epsilon * torch.matmul(self.W, f_yj))

        self.y1 = self.y1 + self.dt * dy1
        self.y2 = self.y2 + self.dt * dy2

        normalized_y1 = self.y1 / torch.sqrt(torch.mean(self.y1**2) + 1e-6)

        return self.output_gain * normalized_y1

    def reset_state(self):
        self.y1 = self.y1.detach()
        self.y2 = self.y2.detach()
        self.y1.zero_()
        self.y2.zero_()