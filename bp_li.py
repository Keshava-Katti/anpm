import torch

'''
    BP/LI model. Basic modules and parameters adapted from:

    @inproceedings{higuchi2024understanding,
        title        = {Understanding the Convergence in Balanced Resonate-and-Fire Neurons},
        author       = {Higuchi, Saya and and Boht{\'e}, Sander and Otte, Sebastian},
        year         = 2024,
        booktitle    = {Austrian Symposium on AI, Robotics, and Vision (AIRoV)},
        publisher    = {innsbruck university press},
        series       = {Proceedings of Austrian Symposium on AI, Robotics, and Vision 2024},
        volume       = 1,
        pages        = {437--445},
        pdf = 	 {https://ulb-dok.uibk.ac.at/ulbtirolfodok/download/pdf/12691798},
    } 
'''

DEFAULT_LI_TAU_M = 20.
DEFAULT_LI_ADAPTIVE_TAU_M_MEAN = 20.
DEFAULT_LI_ADAPTIVE_TAU_M_STD = 5.

DEFAULT_LIF_TAU_M = 20.
DEFAULT_LIF_ADAPTIVE_TAU_M_MEAN = 20.
DEFAULT_LIF_ADAPTIVE_TAU_M_STD = 5.

DEFAULT_MASK_PROB = 0

TRAIN_B_offset = True
DEFAULT_RF_B_offset = 1.

DEFAULT_RF_ADAPTIVE_B_offset_a = 1
DEFAULT_RF_ADAPTIVE_B_offset_b = 6

TRAIN_OMEGA = True
DEFAULT_RF_OMEGA = 10.

DEFAULT_RF_ADAPTIVE_OMEGA_a = 10
DEFAULT_RF_ADAPTIVE_OMEGA_b = 50

DEFAULT_RF_THETA = 1

TRAIN_ZETA = False
DEFAULT_RF_ZETA = .00

DEFAULT_RF_ADAPTIVE_ZETA_a = 0
DEFAULT_RF_ADAPTIVE_ZETA_b = 0

TRAIN_DT = False
DEFAULT_DT = 0.01
DEFAULT_RF_ADAPTIVE_DT = 0.01

def li_update(x, u, alpha):
    u = u.mul(alpha) + x.mul(1.0 - alpha)
    return u

class LICell(torch.nn.Module):
    def __init__(
            self,
            input_size,
            layer_size,
            tau_mem=DEFAULT_LI_TAU_M,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=DEFAULT_LI_ADAPTIVE_TAU_M_MEAN,
            adaptive_tau_mem_std=DEFAULT_LI_ADAPTIVE_TAU_M_STD,
            bias=False,
    ):
        super(LICell, self).__init__()

        self.input_size = input_size
        self.layer_size = layer_size

        self.bias = bias

        self.linear = torch.nn.Linear(
            in_features=input_size,
            out_features=layer_size,
            bias=bias
        )

        torch.nn.init.xavier_uniform_(self.linear.weight)

        if bias:
            torch.nn.init.constant_(self.linear.bias, 0)

        self.adaptive_tau_mem = adaptive_tau_mem
        self.adaptive_tau_mem_mean = adaptive_tau_mem_mean
        self.adaptive_tau_mem_std = adaptive_tau_mem_std

        tau_mem = tau_mem * torch.ones(layer_size)

        if adaptive_tau_mem:
            self.tau_mem = torch.nn.Parameter(tau_mem)
            torch.nn.init.normal_(self.tau_mem, mean=adaptive_tau_mem_mean, std=adaptive_tau_mem_std)
        else:
            self.register_buffer("tau_mem", tau_mem)

    def forward(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:

        in_sum = self.linear(x)

        tau_mem = torch.abs(self.tau_mem)
        alpha = torch.exp(-1 * 1 / tau_mem)

        u = li_update(x=in_sum, u=u, alpha=alpha)
        return u

def hr_update(x, u, v, b, omega, dt=DEFAULT_DT):
    v = v + u.mul(dt)
    u = u + x.mul(dt) - b.mul(u).mul(2 * dt) - torch.square(omega).mul(v).mul(dt)
    return u, v

class HRCell(torch.nn.Module):
    def __init__(
            self,
            input_size,
            layer_size,
            gain=True,
            normalize=True,
            b_offset=DEFAULT_RF_B_offset,
            adaptive_b_offset=TRAIN_B_offset,
            adaptive_b_offset_a=DEFAULT_RF_ADAPTIVE_B_offset_a,
            adaptive_b_offset_b=DEFAULT_RF_ADAPTIVE_B_offset_b,
            omega=DEFAULT_RF_OMEGA,
            adaptive_omega=TRAIN_OMEGA,
            adaptive_omega_a=DEFAULT_RF_ADAPTIVE_OMEGA_a,
            adaptive_omega_b=DEFAULT_RF_ADAPTIVE_OMEGA_b,
            dt=DEFAULT_DT,
            bias=False
    ):
        super(HRCell, self).__init__()

        self.input_size = input_size
        self.layer_size = layer_size

        if gain:
            self.gain = torch.nn.Parameter(torch.ones(layer_size))
        else:
            self.gain = torch.ones(layer_size)

        self.linear = torch.nn.Linear(
            in_features=input_size,
            out_features=layer_size,
            bias=bias
        )

        torch.nn.init.xavier_uniform_(self.linear.weight)

        self.normalize = normalize
        self.adaptive_omega = adaptive_omega
        self.adaptive_omega_a = adaptive_omega_a
        self.adaptive_omega_b = adaptive_omega_b

        omega = omega * torch.ones(layer_size)

        if adaptive_omega:
            self.omega = torch.nn.Parameter(omega)
            torch.nn.init.uniform_(self.omega, adaptive_omega_a, adaptive_omega_b)
        else:
            self.register_buffer('omega', omega)


        self.adaptive_b_offset = adaptive_b_offset
        self.adaptive_b_a = adaptive_b_offset_a
        self.adaptive_b_b = adaptive_b_offset_b

        b_offset = b_offset * torch.ones(layer_size)

        if adaptive_b_offset:
            self.b_offset = torch.nn.Parameter(b_offset)
            torch.nn.init.uniform_(self.b_offset, adaptive_b_offset_a, adaptive_b_offset_b)
        else:
            self.register_buffer('b_offset', b_offset)

        self.dt = dt

    def forward(self, x, state):
        u, v = state

        in_sum = self.linear(x)

        omega = torch.abs(self.omega)

        b_offset = torch.abs(self.b_offset)

        b = omega.square().mul(0.005) + b_offset

        u, v = hr_update(
            x=in_sum,
            u=u,
            v=v,
            b=b,
            omega=omega,
            dt=self.dt,
        )

        if self.normalize:
            norm = torch.norm(u, dim=1, keepdim=True).clamp(min=1e-6)
            u = u / norm

        u= u * self.gain

        return u, v

class SimpleHarmonicRNN(torch.nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            output_size,
            adaptive_omega_a=DEFAULT_RF_ADAPTIVE_OMEGA_a,
            adaptive_omega_b=DEFAULT_RF_ADAPTIVE_OMEGA_b,
            adaptive_b_offset_a=DEFAULT_RF_ADAPTIVE_B_offset_a,
            adaptive_b_offset_b=DEFAULT_RF_ADAPTIVE_B_offset_b,
            out_adaptive_tau_mem_mean=DEFAULT_LIF_ADAPTIVE_TAU_M_MEAN,
            out_adaptive_tau_mem_std=DEFAULT_LI_ADAPTIVE_TAU_M_STD,
            hidden_bias=False,
            output_bias=False
    ):
        super(SimpleHarmonicRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.hidden = HRCell(
            input_size=input_size + hidden_size, 
            layer_size=hidden_size,
            gain=True,
            normalize=False,
            adaptive_omega=True,
            adaptive_omega_a=adaptive_omega_a,
            adaptive_omega_b=adaptive_omega_b,
            adaptive_b_offset=True,
            adaptive_b_offset_a=adaptive_b_offset_a,
            adaptive_b_offset_b=adaptive_b_offset_b,
            bias=hidden_bias
        )

        self.out = LICell(
            input_size=hidden_size,
            layer_size=hidden_size,
            adaptive_tau_mem=True,
            adaptive_tau_mem_mean=out_adaptive_tau_mem_mean,
            adaptive_tau_mem_std=out_adaptive_tau_mem_std,
            bias=output_bias
        )

        self.linear = torch.nn.Linear(hidden_size, output_size)

    def forward(
            self,
            x: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor], float]:

        sequence_length = x.shape[0]
        batch_size = x.shape[1]

        outputs_u = list()

        hidden_u = torch.zeros((batch_size, self.hidden_size)).to(x.device)
        hidden_v = torch.zeros((batch_size, self.hidden_size)).to(x.device)

        out_u = torch.zeros((batch_size, self.hidden_size)).to(x.device)

        for t in range(sequence_length):
            input_t = x[t]

            hidden = hidden_u, hidden_v

            hidden_u, hidden_v = self.hidden(
                torch.cat((input_t, hidden_u), dim=1),
                hidden
            )

            out_u = self.out(hidden_u, out_u)
            out = self.linear(out_u)

            outputs_u.append(out) 

        outputs = torch.sum(torch.stack(outputs_u), dim=0)

        return outputs

