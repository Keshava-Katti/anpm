import torch
from torch import nn
import torch.nn.functional as F

'''
    UH/LI model. Basic modules and parameters adapted from:

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

    def forward(self, x, u):

        in_sum = self.linear(x)

        tau_mem = torch.abs(self.tau_mem)
        alpha = torch.exp(-1 * 1 / tau_mem)

        u = li_update(x=in_sum, u=u, alpha=alpha)
        return u

class HarmonicOscillatorCell(nn.Module):
    """
    Single-step harmonic oscillator cell:

        du/dt = -Omega * v + B * x_t
        dv/dt = u

    where Omega is diagonal (frequencies), B is a learnable projection,
    and x_t is the forcing input at time t (here coming from u_{l-1}(t)).

    Supports four discretization modes:
        - "euler" : forward Euler
        - "im"    : implicit (dissipative)
        - "imex"  : symplectic / IMEX
        - "rk4"   : classical Runge-Kutta (4th order)
    """

    def __init__(self, d_model, dt=0.1, mode: str="imex"):
        super().__init__()
        assert mode in ("euler", "im", "imex", "rk4")
        self.d_model = d_model
        self.dt = dt
        self.mode = mode

        self.omega_raw = nn.Parameter(torch.randn(d_model))

        self.B = nn.Linear(d_model, d_model, bias=False)

    def _get_omega(self, batch_size, device):
        omega = F.softplus(self.omega_raw)
        omega = omega.unsqueeze(0).expand(batch_size, -1)
        return omega.to(device)

    def _rhs(self, u, v, Bx, omega):
        du_dt = -omega * v + Bx
        dv_dt = u
        return du_dt, dv_dt

    def _step_euler(self, u_prev, v_prev, Bx, omega):
        dt = self.dt
        du_dt, dv_dt = self._rhs(u_prev, v_prev, Bx, omega)
        u_next = u_prev + dt * du_dt
        v_next = v_prev + dt * dv_dt
        return u_next, v_next

    def _step_im(self, u_prev, v_prev, Bx, omega):
        dt = self.dt
        S = 1.0 / (1.0 + (dt ** 2) * omega)

        u_next = S * (u_prev - dt * omega * v_prev + dt * Bx)
        v_next = S * (v_prev + dt * u_prev + (dt ** 2) * Bx)
        return u_next, v_next

    def _step_imex(self, u_prev, v_prev, Bx, omega):
        dt = self.dt
        u_next = u_prev + dt * (-omega * v_prev + Bx)
        v_next = v_prev + dt * u_next
        return u_next, v_next

    def _step_rk4(self, u_prev, v_prev, Bx, omega):
        dt = self.dt

        k1_u, k1_v = self._rhs(u_prev, v_prev, Bx, omega)
        k2_u, k2_v = self._rhs(
            u_prev + 0.5 * dt * k1_u,
            v_prev + 0.5 * dt * k1_v,
            Bx,
            omega,
        )
        k3_u, k3_v = self._rhs(
            u_prev + 0.5 * dt * k2_u,
            v_prev + 0.5 * dt * k2_v,
            Bx,
            omega,
        )
        k4_u, k4_v = self._rhs(
            u_prev + dt * k3_u,
            v_prev + dt * k3_v,
            Bx,
            omega,
        )

        u_next = u_prev + (dt / 6.0) * (k1_u + 2 * k2_u + 2 * k3_u + k4_u)
        v_next = v_prev + (dt / 6.0) * (k1_v + 2 * k2_v + 2 * k3_v + k4_v)
        return u_next, v_next

    def forward(self, x_seq):
        B, T, D = x_seq.shape
        device = x_seq.device
        omega = self._get_omega(B, device)

        u_prev = torch.zeros(B, D, device=device)
        v_prev = torch.zeros(B, D, device=device)

        y_list = []
        u_list = []
        v_list = []

        for t in range(T):
            x_t = x_seq[:, t, :]          
            Bx = self.B(x_t)              

            if self.mode == "euler":
                u_next, v_next = self._step_euler(u_prev, v_prev, Bx, omega)
            elif self.mode == "im":
                u_next, v_next = self._step_im(u_prev, v_prev, Bx, omega)
            elif self.mode == "imex":
                u_next, v_next = self._step_imex(u_prev, v_prev, Bx, omega)
            else:  # "rk4"
                u_next, v_next = self._step_rk4(u_prev, v_prev, Bx, omega)

            y_t = v_next

            y_list.append(y_t)
            u_list.append(u_next)
            v_list.append(v_next)

            u_prev, v_prev = u_next, v_next

        y_seq = torch.stack(y_list, dim=1)
        u_seq = torch.stack(u_list, dim=1)
        v_seq = torch.stack(v_list, dim=1)
        return y_seq, u_seq, v_seq

class LISequenceBlock(nn.Module):
    """
    Runs LICell over a full sequence:
    """
    def __init__(
        self,
        input_size: int,
        layer_size: int,
        tau_mem: float,
        adaptive_tau_mem: bool = True,
        adaptive_tau_mem_mean: float = 10.0,
        adaptive_tau_mem_std: float = 1.0,
        bias: bool = False,
    ):
        super().__init__()
        self.cell = LICell(
            input_size=input_size,
            layer_size=layer_size,
            tau_mem=tau_mem,
            adaptive_tau_mem=adaptive_tau_mem,
            adaptive_tau_mem_mean=adaptive_tau_mem_mean,
            adaptive_tau_mem_std=adaptive_tau_mem_std,
            bias=bias,
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        B, T, _ = x_seq.shape
        device = x_seq.device

        u = torch.zeros(B, self.cell.layer_size, device=device)
        outs = []

        for t in range(T):
            u = self.cell(x_seq[:, t, :], u) 
            outs.append(u)

        u_seq = torch.stack(outs, dim=1) 
        return u_seq

class HarmonicBlockWithLI(nn.Module):
    def __init__(
        self,
        d_model: int,
        dt: float = 0.1,
        mode: str = "imex",
        tau_mem: float = 10.0,
        adaptive_tau_mem: bool = True,
    ):
        super().__init__()
        self.d_model = d_model

        self.osc = HarmonicOscillatorCell(
            d_model=d_model,
            dt=dt,
            mode=mode,
        )

        self.li_seq = LISequenceBlock(
            input_size=d_model,
            layer_size=d_model,
            tau_mem=tau_mem,
            adaptive_tau_mem=adaptive_tau_mem,
        )

        self.C = nn.Linear(d_model, d_model, bias=False)
        self.D = nn.Linear(d_model, d_model, bias=False)

        self.glu_proj = nn.Linear(d_model, 2 * d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, u_prev: torch.Tensor):
        osc_y, _, _ = self.osc(u_prev)

        li_y = self.li_seq(osc_y)

        x_l = self.C(li_y) + self.D(u_prev)
        x_l = F.gelu(x_l)

        h = self.glu_proj(x_l)            
        a, b = h.chunk(2, dim=-1)
        glu_out = a * torch.sigmoid(b)

        u_l = glu_out + u_prev
        u_l = self.norm(u_l)

        y_l = li_y
        return u_l, y_l

class HarmonicNDSWithLI(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int,
        output_dim: int,
        num_layers: int = 4,
        dt: float = 0.1,
        modes=None,
        tau_mem: float = 10.0,
        adaptive_tau_mem: bool = True,
    ):
        super().__init__()
        self.encoder = nn.Linear(input_dim, d_model)

        if modes is None:
            modes = ["imex"] * num_layers
        if isinstance(modes, str):
            modes = [modes] * num_layers

        self.blocks = nn.ModuleList(
            [
                HarmonicBlockWithLI(
                    d_model=d_model,
                    dt=dt,
                    mode=modes[l],
                    tau_mem=tau_mem,
                    adaptive_tau_mem=adaptive_tau_mem,
                )
                for l in range(num_layers)
            ]
        )

        self.decoder = nn.Linear(d_model, output_dim)

    def forward(self, x):
        u_l = self.encoder(x)
        y_l = None
        for block in self.blocks:
            u_l, y_l = block(u_l)
        o = self.decoder(y_l) 
        return o, y_l, u_l