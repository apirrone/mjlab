import pickle
from typing import List, Tuple, Union

import torch
import time


def torch_polyval_desc(coeffs_desc: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    # coeffs in DESCENDING power order
    y = torch.zeros_like(x, dtype=coeffs_desc.dtype)
    for a in coeffs_desc:
        y = y * x + a
    return y


class PolyReferenceMotion:
    def __init__(self, polynomial_coefficients: str, device: Union[str, torch.device] = "cpu"):
        self.device = torch.device(device)
        data = pickle.load(open(polynomial_coefficients, "rb"))
        self.dx_range = [0.0, 0.0]
        self.dy_range = [0.0, 0.0]
        self.dtheta_range = [0.0, 0.0]
        self.dxs: List[float] = []
        self.dys: List[float] = []
        self.dthetas: List[float] = []
        self.data_array = []
        self.period = None
        self.fps = None
        self.frame_offsets = None
        self.startend_double_support_ratio = None
        self.start_offset = None
        self.nb_steps_in_period: int = 0

        self.process(data)

        # cached tensors for fast distance/argmin
        self.dxs_t = torch.tensor(self.dxs, dtype=torch.float32, device=self.device)
        self.dys_t = torch.tensor(self.dys, dtype=torch.float32, device=self.device)
        self.dthetas_t = torch.tensor(self.dthetas, dtype=torch.float32, device=self.device)

    def process(self, data):
        print("[Poly ref data] Processing ...")
        _data = {}
        for name in data.keys():
            split = name.split("_")
            dx = float(split[0])
            dy = float(split[1])
            dtheta = float(split[2])

            if self.period is None:
                self.period = float(data[name]["period"])
                self.fps = float(data[name]["fps"])
                self.frame_offsets = data[name]["frame_offsets"]
                self.startend_double_support_ratio = float(data[name]["startend_double_support_ratio"])
                self.start_offset = int(self.startend_double_support_ratio * self.fps)
                self.nb_steps_in_period = int(self.period * self.fps)

            if dx not in self.dxs:
                self.dxs.append(dx)
            if dy not in self.dys:
                self.dys.append(dy)
            if dtheta not in self.dthetas:
                self.dthetas.append(dtheta)

            self.dx_range = [min(dx, self.dx_range[0]), max(dx, self.dx_range[1])]
            self.dy_range = [min(dy, self.dy_range[0]), max(dy, self.dy_range[1])]
            self.dtheta_range = [
                min(dtheta, self.dtheta_range[0]),
                max(dtheta, self.dtheta_range[1]),
            ]

            if dx not in _data:
                _data[dx] = {}
            if dy not in _data[dx]:
                _data[dx][dy] = {}

            # convert coefficients dict -> list of torch tensors (DESC order for Horner)
            coeffs_dict = data[name]["coefficients"]
            coeffs_list = []
            for _, v in coeffs_dict.items():
                a = torch.as_tensor(v, dtype=torch.float32, device=self.device)  # ASC
                coeffs_list.append(a.flip(0))  # DESC
            _data[dx][dy][dtheta] = coeffs_list

        self.dxs = sorted(self.dxs)
        self.dys = sorted(self.dys)
        self.dthetas = sorted(self.dthetas)

        nb_dx = len(self.dxs)
        nb_dy = len(self.dys)
        nb_dtheta = len(self.dthetas)

        self.data_array = nb_dx * [None]
        for x, dx in enumerate(self.dxs):
            self.data_array[x] = nb_dy * [None]
            for y, dy in enumerate(self.dys):
                self.data_array[x][y] = nb_dtheta * [None]
                for th, dtheta in enumerate(self.dthetas):
                    self.data_array[x][y][th] = _data[dx][dy][dtheta]

        print("[Poly ref data] Done processing")

    @torch.no_grad()
    def vel_to_index(self, dx, dy, dtheta) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # accept scalars or 1D tensors
        dx_t = torch.as_tensor(dx, dtype=torch.float32, device=self.device)
        dy_t = torch.as_tensor(dy, dtype=torch.float32, device=self.device)
        dt_t = torch.as_tensor(dtheta, dtype=torch.float32, device=self.device)

        if dx_t.ndim == 0: dx_t = dx_t[None]
        if dy_t.ndim == 0: dy_t = dy_t[None]
        if dt_t.ndim == 0: dt_t = dt_t[None]

        dx_t = dx_t.clamp(self.dx_range[0], self.dx_range[1])
        dy_t = dy_t.clamp(self.dy_range[0], self.dy_range[1])
        dt_t = dt_t.clamp(self.dtheta_range[0], self.dtheta_range[1])

        # vectorized nearest neighbor via argmin of |grid - q|
        ix = torch.argmin((self.dxs_t[None, :] - dx_t[:, None]).abs(), dim=1)
        iy = torch.argmin((self.dys_t[None, :] - dy_t[:, None]).abs(), dim=1)
        it = torch.argmin((self.dthetas_t[None, :] - dt_t[:, None]).abs(), dim=1)
        return ix.to(torch.long), iy.to(torch.long), it.to(torch.long)

    @torch.no_grad()
    def sample_polynomial(self, t: torch.Tensor, coeffs: List[torch.Tensor]):
        # coeffs: list of 1D tensors in DESC order
        outs = []
        for c_desc in coeffs:
            outs.append(torch_polyval_desc(c_desc, t))
        return outs  # list of [B] tensors

    @torch.no_grad()
    def get_reference_motion(self, dx, dy, dtheta, i):
        ix, iy, itheta = self.vel_to_index(dx, dy, dtheta)  # [B]

        B = ix.shape[0]

        i_t = torch.as_tensor(i, device=self.device)
        if i_t.ndim == 0:
            i_t = i_t.repeat(B)
        i_t = i_t.to(torch.long)

        nb = int(self.nb_steps_in_period)
        t = (i_t.remainder(nb).to(torch.float32) / float(nb)).clamp(0.0, 1.0)  # [B]

        # evaluate per item, preserving original structure
        # returns Python list of tensors per output dim
        ret_per_dim: List[List[torch.Tensor]] = []
        # figure number of dims from first cell
        coeffs0 = self.data_array[ix[0].item()][iy[0].item()][itheta[0].item()]
        D = len(coeffs0)
        for _ in range(D):
            ret_per_dim.append([])

        
        for b in range(B):
            coeffs = self.data_array[ix[b].item()][iy[b].item()][itheta[b].item()]
            vals = self.sample_polynomial(t[b:b+1], coeffs)  # list of [1]
            for d in range(D):
                ret_per_dim[d].append(vals[d])  # [1]

        # stack back to [B] per dim, then return list (to mirror original)
        ret = [torch.cat(ret_per_dim[d], dim=0) for d in range(D)]  # list of [B]
        return ret
