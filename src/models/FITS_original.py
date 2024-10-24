import torch
import torch.nn as nn


# Copy of the current FITS model from the official implementation (visitied last at 13-03-2024)
# from: https://github.com/VEWOXIC/FITS/blob/main/models/FITS.py


# Changes made:
# - added debug tensor output if debug mode is on
# - removed bias from linear layers for debugging easier
class Model(nn.Module):

    # FITS: Frequency Interpolation Time Series Forecasting

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.individual = configs.individual
        self.channels = configs.channels

        self.dominance_freq = configs.dominance_freq  # 720/24
        self.length_ratio = (self.seq_len + self.pred_len) / self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(
                    nn.Linear(
                        self.dominance_freq,
                        int(self.dominance_freq * self.length_ratio),
                        bias=False,
                    ).to(torch.cfloat)
                )

        else:
            self.freq_upsampler = nn.Linear(
                self.dominance_freq,
                int(self.dominance_freq * self.length_ratio),
                bias=False,
            ).to(
                torch.cfloat
            )  # complex layer for frequency upcampling]

        # added this ourselves ---
        self.debug = configs.debug
        if configs.debug:
            self.debug_tensors = {}
        # --- added this ourselves

    def forward(self, x):
        # RIN
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        # print(x_var)
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)
        low_specx[:, self.dominance_freq :] = 0  # LPF
        low_specx = low_specx[:, 0 : self.dominance_freq, :]  # LPF

        if self.individual:
            low_specxy_ = torch.zeros(
                [
                    low_specx.size(0),
                    int(self.dominance_freq * self.length_ratio),
                    low_specx.size(2),
                ],
                dtype=low_specx.dtype,
            ).to(low_specx.device)
            for i in range(self.channels):
                low_specxy_[:, :, i] = self.freq_upsampler[i](
                    low_specx[:, :, i].permute(0, 1)
                ).permute(0, 1)
        else:
            low_specxy_ = self.freq_upsampler(low_specx.permute(0, 2, 1)).permute(
                0, 2, 1
            )
        # print(low_specxy_)
        low_specxy = torch.zeros(
            [
                low_specxy_.size(0),
                int((self.seq_len + self.pred_len) / 2 + 1),
                low_specxy_.size(2),
            ],
            dtype=low_specxy_.dtype,
        ).to(low_specxy_.device)
        low_specxy[:, 0 : low_specxy_.size(1), :] = low_specxy_  # zero padding
        low_xy = torch.fft.irfft(low_specxy, dim=1)
        low_xy = low_xy * self.length_ratio  # energy compemsation for the length change
        # dom_x=x-low_x

        # dom_xy=self.Dlinear(dom_x)
        # xy=(low_xy+dom_xy) * torch.sqrt(x_var) +x_mean # REVERSE RIN
        xy = (low_xy) * torch.sqrt(x_var) + x_mean

        # added this ourselves ---
        if self.debug:
            self.debug_tensors = {
                "x": x,
                "low_specx": low_specx,
                "low_specxy_": low_specxy_,
                "low_specxy": low_specxy,
                "low_xy": low_xy,
                "xy": xy,
            }
        # --- added this ourselves

        return xy  # , low_xy* torch.sqrt(x_var)
