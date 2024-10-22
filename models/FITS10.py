import torch
import torch.nn as nn
from torch.fft import rfft, irfft
from argparse import Namespace

class FITS(nn.Module):
    def __init__(self, configs):
        super(FITS, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in
        self.cutoff_frequency = configs.dominance_freq
        self.upsample_rate = (configs.seq_len + configs.pred_len) / configs.seq_len

        self.frequency_upsampler = (
            nn.Linear(
                in_features=configs.dominance_freq,
                out_features=int(configs.dominance_freq * self.upsample_rate),
                dtype=torch.cfloat,
                bias=True,
            )
            if not configs.individual
            else nn.ModuleList(
                [
                    nn.Linear(
                        in_features=configs.dominance_freq,
                        out_features=int(configs.dominance_freq * self.upsample_rate),
                        dtype=torch.cfloat,
                        bias=True,
                    )
                    for _ in range(configs.enc_in)
                ]
            )
        )

        self.individual = configs.individual

    def channel_wise_frequency_upsampler(self, ts_frequency_data_filtered):
        complex_valued_data = torch.zeros(
            [
                ts_frequency_data_filtered.size(0),
                int(self.cutoff_frequency * self.upsample_rate),
                ts_frequency_data_filtered.size(2),
            ],
            dtype=ts_frequency_data_filtered.dtype,
        ).to(ts_frequency_data_filtered.device)
        for i in range(self.channels):
            complex_valued_data[:, :, i] = self.frequency_upsampler[i](
                ts_frequency_data_filtered[:, :, i]
            )
        return complex_valued_data

    def forward(self, x):
        # Normalization
        ts_mean, ts_var = torch.mean(x, dim=1, keepdim=True), torch.var(x, dim=1, keepdim=True) + 1e-5
        normalized_ts_data = (x - ts_mean) / torch.sqrt(ts_var)

        # FFT
        ts_frequency_data = rfft(input=normalized_ts_data, dim=1)

        # Low pass filter
        ts_frequency_data_filtered = ts_frequency_data[:, 0:self.cutoff_frequency, :]

        # Complex valued linear layer
        if self.individual:
            complex_valued_data = self.channel_wise_frequency_upsampler(ts_frequency_data_filtered)
        else:
            complex_valued_data = self.frequency_upsampler(ts_frequency_data_filtered.permute(0, 2, 1)).permute(0, 2, 1)

        # Zero padding
        norm_spec_xy = torch.zeros(
            [
                complex_valued_data.size(0),
                int((self.seq_len + self.pred_len) / 2 + 1),
                complex_valued_data.size(2),
            ],
            dtype=complex_valued_data.dtype,
        ).to(complex_valued_data.device)
        norm_spec_xy[:, 0:complex_valued_data.size(1), :] = complex_valued_data

        # Inverse FFT
        norm_xy = irfft(norm_spec_xy, dim=1)
        norm_xy = norm_xy * self.upsample_rate

        # Reverse normalization
        xy = norm_xy * torch.sqrt(ts_var) + ts_mean

        return xy[:, -self.pred_len:, :]

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        # FITS component
        fits_args = Namespace(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            dominance_freq=10,
            enc_in=configs.enc_in,
            individual=configs.individual,
        )
        self.fits = FITS(fits_args)
        print("FITS cutoff frequency:", fits_args.dominance_freq)

    def forward(self, x):
        return self.fits(x)