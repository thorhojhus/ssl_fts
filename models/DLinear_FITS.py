import torch
import torch.nn as nn
from torch.fft import rfft, irfft
from argparse import Namespace

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        return x.permute(0, 2, 1)

class SeriesDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomp, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class FITS(nn.Module):
    def __init__(self, args):
        super(FITS, self).__init__()
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.cutoff_frequency = args.dominance_freq
        self.upsample_rate = (args.seq_len + args.pred_len) / args.seq_len
        self.enc_in = args.channels

        self.frequency_upsampler = (
            nn.Linear(
                in_features=args.dominance_freq,
                out_features=int(args.dominance_freq * self.upsample_rate),
                dtype=torch.cfloat,
                bias=True,
            )
            if not args.individual
            else nn.ModuleList(
                [
                    nn.Linear(
                        in_features=args.dominance_freq,
                        out_features=int(args.dominance_freq * self.upsample_rate),
                        dtype=torch.cfloat,
                        bias=True,
                    )
                    for _ in range(self.enc_in)
                ]
            )
        )

        self.individual = args.individual

    def channel_wise_frequency_upsampler(self, ts_frequency_data_filtered):
        complex_valued_data = torch.zeros(
            [
                ts_frequency_data_filtered.size(0),
                int(self.cutoff_frequency * self.upsample_rate),
                ts_frequency_data_filtered.size(2),
            ],
            dtype=ts_frequency_data_filtered.dtype,
        ).to(ts_frequency_data_filtered.device)
        for i in range(self.enc_in):
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

        return xy

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        # Decomposition
        kernel_size = 25
        self.decomposition = SeriesDecomp(kernel_size)

        # FITS component
        fits_args = Namespace(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            dominance_freq=49,
            channels=configs.enc_in,
            individual=configs.individual,
        )
        self.fits = FITS(fits_args)
        print("FITS cutoff frequency:", fits_args.dominance_freq)

        # Linear layers
        self.individual = configs.individual
        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for _ in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: [Batch, Input length, Channel]
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        
        # Process seasonal and trend components
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len], 
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len], 
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        # Process residual with FITS
        x_last = x[:, -self.pred_len:, :]  # Use only the last pred_len points of x
        seasonal_trend = (seasonal_output + trend_output).permute(0, 2, 1)
        # Pad or trim x to match pred_len
        if self.pred_len > self.seq_len:
            x_padded = torch.cat([x, x[:, -1:, :].repeat(1, self.pred_len - self.seq_len, 1)], dim=1)
        else:
            x_padded = x[:, -self.pred_len:, :]
        
        residual = x_padded - seasonal_trend
        fits_output = self.fits(residual)

        # Combine all components
        x = seasonal_output + trend_output + fits_output[:, :self.pred_len, :].permute(0, 2, 1)
        return x.permute(0, 2, 1)  # [Batch, Output length, Channel]