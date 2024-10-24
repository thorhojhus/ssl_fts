from argparse import Namespace
import torch
from torch import nn
from torch.fft import rfft

class FITS(nn.Module):
    """Reimplementation of the FITS model with end upsampling."""

    def __init__(
        self,
        args: Namespace,
    ):
        super(FITS, self).__init__()

        self.cutoff_frequency = args.dominance_freq
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        self.upsample_rate = (args.seq_len + args.pred_len) / args.seq_len
        self.channels = args.channels
        
        # Add a small amount of extra upsampling at the ends
        self.end_upsample = int(0.1 * args.seq_len)
        self.total_len = args.seq_len + args.pred_len + 2 * self.end_upsample

        self.frequency_upsampler = (
            nn.Linear(
                in_features=args.dominance_freq,
                out_features=int(args.dominance_freq * self.total_len / args.seq_len),
                dtype=torch.cfloat,
                bias=True,
            )
            if not args.individual
            else nn.ModuleList(
                [
                    nn.Linear(
                        in_features=args.dominance_freq,
                        out_features=int(args.dominance_freq * self.total_len / args.seq_len),
                        dtype=torch.cfloat,
                        bias=True,
                    )
                    for _ in range(args.channels)
                ]
            )
        )

        self.individual = args.individual
        self.debug = args.debug
        if args.debug:
            self.debug_tensors = {}

    def channel_wise_frequency_upsampler(
        self, ts_frequency_data_filtered: torch.Tensor
    ) -> torch.Tensor:
        """Performs the complex valued layer frequency upsampling on a per-channel basis."""
        complex_valued_data = torch.zeros(
            [
                ts_frequency_data_filtered.size(0),
                int(self.cutoff_frequency * self.total_len / self.seq_len),
                ts_frequency_data_filtered.size(2),
            ],
            dtype=ts_frequency_data_filtered.dtype,
        ).to(ts_frequency_data_filtered.device)
        for i in range(self.channels):
            complex_valued_data[:, :, i] = self.frequency_upsampler[i](
                ts_frequency_data_filtered[:, :, i]
            )
        return complex_valued_data

    def forward(self, ts_data: torch.Tensor) -> torch.Tensor:
        # 1) Normalization of the input tensor:
        ts_mean, ts_var = (
            torch.mean(ts_data, dim=1, keepdim=True),
            torch.var(ts_data, dim=1, keepdim=True) + 1e-5,
        )
        normalized_ts_data = (ts_data - ts_mean) / torch.sqrt(ts_var)

        # 2) perform real fast fourier transform on the input tensor
        ts_frequency_data = rfft(input=normalized_ts_data, dim=1)

        # 3) perform a low pass filter to remove high frequency noise
        ts_frequency_data_filtered = ts_frequency_data[:, 0 : self.cutoff_frequency, :]

        # 4) Run the tensor through a complex valued linear layer
        if self.individual:
            complex_valued_data = self.channel_wise_frequency_upsampler(
                ts_frequency_data_filtered
            )
        else:
            complex_valued_data = self.frequency_upsampler(
                ts_frequency_data_filtered.permute(0, 2, 1)
            ).permute(0, 2, 1)

        # 5) obtain new frequencies from the output of the complex valued linear layer
        norm_spec_xy = torch.zeros(
            [
                complex_valued_data.size(0),
                int(self.total_len / 2 + 1),
                complex_valued_data.size(2),
            ],
            dtype=complex_valued_data.dtype,
        ).to(complex_valued_data.device)

        # 6) 0 pad the output tensor
        norm_spec_xy[:, 0 : complex_valued_data.size(1), :] = complex_valued_data

        # 7) perform inverse real fast fourier transform on the output tensor
        norm_xy = torch.fft.irfft(norm_spec_xy, dim=1)
        norm_xy_extra = norm_xy * self.total_len / self.seq_len
        
        # 8) Cut off the extra upsampled ends
        norm_xy = norm_xy_extra[:, self.end_upsample:-self.end_upsample, :]

        # 9) Reverse Normalization
        xy = norm_xy * torch.sqrt(ts_var) + ts_mean

        if self.debug:
            self.debug_tensors = {
                "normalized_ts_data": normalized_ts_data,
                "ts_frequency_data": ts_frequency_data,
                "ts_frequency_data_filtered": ts_frequency_data_filtered,
                "complex_valued_data": complex_valued_data,
                "norm_spec_xy": norm_spec_xy,
                "norm_xy_extra": norm_xy_extra,
                "norm_xy": norm_xy,
                "xy": xy,
            }

        return xy