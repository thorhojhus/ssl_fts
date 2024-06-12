from argparse import Namespace
import torch
from torch import nn
from torch.fft import rfft
from torch.nn import functional as F

# An implementation of the FITS model as described in https://arxiv.org/abs/2307.03756 (FITS: Frequency Interpolation Time Series Forecasting)
# with better annotation and more clear model structure - the original code for the model can be found here: https://github.com/VEWOXIC/FITS

class FITS(nn.Module):
    """Reimplementation of the FITS model.

    This model contains an annotated and more cleany written model, which is true to the model described in the paper* and the original code.
    *(RIN was not performed in the code, but instance wise normalization was performed instead).
    """

    def __init__(
        self,
        args: Namespace,
    ):
        super(FITS, self).__init__()

        self.cutoff_frequency = args.dominance_freq
        self.seq_len = args.seq_len
        self.pred_len = args.pred_len
        if args.upsample_rate == 0:
            self.upsample_rate = (args.seq_len + args.pred_len) / args.seq_len
        else:
            self.upsample_rate = args.upsample_rate

        self.channels = args.channels
        self.num_layers = args.num_layers

        layers = []
        in_features = args.dominance_freq
        for i in range(self.num_layers):
            out_features = (
                int(args.dominance_freq * self.upsample_rate)
                if self.num_layers == 1 or i == self.num_layers - 1
                else args.num_hidden
            )
            layers.append(
                nn.Linear(
                    in_features=in_features,
                    out_features=out_features,
                    dtype=torch.float,
                    bias=True,
                )
            )
            if i != self.num_layers - 1:
                layers.append(nn.Dropout())
                layers.append(nn.ReLU())
            in_features = out_features

        self.frequency_upsampler = (
            nn.Sequential(*layers)
            if not args.individual
            else nn.ModuleList([nn.Sequential(*layers) for _ in range(args.channels)])
        )

        self.individual = args.individual
        self.debug = args.debug
        if args.debug:
            self.debug_tensors = {}

    def channel_wise_frequency_upsampler(
        self, ts_frequency_data_filtered: torch.Tensor
    ) -> torch.Tensor:
        """Performs the real valued layer frequency upsampling on a per-channel basis."""
        real_valued_data = torch.zeros(
            [
                ts_frequency_data_filtered.size(0),
                int(self.cutoff_frequency * self.upsample_rate),
                ts_frequency_data_filtered.size(2),
            ],
            dtype=ts_frequency_data_filtered.dtype,
        ).to(ts_frequency_data_filtered.device)

        for i in range(self.channels):
            real_valued_data[:, :, i] = self.frequency_upsampler[i](
                ts_frequency_data_filtered[:, :, i]
            )
        return real_valued_data

    def forward(self, ts_data: torch.Tensor) -> torch.Tensor:
        # 1) Normalization of the input tensor:
        ts_mean, ts_var = (
            torch.mean(ts_data, dim=1, keepdim=True),
            torch.var(ts_data, dim=1, keepdim=True) + 1e-5,
        )
        normalized_ts_data = (ts_data - ts_mean) / torch.sqrt(ts_var)

        # 2) perform real fast fourier transform on the input tensor
        ts_frequency_data = rfft(input=normalized_ts_data, dim=1)

        # 3) perform a low pass filter to remove high frequency noise, which contributes little to the overall signal
        ts_frequency_data_filtered = ts_frequency_data[:, 0 : self.cutoff_frequency, :]

        # 3.1) concatenate the real and imaginary parts of the tensor to be passed through the real valued linear layer
        ts_frequency_data_filtered = torch.cat((ts_frequency_data_filtered.real, ts_frequency_data_filtered.imag), dim=-1)

        # 4) Run the tensor through a real valued linear layer
        if self.individual:
            real_valued_data = self.channel_wise_frequency_upsampler(
                ts_frequency_data_filtered
            )
        else:
            real_valued_data = self.frequency_upsampler(
                ts_frequency_data_filtered.permute(0, 2, 1)
            ).permute(0, 2, 1)

        # 5) obtain new frequencies from the output of the real valued linear layer
        norm_spec_xy = torch.zeros(
            [
                real_valued_data.size(0),
                int((self.seq_len + self.pred_len) / 2 + 1),
                real_valued_data.size(2),
            ],
            dtype=real_valued_data.dtype,
        ).to(real_valued_data.device)

        # 6) 0 pad the output tensor
        norm_spec_xy[:, 0 : real_valued_data.size(1), :] = real_valued_data

        # 7) perform inverse real fast fourier transform on the output tensor
        norm_spec_xy = torch.complex(norm_spec_xy[:, :, :norm_spec_xy.size(2)//2], norm_spec_xy[:, :, norm_spec_xy.size(2)//2:])
        
        norm_xy = torch.fft.irfft(norm_spec_xy, dim=1)
        norm_xy = norm_xy * self.upsample_rate

        # 8) Reverse Normalization
        xy = norm_xy * torch.sqrt(ts_var) + ts_mean

        if self.debug:
            self.debug_tensors = {
                "normalized_ts_data": normalized_ts_data,
                "ts_frequency_data": ts_frequency_data,
                "ts_frequency_data_filtered": ts_frequency_data_filtered,
                "real_valued_data": real_valued_data,
                "norm_spec_xy": norm_spec_xy,
                "norm_xy": norm_xy,
                "xy": xy,
            }

        return xy
    

if __name__ == "__main__":
    args = Namespace(
        dominance_freq=10,
        seq_len=720,
        pred_len=720,
        upsample_rate=0,
        channels=1,
        num_layers=4,
        num_hidden=512,
        individual=False,
        debug=False,
    )
    model = FITS(args)
    
    ts_data = torch.randn(1, 720, 1)
    output = model(ts_data)
    print(output.shape)