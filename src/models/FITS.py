from argparse import Namespace
import torch
from torch import nn
from torch.nn import functional as fn
from torch.fft import rfft


# An implementation of the FITS model as described in https://arxiv.org/abs/2307.03756 (FITS: Frequency Interpolation Time Series Forecasting)
# with better annotation and more clear model structure - the original code for the model can be found here: https://github.com/VEWOXIC/FITS

class FITS(nn.Module):
    """Reimplementation of the FITS model.

    This model contains an annotated and more cleany written model, which is true to the model described in the paper* and the original code.    
    *(RIN was not performed in the code, but instance wise normalization was performed instead).
    """

    # TODO: figure out configuration to the model?
    def __init__(
            self, 
            args: Namespace,
    ):
        super(FITS, self).__init__()

        self.cutoff_frequency = args.dominance_freq
        self.input_length = args.input_length
        self.output_length = args.output_length
        self.upsample_rate = (args.input_length + args.output_length) / args.input_length

        self.complex_linear_layer = nn.Linear(
            in_features=args.dominance_freq,
            out_features=int(args.dominance_freq * self.upsample_rate),
            dtype=torch.cfloat
        )

        self.individual = args.individual
    

    def forward(self, ts_data: torch.Tensor) -> torch.Tensor:
        # 1) Normalization of the input tensor:
        ts_mean, ts_var = torch.mean(ts_data, dim=1, keepdim=True), torch.var(ts_data, dim=1, keepdim=True)
        normalized_ts_data = fn.normalize(
            input=ts_data, 
            dim=1, 
            eps=1e-12
        )

        # 2) perform real fast fourier transform on the input tensor
        ts_frequency_data = rfft(input=normalized_ts_data, dim=1)

        # 3) perform a low pass filter to remove high frequency noise, which contributes little to the overall signal
        ts_frequency_data_filtered = ts_frequency_data[:,0:self.cutoff_frequency,:]

        # 4) Run the tensor through a complex valued linear layer
        if self.individual:
            raise NotImplementedError("Individual frequency upsampling is not implemented yet")
        else:
            complex_valued_data = self.complex_linear_layer(ts_frequency_data_filtered.permute(0,2,1)).permute(0,2,1)

        # 5) obtain new frequencies from the output of the complex valued linear layer
        norm_spec_xy = torch.zeros(
            [
                complex_valued_data.size(0),
                int((self.input_length+self.output_length)/2+1),
                complex_valued_data.size(2)
            ],
            dtype=complex_valued_data.dtype
        ).to(complex_valued_data.device)

        # 6) 0 pad the output tensor
        norm_spec_xy[:,0:complex_valued_data.size(1), :] = complex_valued_data

        # 7) perform inverse real fast fourier transform on the output tensor
        norm_xy = torch.fft.irfft(norm_spec_xy, dim=1)
        norm_xy = norm_xy * self.upsample_rate

        # 8) Reverse Normalization
        xy = norm_xy * torch.sqrt(ts_var) + ts_mean
        print(xy, torch.sqrt(ts_var), ts_mean)
        raise NotImplementedError("The model is not yet complete")
        return xy
        