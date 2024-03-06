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
    def __init__(self, cutoff_frequency: int, upsample_rate: float):
        super().__init__()

        self.cutoff_frequency = cutoff_frequency
        self.upsample_rate = upsample_rate

        self.complex_linear_layer = nn.Linear(
            in_features=cutoff_frequency,
            out_features=int(cutoff_frequency * upsample_rate),
            dtype=torch.cfloat
        )

    def _low_pass_filter(self, ts_frequency_data: torch.Tensor) -> torch.Tensor:
        """Applies a low pass filter to the input tensor to remove high frequency noise.
        
        The function will zero out all frequencies above the parsed dominant frequency to the model.
        """
        return ts_frequency_data[:,0:self.dominance_freq,:]


    def forward(self, ts_data: torch.Tensor) -> torch.Tensor:
        # 1) Normalization of the input tensor:
        normalized_ts_data = fn.normalize(
            input=ts_data, 
            dim=1, 
            eps=1e-12
        )

        # 2) perform real fast fourier transform on the input tensor
        ts_frequency_data = rfft(input=normalized_ts_data)

        # 3) perform a low pass filter to remove high frequency noise, which contributes little to the overall signal
        ts_frequency_data_filtered = self.low_pass_filter(ts_frequency_data)

        # 4) Run the tensor through a complex valued linear layer
        self.complex_linear_layer(ts_frequency_data_filtered)

        # 5) obtain new frequencies from the output of the complex valued linear layer
        
        
        # 6) 0 pad the output tensor
        # 7) perform inverse real fast fourier transform on the output tensor
        # 8) profit
        raise NotImplementedError("The rest of the forward function is not implemented yet")