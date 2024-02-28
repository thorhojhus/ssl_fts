import torch
from torch import nn



# An implementation of the FITS model as described in https://arxiv.org/abs/2307.03756 (FITS: Frequency Interpolation Time Series Forecasting)
# with better annotation and more clear model structure - the original code for the model can be found here: https://github.com/VEWOXIC/FITS

class FITS(nn.Module):
    """Reimplementation of the FITS model.

    This model contains an annotated and more cleany written model, which is true to the model described in the paper* and the original code.    
    *(RIN was not performed in the code, but instance wise normalization was performed instead).
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Normalization of the input tensor:
        # 2) perform real fast fourier transform on the input tensor
        # 3) perform a low pass filter to remove high frequency noise, which contributes little to the overall signal
        # 4) Run the tensor through a complex valued linear layer
        # 5) obtain new frequencies from the output of the complex valued linear layer
        # 6) 0 pad the output tensor
        # 7) perform inverse real fast fourier transform on the output tensor
        # 8) profit
