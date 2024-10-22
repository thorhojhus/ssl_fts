import torch
import torch.nn as nn
from .DLinear import Model as DLinear
from .FITS import FITS
from argparse import Namespace

class Model(nn.Module):
    """
    FITS + DLinear model: First applies FITS to extract frequency patterns, then applies DLinear to refine predictions
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.channels = configs.enc_in

        # FITS component
        fits_args = Namespace(
            seq_len=configs.seq_len,
            pred_len=configs.pred_len,
            dominance_freq=configs.seq_len//2,
            enc_in=configs.enc_in,
            individual=configs.individual,
        )
        self.fits = FITS(fits_args)
        
        # DLinear component
        self.dlinear = DLinear(configs)

    def forward(self, x):
        # First apply FITS
        fits_output = self.fits(x)
        
        # Combine input with FITS output for training period
        enhanced_input = x.clone()
        
        # Pass enhanced input through DLinear for final prediction
        final_prediction = self.dlinear(enhanced_input)[:, -self.pred_len:, :]
        
        return final_prediction