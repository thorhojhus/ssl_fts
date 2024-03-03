import torch
import torch.nn as nn

# fmt: off
class FITS(nn.Module):
    def __init__(self, args):
        super(FITS, self).__init__()
        self.seq_len = args.input_length
        self.pred_len = args.output_length
        self.individual = args.individual
        self.channels = args.channels

        self.dominance_freq= args.dominance_freq
        
        self.length_ratio = (self.seq_len + self.pred_len)/self.seq_len

        if self.individual:
            self.freq_upsampler = nn.ModuleList()
            for i in range(self.channels):
                self.freq_upsampler.append(nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat))

        else:
            self.freq_upsampler = nn.Linear(self.dominance_freq, int(self.dominance_freq*self.length_ratio)).to(torch.cfloat) # complex layer for frequency upsampling

    def forward(self, x):
        # RIN 
        x_mean = torch.mean(x, dim=1, keepdim=True)
        x = x - x_mean
        x_var = torch.var(x, dim=1, keepdim=True) + 1e-5
        x = x / torch.sqrt(x_var)

        low_specx = torch.fft.rfft(x, dim=1)

        low_specx[:,self.dominance_freq:] = 0 # LPF
        low_specx = low_specx[:,0:self.dominance_freq,:] # LPF

        if self.individual:
        
            low_specxy_ = torch.zeros(
                [low_specx.size(0),
                 int(self.dominance_freq * self.length_ratio),
                 low_specx.size(2)],
                 dtype=low_specx.dtype).to(low_specx.device)
            
            for i in range(self.channels):
                low_specxy_[:,:,i] = self.freq_upsampler[i](
                    low_specx[:,:,i]
                    .permute(0,1)
                    ).permute(0,1)
        else:
            low_specxy_ = self.freq_upsampler(
                low_specx
                .permute(0,2,1)
                ).permute(0,2,1)
                
            
        low_specxy = torch.zeros([
            low_specxy_.size(0),
            int((self.seq_len + self.pred_len) / 2 + 1 ),
            low_specxy_.size(2)],
            dtype=low_specxy_.dtype
            ).to(low_specxy_.device)
        
        
        low_specxy[:, 0:low_specxy_.size(1), :] = low_specxy_ 


        low_xy = torch.fft.irfft(low_specxy, dim=1)


        low_xy = low_xy * self.length_ratio
        
        xy = (low_xy) * torch.sqrt(x_var) + x_mean

        return xy #, low_xy * torch.sqrt(x_var)
# fmt: on