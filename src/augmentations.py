import torch
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def augmentation(augment_time):
    if augment_time == "batch":
        return BatchAugmentation()
    elif augment_time == "dataset":
        return DatasetAugmentation()


class BatchAugmentation:
    def __init__(self):
        pass

    def freq_mask(self, x, y, rate=0.5, dim=1):
        xy = torch.cat([x, y], dim=1)
        xy_f = torch.fft.rfft(xy, dim=dim)
        m = torch.Tensor(xy_f.shape).uniform_() < rate
        freal = xy_f.real.masked_fill(m, 0)
        fimag = xy_f.imag.masked_fill(m, 0)
        xy_f = torch.complex(freal, fimag)
        xy = torch.fft.irfft(xy_f, dim=dim)
        return xy

    def freq_mix(self, x, y, rate=0.5, dim=1):
        xy = torch.cat([x, y], dim=dim)
        xy_f = torch.fft.rfft(xy, dim=dim)

        m = torch.Tensor(xy_f.shape).uniform_() < rate
        amp = abs(xy_f)
        _, index = amp.sort(dim=dim, descending=True)
        dominant_mask = index > 2
        m = torch.bitwise_and(m, dominant_mask)
        freal = xy_f.real.masked_fill(m, 0)
        fimag = xy_f.imag.masked_fill(m, 0)

        b_idx = torch.randperm(x.shape[0])
        x2, y2 = x[b_idx], y[b_idx]
        xy2 = torch.cat([x2, y2], dim=dim)
        xy2_f = torch.fft.rfft(xy2, dim=dim)

        m = torch.bitwise_not(m)
        freal2 = xy2_f.real.masked_fill(m, 0)
        fimag2 = xy2_f.imag.masked_fill(m, 0)

        freal += freal2
        fimag += fimag2

        xy_f = torch.complex(freal, fimag)

        xy = torch.fft.irfft(xy_f, dim=dim)
        return xy

    def noise(self, x, y, rate=0.05, dim=1):
        xy = torch.cat([x, y], dim=1)
        noise_xy = (torch.rand(xy.shape, device=xy.device) - 0.5) * 0.1
        xy = xy + noise_xy
        return xy

    def noise_input(self, x, y, rate=0.05, dim=1):
        noise = (torch.rand(x.shape, device=x.device) - 0.5) * 0.1
        x = x + noise
        xy = torch.cat([x, y], dim=1)
        return xy

    def vFlip(self, x, y, rate=0.05, dim=1):
        # vertically flip the xy
        xy = torch.cat([x, y], dim=1)
        xy = -xy
        return xy

    def hFlip(self, x, y, rate=0.05, dim=1):
        # horizontally flip the xy
        xy = torch.cat([x, y], dim=1)
        # reverse the order of dim 1
        xy = xy.flip(dims=[dim])
        return xy

    def time_combination(self, x, y, rate=0.5, dim=1):
        xy = torch.cat([x, y], dim=dim)

        b_idx = torch.randperm(x.shape[0])
        x2, y2 = x[b_idx], y[b_idx]
        xy2 = torch.cat([x2, y2], dim=dim)

        xy = (xy + xy2) / 2
        return xy

    def magnitude_warping(self, x, y, rate=0.5, dim=1):
        pass

    def linear_upsampling(self, x, y, rate=0.5, dim=1):
        xy = torch.cat([x, y], dim=dim)
        original_shape = xy.shape
        # randomly cut a segment from xy the length should be half of it
        # generate a random integer from 0 to the length of xy
        start_point = torch.randint(0, original_shape[1] // 2, (1,)).item()

        xy = xy[:, start_point : start_point + original_shape[1] // 2, :]

        # interpolate the xy to the original_shape
        xy = xy.permute(0, 2, 1)
        xy = torch.nn.functional.interpolate(xy, scale_factor=2, mode="linear")
        xy = xy.permute(0, 2, 1)
        return xy


class DatasetAugmentation:
    def __init__(self):
        pass

    def freq_dropout(self, x, y, dropout_rate=0.2, dim=0, keep_dominant=True):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)

        xy = torch.cat([x, y], dim=0)
        xy_f = torch.fft.rfft(xy, dim=0)
        
        m = torch.Tensor(xy_f.shape).uniform_() < dropout_rate

        freal = xy_f.real.masked_fill(m, 0)
        fimag = xy_f.imag.masked_fill(m, 0)
        xy_f = torch.complex(freal, fimag)
        xy = torch.fft.irfft(xy_f, dim=dim)

        x, y = xy[:x.shape[0]], xy[-y.shape[0]:]
        return x, y

    def freq_mix(self, x, y, x2, y2, dropout_rate=0.2):
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        xy = torch.cat([x, y], dim=0)
        xy_f = torch.fft.rfft(xy, dim=0)
        m = torch.Tensor(xy_f.shape).uniform_() < dropout_rate
        amp = abs(xy_f)
        _, index = amp.sort(dim=0, descending=True)
        dominant_mask = index > 2
        m = torch.bitwise_and(m, dominant_mask)
        freal = xy_f.real.masked_fill(m, 0)
        fimag = xy_f.imag.masked_fill(m, 0)

        x2, y2 = torch.from_numpy(x2), torch.from_numpy(y2)
        xy2 = torch.cat([x2, y2], dim=0)
        xy2_f = torch.fft.rfft(xy2, dim=0)

        m = torch.bitwise_not(m)
        freal2 = xy2_f.real.masked_fill(m, 0)
        fimag2 = xy2_f.imag.masked_fill(m, 0)

        freal += freal2
        fimag += fimag2

        xy_f = torch.complex(freal, fimag)
        xy = torch.fft.irfft(xy_f, dim=0)
        x, y = xy[: x.shape[0], :].numpy(), xy[-y.shape[0] :, :].numpy()
        return x, y
    
    def freq_warp(x, y, dropout_rate=0.2, dim=0, warp_factor=0.5):
        x, y = torch.from_numpy(x), torch.from_numpy(y)

        xy = torch.cat([x, y], dim=0)
        xy_f = torch.fft.rfft(xy, dim=0)


        freq_indices = torch.arange(xy_f.size(dim)).float()
        warped_indices = torch.pow(freq_indices, warp_factor)
        max_index = xy_f.size(dim) - 1
        warped_indices = torch.clamp(warped_indices, 0, max_index).long()

        xy_f_warped = torch.zeros_like(xy_f)
        for i, wi in enumerate(warped_indices):
            xy_f_warped[wi] += xy_f[i]


        dropout_mask = torch.Tensor(xy_f_warped.shape).uniform_() < dropout_rate
        freal = xy_f_warped.real.masked_fill(dropout_mask, 0)
        fimag = xy_f_warped.imag.masked_fill(dropout_mask, 0)
        xy_f_warped = torch.complex(freal, fimag)

        xy_warped = torch.fft.irfft(xy_f_warped, n=xy.size(dim), dim=dim)

        x, y = xy_warped[:x.shape[0], :].numpy(), xy_warped[-y.shape[0]:, :].numpy()
        return x, y
