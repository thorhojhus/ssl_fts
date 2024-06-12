import torch
import pytest
import torch

from src.dataset import data_setup

from src.models.FITS import FITS
from src.models.FITS_original import Model


def test_normal_versus_normalization():
    a = torch.rand(1, 4)

    # 1) Normalization of the input tensor:
    ts_mean, ts_var = (
        torch.mean(a, dim=1, keepdim=True),
        torch.var(a, dim=1, keepdim=True) + 1e-10,
    )
    normalized_ts_data = (a - ts_mean) / ts_var

    # 2) compare this with the normlization function
    normalized_ts_data_2 = torch.nn.functional.normalize(a, dim=1, p=2)

    assert not torch.allclose(normalized_ts_data, normalized_ts_data_2)


@pytest.mark.parametrize("individual", ["--individual", "--no-individual"])
def test_equal_param_sizes(argparser, individual):

    args = argparser.parse_args(["--device", "cpu", individual])

    model = FITS(args)
    model_2 = Model(args)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert count_parameters(model) == count_parameters(model_2)


def test_equal_train(FITS, ogFITS, argparser):
    args = argparser.parse_args()
    train_loader, _ = data_setup(args)

    FITS.train()
    ogFITS.train()

    for batch_x, _ in train_loader:
        # Do a forward pass to get the tensors into debug

        batch_x = batch_x.float().to("cpu")

        atol = 1e-4  # 4 significant digits in our data

        FITS(batch_x)
        ogFITS(batch_x)

        FITStensors = FITS.debug_tensors
        ogFITStensors = ogFITS.debug_tensors

        test_mean_var_1 = FITStensors["normalized_ts_data"]
        test_mean_var_2 = ogFITS.debug_tensors["x"]

        test_mean_var_1.equal(test_mean_var_2)

        test_rfft_1 = torch.fft.rfft(test_mean_var_1, dim=1)
        test_rfft_1_comp = FITStensors["ts_frequency_data"]
        assert torch.allclose(test_rfft_1, test_rfft_1_comp, atol=atol)

        test_rfft_2 = torch.fft.rfft(test_mean_var_2, dim=1)
        # print indices where the two tensors differ:
        assert torch.allclose(test_rfft_1_comp, test_rfft_2, atol=atol)

        test_lpf_1 = FITStensors["ts_frequency_data_filtered"]

        test_rfft_2[:, ogFITS.dominance_freq :] = 0  # LPF
        test_lpf_2 = test_rfft_2[:, 0 : ogFITS.dominance_freq, :]  # LPF

        assert torch.allclose(test_lpf_1, test_lpf_2, atol=atol)

        test_complex_data_1 = FITStensors["complex_valued_data"]
        test_complex_data_2 = ogFITStensors["low_specxy_"]

        assert torch.allclose(test_complex_data_1, test_complex_data_2, atol=atol)

        test_norm_spec_xy_1 = FITStensors["norm_spec_xy"]
        test_norm_spec_xy_2 = ogFITStensors["low_specxy"]

        assert torch.allclose(test_norm_spec_xy_1, test_norm_spec_xy_2, atol=atol)

        test_norm_xy_1 = FITStensors["norm_xy"]
        test_norm_xy_2 = ogFITStensors["low_xy"]

        assert torch.allclose(test_norm_xy_1, test_norm_xy_2, atol=atol)

        test_xy_1 = FITStensors["xy"]
        test_xy_2 = ogFITStensors["xy"]

        assert torch.allclose(test_xy_1, test_xy_2, atol=atol)

        break


def test_low_pass_equal():
    a = torch.rand(2, 3, 2)
    b = a.clone()

    assert a.equal(b)
    assert a is not b

    a = a[:, 0:2, :]
    b[:, 2:] = 0
    b = b[:, 0:2, :]

    assert a.equal(b)
