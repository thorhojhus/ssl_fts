import torch
import pytest
import torch

from src.dataset import data_setup

from src.models.FITS import FITS
from src.models.copy_fits import Model

def test_normal_versus_normalization():
    a = torch.rand(1, 4)

    # 1) Normalization of the input tensor:
    ts_mean, ts_var = torch.mean(a, dim=1, keepdim=True), torch.var(a, dim=1, keepdim=True) + 1e-10
    normalized_ts_data = (a - ts_mean) / ts_var

    # 2) compare this with the normlization function
    normalized_ts_data_2 = torch.nn.functional.normalize(a, dim=1, p=2)

    assert not torch.allclose(normalized_ts_data, normalized_ts_data_2)


@pytest.mark.parametrize("individual", ['--individual', '--no-individual'])
def test_equal_param_sizes(argparser, individual):

    args = argparser.parse_args(['--device', 'cpu', individual])

    model = FITS(args)
    model_2 = Model(args)

    def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert count_parameters(model) == count_parameters(model_2)
    

def test_equal_train(FITS, ogFITS, argparser):
    args = argparser.parse_args()
    train_loader, _ = data_setup(args)

    FITS.train()
    ogFITS.train()

    for batch_x, _ in train_loader:
        # Do a forward pass to get the tensors into debug
        FITS(batch_x)
        ogFITS(batch_x)

        FITStensors = FITS.debug_tensors
        ogFITStensors = ogFITS.debug_tensors

        test_tensor_1 = FITStensors['ts_frequency_data']
        test_tensor_2 = ogFITStensors['x']
        test_tensor_2 = torch.fft.rfft(test_tensor_2, dim=1)
        assert test_tensor_1.equal(test_tensor_2)

        test_tensor_1 = test_tensor_1[:,0:FITS.cutoff_frequency,:]

        test_tensor_2[:,ogFITS.dominance_freq:]=0 # LPF
        test_tensor_2 = test_tensor_2[:,0:ogFITS.dominance_freq,:] # LPF

        assert test_tensor_1.equal(test_tensor_2)

        assert torch.allclose(FITStensors['normalized_ts_data'], ogFITStensors['x'])

        print(FITStensors['ts_frequency_data_filtered'][8,0:10,0])
        print(ogFITStensors['low_specx'][8,0:10,0])
        for i in range(0, 64):
            assert torch.allclose(FITStensors['ts_frequency_data_filtered'][9,:,:], ogFITStensors['low_specx'][9,:,:])
        break

def test_low_pass_equal():
    a = torch.rand(2,3,2)
    b = a.clone()

    assert a.equal(b)
    assert a is not b

    a = a[:, 0:2, :]
    b[:,2:]=0
    b = b[:, 0:2, :]

    assert a.equal(b)