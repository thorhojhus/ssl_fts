import torch
import pytest

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
    from src.models.FITS import FITS
    from src.models.copy_fits import Model

    args = argparser.parse_args(['--device', 'cpu', individual])

    model = FITS(args)
    model_2 = Model(args)

    def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert count_parameters(model) == count_parameters(model_2)

    