import torch

def test_normal_versus_normalization():
    a = torch.rand(1, 4)

    # 1) Normalization of the input tensor:
    ts_mean, ts_var = torch.mean(a, dim=1, keepdim=True), torch.var(a, dim=1, keepdim=True) + 1e-10
    normalized_ts_data = (a - ts_mean) / ts_var

    # 2) compare this with the normlization function
    normalized_ts_data_2 = torch.nn.functional.normalize(a, dim=1, p=2)

    assert not torch.allclose(normalized_ts_data, normalized_ts_data_2)