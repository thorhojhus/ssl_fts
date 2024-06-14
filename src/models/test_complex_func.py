import torch
from torch import nn
from torch.nn import functional as F

class ComplexReLU(nn.Module):
    def __init__(self):
        super(ComplexReLU, self).__init__()
        self.relu = nn.ReLU()

    def forward(self, x):
        real = self.relu(x.real)
        imag = self.relu(x.imag)
        return torch.complex(real, imag)

class ModReLU(nn.Module):
    def __init__(self, input_size):
        super(ModReLU, self).__init__()
        self.b = nn.Parameter(torch.zeros(input_size))

    def forward(self, z):
        magnitude = torch.abs(z)
        relu = F.relu(magnitude + self.b)
        normalized = z / (magnitude + 1e-8)
        return relu * normalized

def dropout_complex(x, p=0.5, training=True):
    if x.is_complex():
        mask_real = F.dropout(torch.ones_like(x.real), p, training)
        mask_imag = F.dropout(torch.ones_like(x.imag), p, training)
        return torch.complex(x.real * mask_real, x.imag * mask_imag)
    else:
        return F.dropout(x, p, training)

class ComplexDropout(nn.Module):
    def __init__(self, p=0.5):
        super(ComplexDropout, self).__init__()
        self.p = p

    def forward(self, x):
        return dropout_complex(x, self.p, self.training)

# Example test code to ensure the dropout complex is working correctly:
if __name__ == "__main__":
    complex_data = torch.complex(torch.randn(10, 5), torch.randn(10, 5))
    dropout_layer = ComplexDropout(p=0.5)
    dropout_layer.train()
    dropped_data = dropout_layer(complex_data)
    print("Original Data:", complex_data)
    print("Dropped Data:", dropped_data)
    dropout_layer.eval()
    dropped_data = dropout_layer(complex_data)
    print("Dropped Data:", dropped_data)