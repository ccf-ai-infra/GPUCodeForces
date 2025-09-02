import torch
import torch.nn as nn

class Model(nn.Module):
    """
    Model that performs matrix multiplication followed by ReLU activation.
    """
    def __init__(self, weight):
        super(Model, self).__init__()
        self.weight = nn.Parameter(weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs matrix multiplication and applies ReLU activation.

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, input_dim]

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, output_dim]
        """
        x = torch.matmul(x, self.weight)
        return torch.relu(x)

batch_size = 16
input_dim = 1024
output_dim = 2048

def get_inputs():
    x = torch.randn(batch_size, input_dim)
    return [x]

def get_init_inputs():
    weight = torch.randn(input_dim, output_dim)
    return [weight]