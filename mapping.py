import torch.nn as nn
from lora_layers import LoRALinear

class Mapper(nn.Module):
    def __init__(self, input_size, token_length, output_n):
        super(Mapper, self).__init__()
        self._tied_weights_keys = None
        self.model = nn.Sequential(
            nn.Linear(input_size, (token_length * output_n) // 2),
            nn.LeakyReLU(),
            nn.Linear((token_length * output_n) // 2, (token_length * output_n)),
            nn.LeakyReLU())

    def forward(self, x):
        return self.model(x)

class MapperLoRA(nn.Module):
    def __init__(self, input_size, token_length, output_n, r=16, alpha=32):
        super(MapperLoRA, self).__init__()
        self._tied_weights_keys = None
        self.model = nn.Sequential(
            LoRALinear(input_size, (token_length * output_n) // 2, r=r, alpha=alpha),
            nn.LeakyReLU(),
            LoRALinear((token_length * output_n) // 2, (token_length * output_n), r=r, alpha=alpha),
            nn.LeakyReLU())

    def forward(self, x):
        return self.model(x)