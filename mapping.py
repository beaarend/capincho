import torch.nn as nn
from lora_layers import LinearLoRA

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
    def __init__(self, input_size, token_length, output_n, r=2, alpha=32):
        super(MapperLoRA, self).__init__()
        self._tied_weights_keys = None

        existing_linear_layer_1 = nn.Linear(input_size, (token_length * output_n) // 2)
        existing_linear_layer_2 = nn.Linear((token_length * output_n) // 2, (token_length * output_n))

        self.model = nn.Sequential(
            LinearLoRA(existing_linear_layer_1, r=r, alpha=alpha, fan_in_fan_out=False, dropout_rate=0.25),
            nn.LeakyReLU(),
            LinearLoRA(existing_linear_layer_2, r=r, alpha=alpha, fan_in_fan_out=False, dropout_rate=0.25),
            nn.LeakyReLU())

    def forward(self, x):
        return self.model(x)