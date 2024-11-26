import torch.nn as nn


class Mapper(nn.Module):
    def __init__(self, input_size, token_length, output_n, device):
        super(Mapper, self).__init__()
        self.device = device
        self.model = nn.Sequential(
            nn.Linear(input_size, (token_length * output_n) // 2),
            nn.LeakyReLU(),
            nn.Linear((token_length * output_n) // 2, (token_length * output_n)),
            nn.LeakyReLU())

    def forward(self, x):
        device = x.get_device()
        x = self.model(x.to(self.device()))
        if device >= 0:
            x = x.to(f'cuda:{device}')
        else:
            x = x.to(f'cpu')
        return x


