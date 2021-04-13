from torch import nn
import torch 

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()

    self.model = nn.Sequential(
        nn.Conv2d(3, 6, 4, 2, 1, bias = False),
        nn.BatchNorm2d(6),
        nn.LeakyReLU(0.2),

        nn.Conv2d(6, 64, 4, 2, 1, bias = False),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2),

        nn.Conv2d(64, 128, 4, 2, 1, bias = False),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2),

        nn.Conv2d(128, 256, 5, 2, 1, bias = False),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2),

        nn.Flatten(),
        nn.Sigmoid(),
        ## need to calculate number of in_features
        nn.Linear( , 1)
    )

  def forward(self, x):
    return self.model(x)