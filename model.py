from torch import nn
import torch 

from torch import nn
import torch 

class Encoder(nn.Module):
  def __init__(self):
    super (Encoder, self).__init__()

    self.encoder = nn.Sequential(
        nn.Conv2d(3, 32, 3, 2, 1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2),

        nn.Conv2d(32, 64, 3, 2, 1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2),

        nn.Conv2d(64, 128, 3, 2, 1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2),

        nn.Conv2d(128, 128, 3, 2, 1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2)
    )
  def forward(self, x):
    return self.encoder(x)


class Decoder(nn.Module):
  def __init__(self):
    super (Decoder, self).__init__()

    self.decoder = nn.Sequential(
        nn.ConvTranspose2d(128, 128, 3, 2, 1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2),

        nn.ConvTranspose2d(128, 64, 5, 2, 1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2),

        nn.ConvTranspose2d(64, 32, 5, 2, 1),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.2),

        nn.ConvTranspose2d(32, 3, 5, 2, 1, 1),
        nn.BatchNorm2d(3),
        nn.LeakyReLU(0.2)
    )

  def forward(self, x):
    return self.decoder(x)


class SourceAE(nn.Module):
  def __init__(self):
    super(SourceAE, self).__init__()

    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self, x):
    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self, x):
    return self.decoder(self.encoder(x))

class TargetAE(nn.Module):
  def __init__(self):
    super(TargetAE, self).__init__()
    
    self.encoder = Encoder()
    self.decoder = Decoder()

  def forward(self, x):
    return self.decoder(self.encoder(x))