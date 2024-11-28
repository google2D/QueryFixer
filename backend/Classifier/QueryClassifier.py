import torch.nn as nn
import torch

# Model Definition
class QueryClassifier(nn.Module):
  def __init__(self, input_size):
    super(QueryClassifier, self).__init__()
    self.hid1 = nn.Sequential(
        nn.Linear(input_size, 100),
        nn.BatchNorm1d(100),
        nn.LeakyReLU()
    )
    self.hid2 = nn.Sequential(
        nn.Linear(100, 50),
        nn.BatchNorm1d(50),
        nn.LeakyReLU()
    )
    self.hid3 = nn.Sequential(
        nn.Linear(50, 25),
        nn.BatchNorm1d(25),
        nn.LeakyReLU()
    )
    self.hid4 = nn.Sequential(
        nn.Linear(25, 10),
        nn.BatchNorm1d(10),
        nn.LeakyReLU()
    )

    self.output = nn.Linear(10, 2)

    # Weight Initialization
    for layer in self.children():
        if isinstance(layer, nn.Linear):
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='leaky_relu')
            nn.init.zeros_(layer.bias)

  def forward(self, x):
    z = self.hid1(x)
    z = self.hid2(z)
    z = self.hid3(z)
    z = self.hid4(z)
    z = torch.softmax(self.output(z), dim=1)
    return z



