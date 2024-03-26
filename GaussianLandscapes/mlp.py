import torch
import torch.nn as nn

import constants

device = constants.DEVICE
class MLP(nn.Module):
    def __init__(self, image_dim = (28, 28), num_classes = 10):
        super(MLP, self).__init__()
        self.input_size = image_dim[0] * image_dim[1]
        self.num_classes = num_classes

        # self.linear_architecture_large = nn.Sequential(
        #     nn.Linear(self.input_size, int(self.input_size * 1.5)),
        #     nn.ReLU(),
        #     nn.Linear(int(self.input_size * 1.5), self.input_size),
        #     nn.ReLU(),
        #     nn.Linear(self.input_size,num_classes)
        # )

        self.linear_architecture_small = nn.Sequential(
            nn.Linear(self.input_size, 100),
            nn.ReLU(),
            nn.Linear(100,num_classes)
        )

        self.softmax = nn.Softmax(dim=1)
    def forward(self, x):
        x = x.squeeze()
        x = x.reshape(-1, self.input_size)
        x = self.linear_architecture_small(x)
        return self.softmax(x)

if __name__ == "__main__":
    sample_x = torch.randn(10, 1, 28, 28).to(device)
    mlp = MLP().to(device)
    output = mlp(sample_x)