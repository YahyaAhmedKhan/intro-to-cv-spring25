from torch import nn

class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(LogisticRegression, self).__init__()
        self.layers = nn.Sequential(nn.Linear(input_dim, num_classes))

    def forward(self, x):
        return self.layers(x)
