from torch import nn


class LogisticRegression(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(LogisticRegression, self).__init__()
        raise NotImplementedError("Your code here. Hint: 1 line in the answer key.")

    def forward(self, x):
        raise NotImplementedError("Your code here. Hint: 1-2 lines in the answer key.")
