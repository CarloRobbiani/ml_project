import torch.nn as nn


class RegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(RegressionModel, self).__init__()

        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation2 = nn.ReLU()
        self.layer3 = nn.Linear(hidden_dim, output_dim)

        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = self.activation2(x)
        x = self.dropout(x)
        x = self.layer3(x)
        return x
