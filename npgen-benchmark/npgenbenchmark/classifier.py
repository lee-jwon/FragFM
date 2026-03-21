import torch
import torch.nn as nn


class ConvertedModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.fc1 = nn.Linear(2048 + 4096, 6144)
        self.bn1 = nn.BatchNorm1d(6144)

        self.fc2 = nn.Linear(6144, 3072)
        self.bn2 = nn.BatchNorm1d(3072)

        self.fc3 = nn.Linear(3072, 2304)
        self.bn3 = nn.BatchNorm1d(2304)

        self.fc4 = nn.Linear(2304, 1152)
        self.relu4 = nn.ReLU()

        self.dropout = nn.Dropout(0.2)
        self.output_fc = nn.Linear(1152, num_classes)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_f, input_b):
        x = torch.cat((input_f, input_b), dim=1)

        # Pass through layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.bn1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.bn2(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.bn3(x)

        x = self.fc4(x)
        x = self.relu(x)

        x = self.dropout(x)
        x = self.output_fc(x)
        x = self.sigmoid(x)
        return x
