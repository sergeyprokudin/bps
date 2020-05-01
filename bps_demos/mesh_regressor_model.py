import torch
import torch.nn as nn
import torch.nn.functional as F


class MeshRegressorMLP(nn.Module):
    """
    Weights for the model can be downloaded from here:

    https://www.dropbox.com/s/u3d1uighrtcprh2/mesh_regressor.h5?dl=0

    """
    def __init__(self, n_features, hsize=1024):
        super(MeshRegressorMLP, self).__init__()

        self.bn0 = nn.BatchNorm1d(n_features)
        self.fc1 = nn.Linear(in_features=n_features, out_features=hsize)
        self.bn1 = nn.BatchNorm1d(hsize)
        self.fc2 = nn.Linear(in_features=hsize, out_features=hsize)
        self.bn2 = nn.BatchNorm1d(hsize)

        self.fc3 = nn.Linear(in_features=hsize + n_features, out_features=hsize)
        self.bn3 = nn.BatchNorm1d(hsize)
        self.fc4 = nn.Linear(in_features=hsize, out_features=hsize)
        self.bn4 = nn.BatchNorm1d(hsize)

        self.fc5 = nn.Linear(in_features=hsize*2 + n_features, out_features=6890*3)

    def forward(self, x):

        x = self.bn0(x)
        x0 = x
        x = self.bn1(F.relu(self.fc1(x)))
        x = self.bn2(F.relu(self.fc2(x)))
        x1 = x
        x = torch.cat([x0, x], 1)
        x = self.bn3(F.relu(self.fc3(x)))
        x = self.bn4(F.relu(self.fc4(x)))

        x = torch.cat([x0, x1, x], 1)

        x = self.fc5(x).reshape([-1, 6890, 3])

        return x