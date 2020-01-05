import numpy as np
import os
import sys
from functools import partial
import multiprocessing
import time

# PyTorch dependencies
import torch as pt
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F

# local dependencies
from bps import bps
from modelnet40 import load_modelnet40


SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, 'data')
LOGS_PATH = os.path.join(PROJECT_DIR, 'logs')

BPS_CACHE_FILE = os.path.join(DATA_PATH, 'bps_conv3d_data.npz')

N_MODELNET_CLASSES = 40

N_BPS_POINTS = 32**3
BPS_RADIUS = 1.2

N_CPUS = multiprocessing.cpu_count()
N_GPUS = torch.cuda.device_count()

if N_GPUS > 0:
    DEVICE = 'cuda'
    print("GPU device found...")
else:
    DEVICE = 'cpu'
    print("GPU device not found, using %d CPU(s), might be slow..." % N_CPUS)

if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)


class ShapeClassifierConv3D(nn.Module):

    def __init__(self, n_features, n_classes):
        super(ShapeClassifierConv3D, self).__init__()

        self.conv11 = nn.Conv3d(in_channels=n_features, out_channels=8, kernel_size=(3, 3, 3))
        self.bn11 = nn.BatchNorm3d(8)
        self.conv12 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3))
        self.bn12 = nn.BatchNorm3d(16)
        self.mp1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv21 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3))
        self.bn21 = nn.BatchNorm3d(32)
        self.conv22 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3, 3, 3))
        self.bn22 = nn.BatchNorm3d(64)
        self.mp2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.do1 = nn.Dropout(0.8)
        self.fc1 = nn.Linear(in_features=8000, out_features=512)
        self.bn1 = nn.BatchNorm1d(512)
        self.do2 = nn.Dropout(0.8)
        self.fc3 = nn.Linear(in_features=512, out_features=n_classes)

    def forward(self, x):

        x = self.bn11(F.relu(self.conv11(x)))
        x = self.bn12(F.relu(self.conv12(x)))
        x = self.mp1(x)

        x = self.bn21(F.relu(self.conv21(x)))
        x = self.bn22(F.relu(self.conv22(x)))
        x = self.mp2(x)

        x = self.do1(x.reshape([-1, 8000]))
        x = self.do2(self.bn1(F.relu(self.fc1(x))))

        x = self.fc3(x)

        return x


def fit(model, device, train_loader, optimizer):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()


def test(model, device, test_loader, epoch_id):
    model.eval()
    test_loss = 0
    n_test_samples = len(test_loader.dataset)
    n_correct = 0
    with pt.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            n_correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= n_test_samples
    test_acc = 100.0 * n_correct / n_test_samples
    print(
        "Epoch {} test loss: {:.4f}, test accuracy: {}/{} ({:.2f}%)".format(epoch_id, test_loss, n_correct, n_test_samples, test_acc))

    return test_loss, test_acc


def prepare_data_loaders():

    if not os.path.exists(BPS_CACHE_FILE):
        # load modelnet point clouds
        xtr, ytr, xte, yte = load_modelnet40(root_data_dir=DATA_PATH)

        # this will normalise your point clouds and return scaler parameters for inverse operation
        xtr_normalized = bps.normalize(xtr)
        xte_normalized = bps.normalize(xte)

        print("converting data to BPS representation..")
        print("number of basis points: %d" % N_BPS_POINTS)
        print("BPS sampling radius: %f" % BPS_RADIUS)

        print("converting train..")
        xtr_bps = bps.encode(xtr_normalized, bps_arrangement='grid', n_bps_points=N_BPS_POINTS,
                             radius=BPS_RADIUS, bps_cell_type='deltas')
        xtr_bps = xtr_bps.reshape([-1, 32, 32, 32, 3])

        print("converting test..")
        xte_bps = bps.encode(xte_normalized, bps_arrangement='grid', n_bps_points=N_BPS_POINTS,
                             radius=BPS_RADIUS, bps_cell_type='deltas')

        xte_bps = xte_bps.reshape([-1, 32, 32, 32, 3])

        print("saving cache file for future runs..")
        np.savez(BPS_CACHE_FILE, xtr=xtr_bps, ytr=ytr, xte=xte_bps, yte=yte)

    else:
        print("loading converted data from cache..")
        data = np.load(BPS_CACHE_FILE)
        xtr_bps = data['xtr']
        ytr = data['ytr']
        xte_bps = data['xte']
        yte = data['yte']

    xtr_bps = xtr_bps.transpose(0, 4, 2, 3, 1)
    dataset_tr = pt.utils.data.TensorDataset(pt.Tensor(xtr_bps), pt.Tensor(ytr[:, 0]).long())
    tr_loader = pt.utils.data.DataLoader(dataset_tr, batch_size=64, shuffle=True)

    xte_bps = xte_bps.transpose(0, 4, 2, 3, 1)
    dataset_te = pt.utils.data.TensorDataset(pt.Tensor(xte_bps), pt.Tensor(yte[:, 0]).long())
    te_loader = pt.utils.data.DataLoader(dataset_te, batch_size=64, shuffle=True)

    return tr_loader, te_loader


def main():

    train_loader, test_loader = prepare_data_loaders()

    n_bps_features = train_loader.dataset[0][0].shape[0]

    print("defining the model..")
    model = ShapeClassifierConv3D(n_features=n_bps_features, n_classes=N_MODELNET_CLASSES)

    optimizer = pt.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 1000
    pbar = range(0, n_epochs)
    test_accs = []
    test_losses = []

    print("training started..")
    model = model.to(DEVICE)

    start = time.time()
    for epoch_idx in pbar:
        fit(model, DEVICE, train_loader, optimizer)
        if epoch_idx == 800:
            for param_group in optimizer.param_groups:
                print("decreasing the learning rate to 1e-4..")
                param_group['lr'] = 1e-4
        if epoch_idx % 10 == 0:
            test_loss, test_acc = test(model, DEVICE, test_loader, epoch_idx)
            test_accs.append(test_acc)
            test_losses.append(test_loss)

    _, test_acc = test(model, DEVICE, test_loader, n_epochs)

    end = time.time()
    total_training_time = (end - start) / 60

    print("Training finished. Test accuracy: %f . Total training time: %f minutes." % (test_acc, total_training_time))
    ckpt_path = os.path.join(LOGS_PATH, 'bps_conv3d_model.h5')

    pt.save(model.state_dict(), ckpt_path)
    print("Model saved: %s" % ckpt_path)

    return


if __name__ == '__main__':
    main()
