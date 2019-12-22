import numpy as np
import os
import sys
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
BPS_CACHE_FILE = os.path.join(DATA_PATH, 'bps_mlp_data.npz')
LOGS_PATH = os.path.join(PROJECT_DIR, 'logs')

N_MODELNET_CLASSES = 40

N_BPS_POINTS = 512
BPS_RADIUS = 1.7

N_CPUS = multiprocessing.cpu_count()
N_GPUS = torch.cuda.device_count()

if N_GPUS > 0:
    DEVICE = 'cuda'
    print("GPU device found...")
else:
    DEVICE = 'cpu'
    print("GPU device not found, using %d CPU(s)..." % N_CPUS)

if not os.path.exists(LOGS_PATH):
    os.makedirs(LOGS_PATH)


class ShapeClassifierMLP(nn.Module):

    def __init__(self, n_features, n_classes, hsize1=512,  hsize2=512, dropout1=0.8, dropout2=0.8):
        super(ShapeClassifierMLP, self).__init__()

        self.bn0 = nn.BatchNorm1d(n_features)
        self.fc1 = nn.Linear(in_features=n_features, out_features=hsize1)
        self.bn1 = nn.BatchNorm1d(hsize1)
        self.do1 = nn.Dropout(dropout1)
        self.fc2 = nn.Linear(in_features=hsize1, out_features=hsize2)
        self.bn2 = nn.BatchNorm1d(hsize2)
        self.do2 = nn.Dropout(dropout2)
        self.fc3 = nn.Linear(in_features=hsize2, out_features=n_classes)

    def forward(self, x):

        x = self.bn0(x)
        x = self.do1(self.bn1(F.relu(self.fc1(x))))
        x = self.do2(self.bn2(F.relu(self.fc2(x))))
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

        # this will encode your normalised point clouds with random basis of 512 points,
        # each BPS cell containing l2-distance to closest point
        start = time.time()
        print("converting data to BPS representation..")
        print("number of basis points: %d" % N_BPS_POINTS)
        print("BPS sampling radius: %f" % BPS_RADIUS)
        print("converting train..")
        xtr_bps = bps.encode(xtr_normalized, n_bps_points=N_BPS_POINTS, bps_cell_type='dists', radius=BPS_RADIUS)
        print("converting test..")
        xte_bps = bps.encode(xte_normalized, n_bps_points=N_BPS_POINTS, bps_cell_type='dists', radius=BPS_RADIUS)
        end = time.time()
        total_training_time = (end - start) / 60
        print("conversion finished. ")
        print("saving cache file for future runs..")

        np.savez(BPS_CACHE_FILE, xtr=xtr_bps, ytr=ytr, xte=xte_bps, yte=yte)

    else:
        print("loading converted data from cache..")
        data = np.load(BPS_CACHE_FILE)
        xtr_bps = data['xtr']
        ytr = data['ytr']
        xte_bps = data['xte']
        yte = data['yte']

    dataset_tr = pt.utils.data.TensorDataset(pt.Tensor(xtr_bps), pt.Tensor(ytr[:, 0]).long())
    train_loader = pt.utils.data.DataLoader(dataset_tr, batch_size=512, shuffle=True)

    dataset_te = pt.utils.data.TensorDataset(pt.Tensor(xte_bps), pt.Tensor(yte[:, 0]).long())
    test_loader = pt.utils.data.DataLoader(dataset_te, batch_size=512, shuffle=True)

    return train_loader, test_loader


def main():

    train_loader, test_loader = prepare_data_loaders()

    n_bps_features = train_loader.dataset[0][0].shape[0]

    print("defining the model..")
    model = ShapeClassifierMLP(n_features=n_bps_features, n_classes=N_MODELNET_CLASSES)

    optimizer = pt.optim.Adam(model.parameters(), lr=1e-3)

    n_epochs = 1200
    pbar = range(0, n_epochs)
    test_accs = []
    test_losses = []

    print("training started..")
    model = model.to(DEVICE)

    start = time.time()

    for epoch_idx in pbar:
        fit(model, DEVICE, train_loader, optimizer)
        if epoch_idx == 1000:
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
    ckpt_path = os.path.join(LOGS_PATH, 'bps_mlp_model.h5')

    pt.save(model.state_dict(), ckpt_path)

    print("Model saved: %s" % ckpt_path)

    return


if __name__ == '__main__':
    main()