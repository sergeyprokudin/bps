import os
import sys
import h5py
import numpy as np


def download_modelnet40_data(url, root_data_dir):
    import urllib

    def _download_reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            percent = readsofar * 1e2 / totalsize
            s = "\r%5.1f%% %*d / %d" % (
                percent, len(str(totalsize)), readsofar, totalsize)
            sys.stderr.write(s)
            if readsofar >= totalsize:  # near the end
                sys.stderr.write("\n")
        else:  # total size is unknown
            sys.stderr.write("read %d\n" % (readsofar,))

    def _unzip_data(zip_path, target_path):

        from zipfile import ZipFile

        with ZipFile(zip_path, 'r') as zip:
            zip.printdir()
            zip.extractall(path=target_path)

        return

    if not os.path.exists(root_data_dir):
        os.makedirs(root_data_dir)

    download_path = os.path.join(root_data_dir, 'modelnet40_ply_hdf5_2048.zip')

    print("downloading ModelNet40 data..")
    urllib.request.urlretrieve(url, download_path, _download_reporthook)

    print('unzipping files..')
    _unzip_data(download_path, root_data_dir)

    return


def load_modelnet40(root_data_dir):
    """
    Loads ModelNet40 point cloud data. The dataset itself can be downloaded from here:

    https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip

    Parameters
    ----------
    data_path: string
        target directory for data

    Returns
    -------
    point clouds and corresponding class labels:

    xtr: [n_train_samples, 2048, 3]
    ytr: [n_train_samples, 1]
    xte: [n_test_samples, 2048, 3]
    yte: [n_test_samples, 1]

    """
    print("loading ModelNet40 point clouds...")

    def _load_data(file_paths, num_points=2048):
        def _load_h5(h5_filename):
            f = h5py.File(h5_filename, 'r')
            data = f["data"][:]
            label = f["label"][:]
            return (data, label)

        points = None
        labels = None
        for d in file_paths:
            cur_points, cur_labels = _load_h5(d)
            cur_points = cur_points.reshape(1, -1, 3)
            cur_labels = cur_labels.reshape(1, -1)
            if labels is None or points is None:
                labels = cur_labels
                points = cur_points
            else:
                labels = np.hstack((labels, cur_labels))
                points = np.hstack((points, cur_points))
        points_r = points.reshape(-1, num_points, 3)
        labels_r = labels.reshape(-1, 1)

        assert points_r.shape[0] == labels_r.shape[0]
        assert labels_r.shape[1] == 1
        assert points_r.shape[1] == 2048
        assert points_r.shape[2] == 3

        return points_r, labels_r

    def _get_file_names(data_path, file_lst):

        with open(file_lst, "r") as f:
            files = f.readlines()

        files = [os.path.join(data_path, os.path.basename(path)[:-1]) for path in files]

        return files

    data_dir = os.path.join(root_data_dir, 'modelnet40_ply_hdf5_2048')

    if not os.path.exists(data_dir):

        os.makedirs(data_dir)

        modelnet40_url = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"

        download_modelnet40_data(modelnet40_url, root_data_dir)

    train_file_lst = os.path.join(data_dir, "train_files.txt")
    test_file_lst = os.path.join(data_dir, "test_files.txt")

    xtr, ytr = _load_data(_get_file_names(data_dir, train_file_lst))
    xte, yte = _load_data(_get_file_names(data_dir, test_file_lst))

    print("loaded %d training and %d test samples." % (xtr.shape[0], xte.shape[0]))

    return xtr, ytr, xte, yte
