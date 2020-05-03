"""Predict SMPL mesh from a noisy human body scan

Download the model and adjust CKPT_PATH accordingly:

mkdir ../data
cd ../data
wget --output-document=mesh_regressor.h5 https://www.dropbox.com/s/u3d1uighrtcprh2/mesh_regressor.h5?dl=0

Usage example:

python run_alignment.py demo_scan.ply ../logs/demo_output

If a directory is provided as a first parameter, the alignment model will be ran on all *.ply files found.

"""

import os
import ntpath
import numpy as np
import trimesh
import torch
import matplotlib
import matplotlib.pyplot as plt
import sys

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

from bps import bps
from mesh_regressor_model import MeshRegressorMLP
from chamfer_distance import chamfer_distance

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_PATH = os.path.join(PROJECT_DIR, 'data')
CKPT_PATH = os.path.join(DATA_PATH, 'mesh_regressor.h5')

BPS_RADIUS = 1.7
N_BPS_POINTS = 1024
MESH_SCALER = 1000
SMPL_FACES = np.loadtxt('smpl_mesh_faces.txt')
DBSCAN_EPS = 0.05
DBSCAN_MIN_SAMPLES = 10
MAX_POINTS_TO_SHOW = 10**5


def load_scan(scan_path, n_sample_points=10000, denoise=True,
              dbscan_eps=DBSCAN_EPS, dbscan_min_samples=DBSCAN_MIN_SAMPLES):
    """Load ply file and sample n_scan_points from it

    """

    scan = trimesh.load_mesh(scan_path)
    n_scan_points = len(scan.vertices)
    if hasattr(scan, 'colors') and (scan.colors[0] is not None):
        scan_rgb = np.asarray(scan.colors[0]) / 255
    else:
        print("no color channel for the point cloud detected, using default green color...")
        scan_rgb = np.tile(np.asarray([0, 1.0, 0.5]), [n_scan_points, 1])

    scan_orig = np.concatenate([np.asarray(scan.vertices), scan_rgb], 1)

    if type(scan) == trimesh.points.PointCloud:
        scan_processed = np.asarray(scan.vertices[np.random.choice(scan.vertices.shape[0], n_sample_points)])
    elif type(scan) == trimesh.base.Trimesh:
        scan_processed = scan.sample(n_sample_points)
    else:
        raise ValueError("Unrecognized ply file format!")

    if denoise:
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(scan_processed)
        scan_processed = scan_processed[clustering.labels_ == 0]

    return scan_orig, scan_processed


def get_alignment(x_scan, ckpt_path):
    x_norm, x_mean, x_max = bps.normalize(x_scan.reshape([1, -1, 3]), max_rescale=False, return_scalers=True, verbose=False)
    x_bps = bps.encode(x_norm, radius=BPS_RADIUS, n_bps_points=N_BPS_POINTS, bps_cell_type='dists', verbose=False)

    model = MeshRegressorMLP(n_features=N_BPS_POINTS)
    model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
    model.eval()

    x_align = model(torch.Tensor(x_bps)).detach().numpy()
    x_align /= MESH_SCALER
    x_align += x_mean

    return x_align[0]


def save_obj(mesh_verts, mesh_faces, save_path, face_colors=[0, 128, 255]):
    mesh_preds = trimesh.Trimesh(vertices=mesh_verts, faces=mesh_faces, face_colors=face_colors)

    res = mesh_preds.export(save_path)

    print("predicted alignment saved at: %s" % save_path)

    return


def save_visualisations(save_path, x_orig, x_proc, x_align, smpl_faces, scan2mesh_dist, scan_path,
                        max_points_to_show=100000):
    fig = plt.figure(figsize=(25, 10))

    fig1 = fig.add_subplot(1, 3, 1, projection='3d')
    fig1.axis('off')
    fig1.set_xlim(np.min(x_align[:, 0]), np.max(x_align[:, 0]))
    fig1.set_ylim(np.min(x_align[:, 1]), np.max(x_align[:, 1]))
    fig1.view_init(elev=90, azim=-90)
    x_orig_show = x_orig[np.random.choice(len(x_orig), max_points_to_show)]
    fig1.scatter(x_orig_show[:, 0], x_orig_show[:, 1], x_orig_show[:, 2], s=0.1, c=x_orig_show[:, 3:])
    plt.title('Input Scan', size=30)

    # fig1 = fig.add_subplot(1, 4, 2, projection='3d')
    # fig1.axis('off')
    # fig1.set_xlim(np.min(x_align[:, 0]), np.max(x_align[:, 0]))
    # fig1.set_ylim(np.min(x_align[:, 1]), np.max(x_align[:, 1]))
    # fig1.view_init(elev=90, azim=-90)
    # plt.scatter(x_proc[:, 0], x_proc[:, 1], s=1.0, c='lightgreen')
    # plt.title('Downsampled & Denoised', size=30)

    fig1 = fig.add_subplot(1, 3, 2, projection='3d')
    fig1.axis('off')
    fig1.set_xlim(np.min(x_align[:, 0]), np.max(x_align[:, 0]))
    fig1.set_ylim(np.min(x_align[:, 1]), np.max(x_align[:, 1]))
    fig1.view_init(elev=90, azim=-90)
    fig1.plot_trisurf(x_align[:, 0], x_align[:, 1], x_align[:, 2], triangles=smpl_faces, color=[0, 0.5, 1.0])
    plt.title('Predicted Alignment', size=30)

    fig1 = fig.add_subplot(1, 3, 3, projection='3d')
    fig1.axis('off')
    fig1.set_xlim(np.min(x_align[:, 0]), np.max(x_align[:, 0]))
    fig1.set_ylim(np.min(x_align[:, 1]), np.max(x_align[:, 1]))
    fig1.view_init(elev=90, azim=-90)

    fig1.scatter(x_orig_show[:, 0], x_orig_show[:, 1], x_orig_show[:, 2], s=0.1, c=x_orig_show[:, 3:])
    fig1.plot_trisurf(x_align[:, 0], x_align[:, 1], x_align[:, 2], triangles=smpl_faces, color=[0, 0.5, 1.0])
    plt.title('Overlay (scan2mesh: %3.1f mms)' % scan2mesh_dist, size=30)

    fig.suptitle("%s" % scan_path, size=15, fontweight="bold", ha='left', y=0.15, x=0.05)
    fig.tight_layout()
    plt.savefig(save_path, dpi=100, transparent=True)
    plt.close()

    print("predictions visualisations saved at: %s" % save_path)

    return


def process_scan(scan_path, ckpt_path, out_dir):

    print("processing %s" % scan_path)

    scan_name = ntpath.basename(scan_path).split('.')[0]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    obj_save_path = os.path.join(out_dir, scan_name + '.obj')
    img_save_path = os.path.join(out_dir, scan_name + '.png')

    print("loading and denoising scan..")
    x_orig, x_proc = load_scan(scan_path)

    print("predicting alignment..")
    x_align = get_alignment(x_proc, ckpt_path)

    save_obj(mesh_verts=x_align, mesh_faces=SMPL_FACES, save_path=obj_save_path)
    scan2mesh_dist = MESH_SCALER * chamfer_distance(x_align, x_proc, direction='y_to_x')

    print("scan2mesh distance: %3.1f mms" % scan2mesh_dist)
    if scan2mesh_dist > 40:
        print("warning: scan2mesh distance is quite high, make sure that your scan is in the canonical orientation "
              "(scanned body is facing z axis, use demo_scan.ply for calibration). You can also experiment with "
              "DBSCAN_EPS, DBSCAN_MIN_SAMPLES for input scan initial denoising")

    save_visualisations(img_save_path, x_orig, x_proc, x_align, SMPL_FACES, scan2mesh_dist, scan_path,
                        max_points_to_show=MAX_POINTS_TO_SHOW)

    return


def main():

    scan_path = sys.argv[1]
    out_dir = sys.argv[2]

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if os.path.isfile(scan_path):
        if scan_path.split('.')[-1] == 'ply':
            process_scan(scan_path, CKPT_PATH, out_dir)
        else:
            raise ValueError("%s is not a PLY file!" % scan_path)
    else:
        import glob
        scans = glob.glob(os.path.join(scan_path, '*.ply'))
        print("found %d scans in %s folder" % (len(scans), scan_path))
        for scan in scans:
            process_scan(scan, CKPT_PATH, out_dir)
    return


if __name__ == '__main__':
    main()