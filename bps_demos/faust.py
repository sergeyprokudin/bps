"""

FAUST dataset Python loader. The dataset itself can be downloaded here:
http://faust.is.tue.mpg.de/

"""

import os
import numpy as np
from tqdm import tqdm
import trimesh
import pyntcloud as pynt
import shutil

INTENSE_GREEN = [0, 255, 128]
INTENSE_BLUE = [0, 128, 255]


def get_faust_train(data_dir, n_scan_points=10000):
    """ Get FAUST train scans and ground truth SMPL registrations

    Parameters
    ----------
    data_dir : str
        path to FAUST dataset directory (../MPI-FAUST/)
    n_scan_points: int
        number of points to take from the scan

    Returns
    -------
    scans : numpy array [n_scans, n_scan_points, 3]
        scans point clouds
    meshes : numpy array [n_scans, 6890, 3]
        ground truth SMPL registrations

    """
    n_mesh_vertices = 6890
    n_train_scans = 100

    scans_path = os.path.join(data_dir, 'training/scans/')
    registrations_path = os.path.join(data_dir, 'training/registrations/')

    scans = np.zeros([n_train_scans, n_scan_points, 3])
    meshes = np.zeros([n_train_scans, n_mesh_vertices, 3])

    for fid in tqdm(range(0, n_train_scans)):
        mesh_scan = trimesh.load_mesh(os.path.join(scans_path, 'tr_scan_%03d.ply' % fid))
        x = np.asarray(mesh_scan.sample(n_scan_points))

        mesh_reg = pynt.PyntCloud.from_file(os.path.join(registrations_path, 'tr_reg_%03d.ply' % fid))
        y = mesh_reg.xyz

        scans[fid] = x
        meshes[fid] = y

    return scans, meshes


def get_faust_scan_by_id(data_dir, scan_id, part='test', mesh_lib='trimesh'):
    """Get FAUST scan by its id

    """
    if part == 'test':
        scans_path = os.path.join(data_dir, 'test/scans/')
    else:
        scans_path = os.path.join(data_dir, 'training/scans/')

    mesh_path = os.path.join(scans_path, 'test_scan_%03d.ply' % scan_id)

    mesh_scan = trimesh.load_mesh(mesh_path)

    return mesh_scan


def get_faust_test(data_dir, n_scan_points=10000):
    """ Get FAUST test scans

    Parameters
    ----------
    data_dir : str
        path to FAUST dataset directory (../MPI-FAUST/)
    n_scan_points: int
        number of points to take from the scan

    Returns
    -------
    scans : numpy array [n_scans, n_scan_points, 3]
        scans point clouds

    """
    n_test_scans = 200

    scans_path = os.path.join(data_dir, 'test/scans/')

    scans = np.zeros([n_test_scans, n_scan_points, 3])

    for fid in tqdm(range(0, n_test_scans)):
        mesh_scan = trimesh.load_mesh(os.path.join(scans_path, 'test_scan_%03d.ply' % fid))
        x = np.asarray(mesh_scan.sample(n_scan_points))

        scans[fid] = x

    return scans


def merge_meshes(mesh1, mesh2):
    """Merge two trimesh meshes for easy visualisation

    """

    verts1 = np.asarray(mesh1.vertices)
    faces1 = np.asarray(mesh1.faces)

    verts2 = np.asarray(mesh2.vertices)
    faces2 = np.asarray(mesh2.faces)

    n_faces1 = len(faces1)
    n_verts1 = len(verts1)

    vertices_pair = np.concatenate([verts1, verts2])
    faces_pair = np.concatenate([faces1, faces2 + n_verts1])
    face_colors_pair = np.zeros_like(faces_pair)
    face_colors_pair[0:n_faces1, :] += INTENSE_GREEN
    face_colors_pair[n_faces1:, :] += INTENSE_BLUE

    mesh = trimesh.Trimesh(vertices=vertices_pair, faces=faces_pair, face_colors=face_colors_pair)

    return mesh


def visualise_predictions(scan, align_verts, align_faces, scan_id, scan2mesh_loss, save_dir):

    import matplotlib
    import matplotlib.pyplot as plt
    import PIL

    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'

    mesh_scan = trimesh.Trimesh(vertices=scan.vertices, faces=scan.faces, face_colors=INTENSE_GREEN)

    mesh_preds = trimesh.Trimesh(vertices=align_verts, faces=align_faces, face_colors=INTENSE_BLUE)

    mesh_pair = merge_meshes(mesh_scan, mesh_preds)

    aligns_dir = os.path.join(save_dir, 'faust_test_alignments')
    images_dir = os.path.join(save_dir, 'faust_test_images')

    if not os.path.exists(aligns_dir):
        os.makedirs(aligns_dir)
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)

    scan_img_path = os.path.join(images_dir, '%03d_01_scan.png' % scan_id)
    alignment_img_path = os.path.join(images_dir, '%03d_02_pred.png' % scan_id)
    pair_image_path = os.path.join(images_dir, '%03d_03_pair.png' % scan_id)
    merged_img_path = os.path.join(images_dir, '%03d_scan_align_pair.png' % scan_id)
    alignment_obj_path = os.path.join(aligns_dir, '%03d_align.obj' % scan_id)

    with open(scan_img_path, 'wb') as f:
        f.write(mesh_scan.scene().save_image())

    with open(alignment_img_path, 'wb') as f:
        f.write(mesh_preds.scene().save_image())

    with open(pair_image_path, 'wb') as f:
        f.write(mesh_pair.scene().save_image())

    scan_img = np.asarray(PIL.Image.open(scan_img_path))[:, 600:1200]
    align_img = np.asarray(PIL.Image.open(alignment_img_path))[:, 600:1200]
    pair_img = np.asarray(PIL.Image.open(pair_image_path))[:, 600:1200]

    fig = plt.figure(figsize=(30, 10))

    plt.subplot(1, 3, 1)
    plt.axis('off')
    plt.title('Input Scan', size=30)
    plt.imshow(scan_img)

    plt.subplot(1, 3, 2)
    plt.axis('off')
    plt.title('Predicted Alignment', size=30)
    plt.imshow(align_img)

    plt.subplot(1, 3, 3)
    plt.axis('off')
    plt.title('Overlay (scan2mesh: %3.1f mms)' % scan2mesh_loss, size=30)
    plt.imshow(pair_img)
    fig.suptitle("FAUST Test Scan ID: %03d" % scan_id, size=15, fontweight="bold", ha='right', y=0.15, x=0.25)

    plt.savefig(merged_img_path, dpi=100)

    merged_img = PIL.Image.fromarray(np.asarray(PIL.Image.open(merged_img_path))[30:900, 450:2700, :])
    merged_img.save(merged_img_path)

    os.remove(scan_img_path)
    os.remove(alignment_img_path)
    os.remove(pair_image_path)

    res = mesh_preds.export(alignment_obj_path)

    return


def compute_faust_correspondences(alignment_verts, faust_data_path, output_dir, mesh_faces, mode='intra'):

    from psbody.mesh import Mesh

    assert (mode in ('intra', 'inter'))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    pairs_path = os.path.join(faust_data_path, 'test/challenge_pairs/{}_challenge.txt'.format(mode))
    scan_dir = os.path.join(faust_data_path, 'test/scans')

    with open(pairs_path, 'r') as f:
        for ipair, pair in tqdm(enumerate(f.readlines())):
            src_scan, dst_scan = [Mesh(filename=os.path.join(scan_dir, 'test_scan_{}.ply'.format(idx)))
                                  for idx in pair.strip().split('_')]
            src_algn, dst_algn = [Mesh(v=alignment_verts[int(pair.strip().split('_')[i])], f=mesh_faces) for i in
                                  range(2)]
            faces, pts = src_algn.closest_faces_and_points(src_scan.v)
            v_idxs, bary = src_algn.barycentric_coordinates_for_points(pts, faces)
            corrs = np.einsum('ijk,ij->ik', dst_algn.v[v_idxs], bary)
            np.savetxt(os.path.join(output_dir, '{}.txt'.format(pair.strip())), corrs)