from functools import partial
import multiprocessing
import numpy as np
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


def normalize(x, known_scalers=False, x_mean=None, x_max=None, max_rescale=True, return_scalers=False, verbose=False):
    """Normalize point clouds to fit a unit sphere

    Parameters
    ----------
    x : [n_clouds, n_points, n_dims]
        Input point clouds
    known_scalers: bool
        if True, use provided x_mean and x_max scalers for normalization
    max_rescale: bool
        if False, normalization will only include shifting by the mean value
    x_mean : numpy array [n_clouds, n_dims]
        if provided, mean value for every cloud used for normalization
    x_max : None or numpy array [n_clouds, 1]
        if provided, max norm for every cloud used for normalization
    return_scalers: bool
        whether to return point cloud scalers (needed for denormalisation)
    verbose: bool
        whether to print progress
    Returns
    -------
    x_norm : numpy array [n_clouds, n_points, n_dims]
        Normalized point clouds
    x_mean : numpy array [n_clouds, n_dims]
        Mean value of every cloud
    x_max : numpy array [n_clouds, 1]
        Max norm of every cloud
    """

    def _normalize_cloud(x, x_mean=None, x_max=None, max_rescale=True):
        """normalize single cloud"""

        if x_mean is None:
            x_mean = np.mean(x, axis=0)

        x_norm = np.copy(x - x_mean)

        # note: max norm could be not robust to outliers!
        if x_max is None:
            if max_rescale:
                x_max = np.max(np.sqrt(np.sum(np.square(x), axis=1)))
            else:
                x_max = 1.0
        x_norm = x_norm / x_max

        return x_norm, x_mean, x_max

    n_clouds, n_points, n_dims = x.shape

    x_norm = np.zeros([n_clouds, n_points, n_dims])

    if known_scalers is False:
        x_mean = np.zeros([n_clouds, n_dims])
        x_max = np.zeros([n_clouds, 1])

    fid_lst = range(0, n_clouds)

    if verbose:
        fid_lst = tqdm(fid_lst)

    for pid in fid_lst:
        if known_scalers is False:
            x_norm[pid], x_mean[pid], x_max[pid] = _normalize_cloud(x[pid])
        else:
            x_norm[pid], _, _ = _normalize_cloud(x[pid], x_mean[pid], x_max[pid], max_rescale=max_rescale)

    if return_scalers:
        return x_norm, x_mean, x_max
    else:
        return x_norm


def denormalize(x_norm, x_mean, x_max):
    """Denormalize point clouds

    Parameters
    ----------
    x : [n_clouds, n_points, n_dims]
        Input point clouds
    rescale: bool
        if False, normalization will only include shifting by the mean value
    x_mean : numpy array [n_clouds, n_dims]
        if provided, mean value for every cloud used for normalization
    x_max : None or numpy array [n_clouds, 1]
        if provided, max norm for every cloud used for normalization

    Returns
    -------
    x_norm : numpy array [n_clouds, n_points, n_dims]
        Normalized point clouds
    x_mean : numpy array [n_clouds, n_dims]
        Mean value of every cloud
    x_max : numpy array [n_clouds, 1]
        Max norm of every cloud
    """

    def _denormalize_cloud(x_norm, x_mean, x_max):
        """denormalize single cloud"""

        x_denorm = x_norm * x_max + x_mean

        return x_denorm

    x_denorm = np.zeros(x_norm.shape)

    for pid in range(0, len(x_norm)):
        x_denorm[pid] = _denormalize_cloud(x_norm[pid], x_mean[pid], x_max[pid])

    return x_denorm


def generate_random_basis(n_points=1000, n_dims=3, radius=1.0, random_seed=13):
    """Sample uniformly from d-dimensional unit ball

    The code is inspired by this small note:
    https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html

    Parameters
    ----------
    n_points : int
        number of samples
    n_dims : int
        number of dimensions
    radius: float
        ball radius
    random_seed: int
        random seed for basis point selection
    Returns
    -------
    x : numpy array
        points sampled from d-ball
    """
    np.random.seed(random_seed)
    # sample point from d-sphere
    x = np.random.normal(size=[n_points, n_dims])
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1])
    x_unit = x / x_norms

    # now sample radiuses uniformly
    r = np.random.uniform(size=[n_points, 1])
    u = np.power(r, 1.0 / n_dims)
    x = radius * x_unit * u
    np.random.seed(None)

    return x


def generate_grid_basis(grid_size=32, n_dims=3, minv=-1.0, maxv=1.0):
    """ Generate d-dimensional grid BPS basis

    Parameters
    ----------
    grid_size: int
        number of elements in each grid axe
    minv: float
        minimum element of the grid
    maxv
        maximum element of the grid

    Returns
    -------
    basis: numpy array [grid_size**n_dims, n_dims]
        n-d grid points
    """

    linspaces = [np.linspace(minv, maxv, num=grid_size) for d in range(0, n_dims)]
    coords = np.meshgrid(*linspaces)
    basis = np.concatenate([coords[i].reshape([-1, 1]) for i in range(0, n_dims)], axis=1)

    return basis


def encode(x, bps_arrangement='random', n_bps_points=512, radius=1.5, bps_cell_type='dists',
           verbose=1, random_seed=13, x_features=None, custom_basis=None, n_jobs=-1):
    """Converts point clouds to basis point set (BPS) representation, multi-processing version

    Parameters
    ----------
    x: numpy array [n_clouds, n_points, n_dims]
        batch of point clouds to be converted
    bps_arrangement: str
        supported BPS arrangements: "random", "grid", "custom"
    n_bps_points: int
        number of basis points
    radius: float
        radius for BPS sampling area
    bps_cell_type: str
        type of information stored in every BPS cell. Supported:
            'dists': Euclidean distance to the nearest point in cloud
            'deltas': delta vector from basis point to the nearest point
            'closest': closest point itself
            'features': return features of the closest point supplied by x_features.
                        e.g. RGB values of points, surface normals, etc.
    verbose: boolean
        whether to show conversion progress
    x_features: numpy array [n_clouds, n_points, n_features]
        point features that will be stored in BPS cells if return_values=='features'
    custom_basis: numpy array [n_basis_points, n_dims]
        custom basis to use
    n_jobs: int
        number of parallel jobs used for encoding. If -1, use all available CPUs

    Returns
    -------
    x_bps: [n_clouds, n_points, n_bps_features]
        point clouds converted to BPS representation.
    """

    if n_jobs == -1:
        n_jobs = multiprocessing.cpu_count()

    if n_jobs == 1:

        n_clouds, n_points, n_dims = x.shape

        if bps_arrangement == 'random':
            basis_set = generate_random_basis(n_bps_points, n_dims=n_dims, radius=radius, random_seed=random_seed)
        elif bps_arrangement == 'grid':
            # in case of a grid basis, we need to find the nearest possible grid size
            grid_size = int(np.round(np.power(n_bps_points, 1 / n_dims)))
            basis_set = generate_grid_basis(grid_size=grid_size, minv=-radius, maxv=radius)
        elif bps_arrangement == 'custom':
            # in case of a grid basis, we need to find the nearest possible grid size
            if custom_basis is not None:
                basis_set = custom_basis
            else:
                raise ValueError("Custom BPS arrangement selected, but no custom_basis provided.")
        else:
            raise ValueError("Invalid basis type. Supported types: \'random\', \'grid\', \'custom\'")

        n_bps_points = basis_set.shape[0]

        if bps_cell_type == 'dists':
            x_bps = np.zeros([n_clouds, n_bps_points])
        elif bps_cell_type == 'deltas':
            x_bps = np.zeros([n_clouds, n_bps_points, n_dims])
        elif bps_cell_type == 'closest':
            x_bps = np.zeros([n_clouds, n_bps_points, n_dims])
        elif bps_cell_type == 'features':
            n_features = x_features.shape[2]
            x_bps = np.zeros([n_clouds, n_bps_points, n_features])
        else:
            raise ValueError("Invalid cell type. Supported types: \'dists\', \'deltas\', \'closest\', \'features\'")
        fid_lst = range(0, x.shape[0])

        if verbose:
            fid_lst = tqdm(fid_lst)

        for fid in fid_lst:
            nbrs = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm="ball_tree").fit(x[fid])
            fid_dist, npts_ix = nbrs.kneighbors(basis_set)
            if bps_cell_type == 'dists':
                x_bps[fid] = fid_dist.squeeze()
            elif bps_cell_type == 'deltas':
                x_bps[fid] = x[fid][npts_ix].squeeze() - basis_set
            elif bps_cell_type == 'closest':
                x_bps[fid] = x[fid][npts_ix].squeeze()
            elif bps_cell_type == 'features':
                x_bps[fid] = x_features[fid][npts_ix].squeeze()

        return x_bps

    else:

        if verbose:
            print("using %d available CPUs for BPS encoding.." % n_jobs)

        bps_encode_func = partial(encode, bps_arrangement=bps_arrangement, n_bps_points=n_bps_points, radius=radius,
                                  bps_cell_type=bps_cell_type, verbose=verbose, random_seed=random_seed,
                                  x_features=x_features, custom_basis=custom_basis, n_jobs=1)

        pool = multiprocessing.Pool(n_jobs)
        x_chunks = np.array_split(x, n_jobs)
        x_bps = np.concatenate(pool.map(bps_encode_func, x_chunks), 0)
        pool.close()

        return x_bps
