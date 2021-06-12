import time
import warnings
import multiprocessing as mp

import pyart
import numpy as np
from scipy.spatial import cKDTree
from scipy.signal import savgol_filter
from scipy.interpolate import RegularGridInterpolator
from astropy.convolution import convolve
from pyart.config import get_metadata

def get_data_mask(radar, fields, gatefilter=None):
    """
    Create a mask for the gridding algorithm by combining
    all masks from selected radar fields and optionally a
    pyart gatefilter

    Parameters:
    -----------
    radar : (object)
        Pyart radar object
    fields : (list)
        List of fields to accumulate the mask from
    gatefilter : (object)
        A pyart gatefilter
        
    Returns:
    --------
    mask : (np.array)
        Data mask same shape as radar fields, 1 = masked
    """
    
    mask = np.ones((radar.nrays, radar.ngates)).astype('bool')
    
    # combine data masks
    for field in fields:
        mask *= ~radar.fields[field]['data'].mask
        
    # combine gatefilter    
    if gatefilter is not None:
        mask *= ~gatefilter.gate_excluded
    return ~mask

def get_leroy_roi(radar, coords, frac=0.55):
    """
    Get a radius of influence for the ppis based on the azimuthal spacing of each sweep
    
    Refer to Dahl et al (2019) for details here. 
    """
    roi = 0
    rmax = np.sqrt(max(coords[0]) ** 2 + max(coords[1]) ** 2 + max(coords[2]) ** 2)
    sort_idx = np.argsort(radar.fixed_angle['data'])
    for i in sort_idx:
        az = np.amax(np.radians(np.amax(np.diff(np.sort(radar.azimuth["data"][radar.get_slice(i)])))))
        r = frac * az * rmax
        if r > roi:
            roi = r
    return roi

def calculate_ppi_heights(radar, coords, Rc, ground_elevation = -9999):
    slices = []
    elevations = np.sort(radar.fixed_angle["data"])

    Y, X = np.meshgrid(coords[1], coords[2], indexing = 'ij')
    #sort sweep index to process from lowest sweep and ascend
    sort_idx = np.argsort(radar.fixed_angle['data'])
    for i in sort_idx:
        x, y, z = radar.get_gate_x_y_z(i)
        data = z.ravel()
        tree = cKDTree(np.c_[y.ravel(), x.ravel()])
        d, idx = tree.query(np.c_[Y.ravel(), X.ravel()], k=10, distance_upper_bound=Rc, workers=mp.cpu_count())
        idx[idx == len(data)] = 0

        # do all of the weighting stuff based on kdtree distance
        d[np.isinf(d)] = Rc + 1e3
        d2, r2 = d ** 2, Rc ** 2
        w = (r2 - d2) / (r2 + d2)
        w[w < 0] = 0
        sw = np.sum(w, axis=1)
        valid = sw != 0

        # put valid data into a resultant array and reshape to model grid
        slce = np.zeros(sw.shape)
        if ((i==0) and (elevations[i] <= ground_elevation)):
            pass
        elif len(data) == 0:
            pass
        else:
            slce[valid] = np.sum(data[idx] * w, axis=1)[valid] / sw[valid]
        slce = np.ma.masked_array(slce, mask=~valid)
        slices.append(slce.reshape((len(coords[1]), len(coords[2]))))
        
    #sort lists by order of sweeps in radar    
    return np.ma.asarray(slices)
    

def smooth_grid(grid, coords,kernel,corr_lens,filter_its, verbose):
    
    dz = np.mean(np.diff(np.sort(coords[0])))
    dy = np.mean(np.diff(np.sort(coords[1])))
    dx = np.mean(np.diff(np.sort(coords[2])))
    dh = np.mean((dy, dx))
    
    if verbose:
        print("Filtering...")

    if kernel is None:
        if corr_lens == None:
            raise NotImplementedError(
                """You must either input a convolution kernel 
                ('kernel') or some correlation lengths ('corr_len')."""
            )
        v_window = int(np.ceil(corr_lens[0] / dz) // 2 * 2 + 1)
        h_window = int(np.ceil(corr_lens[1] / dh) // 2 * 2 + 1)
        kernel = np.ones((v_window, h_window, h_window)) / np.float(v_window * h_window * h_window)

    smooth = grid.copy()
    for i in range(filter_its):
        smooth = convolve(smooth, kernel, boundary="extend")

    return smooth.copy()

def interp_along_axis(y, x, newx, axis, inverse=False, method="linear"):
    """ Linear interpolation with irregular grid, from:
    https://stackoverflow.com/questions/28934767/best-way-to-interpolate-a-numpy-ndarray-along-an-axis
    
    Interpolate vertical profiles, e.g. of atmospheric variables
    using vectorized numpy operations
    This function assumes that the x-coordinate increases monotonically
    Peter Kalverla
    March 2018
    --------------------
    More info:
    Algorithm from: http://www.paulinternet.nl/?page=bicubic
    """
    # View of x and y with axis as first dimension
    if inverse:
        _x = np.moveaxis(x, axis, 0)[::-1, ...]
        _y = np.moveaxis(y, axis, 0)[::-1, ...]
        _newx = np.moveaxis(newx, axis, 0)[::-1, ...]
    else:
        _y = np.moveaxis(y, axis, 0)
        _x = np.moveaxis(x, axis, 0)
        _newx = np.moveaxis(newx, axis, 0)

    # Sanity checks
    if np.any(_newx[0] < _x[0]) or np.any(_newx[-1] > _x[-1]):
        # raise ValueError('This function cannot extrapolate')
        warnings.warn("Some values are outside the interpolation range. " "These will be filled with NaN")
    if np.any(np.diff(_x, axis=0) < 0):
        raise ValueError("x should increase monotonically")
    if np.any(np.diff(_newx, axis=0) < 0):
        raise ValueError("newx should increase monotonically")
    # Cubic interpolation needs the gradient of y in addition to its values
    if method == "cubic":
        # For now, simply use a numpy function to get the derivatives
        # This produces the largest memory overhead of the function and
        # could alternatively be done in passing.
        ydx = np.gradient(_y, axis=0, edge_order=2)
    # This will later be concatenated with a dynamic '0th' index
    ind = [i for i in np.indices(_y.shape[1:])]
    # Allocate the output array
    original_dims = _y.shape
    newdims = list(original_dims)
    newdims[0] = len(_newx)
    newy = np.zeros(newdims)
    # set initial bounds
    i_lower = np.zeros(_x.shape[1:], dtype=int)
    i_upper = np.ones(_x.shape[1:], dtype=int)
    x_lower = _x[0, ...]
    x_upper = _x[1, ...]
    for i, xi in enumerate(_newx):
        # Start at the 'bottom' of the array and work upwards
        # This only works if x and newx increase monotonically
        # Update bounds where necessary and possible
        needs_update = (xi > x_upper) & (i_upper + 1 < len(_x))
        # print x_upper.max(), np.any(needs_update)
        while np.any(needs_update):
            i_lower = np.where(needs_update, i_lower + 1, i_lower)
            i_upper = i_lower + 1
            x_lower = _x[[i_lower] + ind]
            x_upper = _x[[i_upper] + ind]
            # Check again
            needs_update = (xi > x_upper) & (i_upper + 1 < len(_x))
        # Express the position of xi relative to its neighbours
        xj = (xi - x_lower) / (x_upper - x_lower)
        # Determine where there is a valid interpolation range
        within_bounds = (_x[0, ...] < xi) & (xi < _x[-1, ...])
        if method == "linear":
            f0, f1 = _y[[i_lower] + ind], _y[[i_upper] + ind]
            a = f1 - f0
            b = f0
            newy[i, ...] = np.where(within_bounds, a * xj + b, np.nan)
        elif method == "cubic":
            f0, f1 = _y[[i_lower] + ind], _y[[i_upper] + ind]
            df0, df1 = ydx[[i_lower] + ind], ydx[[i_upper] + ind]
            a = 2 * f0 - 2 * f1 + df0 + df1
            b = -3 * f0 + 3 * f1 - 2 * df0 - df1
            c = df0
            d = f0
            newy[i, ...] = np.where(within_bounds, a * xj ** 3 + b * xj ** 2 + c * xj + d, np.nan)
        else:
            raise ValueError("invalid interpolation method" "(choose 'linear' or 'cubic')")
    if inverse:
        newy = newy[::-1, ...]
    return np.moveaxis(newy, 0, axis)


def setup_interpolate(radar, coords, dmask, Rc, k=200, verbose = True, multiprocess=False):
    """
    A function for interpolating radar fields to ppi surfaces in 
    Cartesian coordinates. 
    
    Inputs:
    radar (Pyart object): radar to be interpolated
    coords (tuple): tuple of 3 1d arrays containing z, y, x of grid
    Rc (float): Cressman radius of interpolation
    field (string): field name in radar or 'height' for altitude
    k (int): max number of points within a radius of influence
    multiprocess (logical): True enables multiprocessing on KDtree query
    """
    t0 = time.time()
    # setup stuff
    nsweeps = radar.nsweeps
    weights, idxs = [], []
    elevations = np.sort(radar.fixed_angle["data"])
    Y, X = np.meshgrid(coords[1], coords[2], indexing="ij")
    trim = 0
    model_idxs = -np.ones((nsweeps, len(coords[1])*len(coords[2])))
    sws, model_lens = [], []
    
    # loop through grid and define data, no mask for height data
    for i in range(nsweeps):
        x, y, z = radar.get_gate_x_y_z(i)
        mask = ~dmask[radar.get_slice(i)].flatten()

        # dont bother with ckdtree if there's no data
        if mask.sum() == 0:
            weights.append(np.empty((1,1)))
            idxs.append(np.empty((1,1)))
            sws.append(np.zeros(len(coords[1])*len(coords[2])))
            model_lens.append(0)
            continue
            
        # define a lookup tree for the horizontal coordinates
        valid_radar_points = np.c_[y.ravel()[mask], x.ravel()[mask]]
        ndata = valid_radar_points.shape[0]
        tree = cKDTree(valid_radar_points)
        if multiprocess:
            ncpu = mp.cpu_count()
        else:
            ncpu = 1
        d, idx = tree.query(np.c_[Y.ravel(), X.ravel()], k=k, distance_upper_bound=Rc, workers=ncpu)
        
        # check if any kth weight is valid, and trim if possible
        valid = ~(idx == ndata)
        kidx = max(np.where(valid==1)[1])+1
        
        if valid[:,-1].sum() > 0:
            warnings.warn("\n Some points are being left out of radius of influence, make 'k' bigger!")
        
        # set invalid indicies to 0 to avoid errors, they are masked out by the weights anyway
        idx[idx == ndata] = 0

        # do all of the weighting stuff based on kdtree distance
        d[np.isinf(d)] = Rc + 1e3
        d2, r2 = d ** 2, Rc ** 2
        w = (r2 - d2) / (r2 + d2)
        w[w < 0] = 0
        sw = np.sum(w, axis=1)
        model_idx = np.where(sw!=0)[0]
        model_idxs[i][:len(model_idx)] = model_idx #use c index to preserve sweep order that's in radar objects

        weights.append(w[model_idx,:kidx])
        idxs.append(idx[model_idx,:kidx])
        model_lens.append(len(model_idx))
        sws.append(sw)
    
    # stack weights 
    return weights, idxs, model_idxs[:,:max(model_lens)].astype(int), np.array(sws), model_lens

def cressman_ppi_interp(radar, coords, field_names, gatefilter = None, Rc=None, k=100, 
                        filter_its=0, verbose=True, kernel=None, corr_lens=None, ground_elevation = -999, multiprocess=False):
    """
    Interpolate multiple fields from a radar object to a grid. This 
    is an implementation of the method described in Dahl et. al. (2019).
    
    Inputs:
    radar (Pyart object): radar to be interpolated
    coords (tuple): tuple of 3 1d arrays containing z, y, x of grid
    field_names (list or string): field names in radar to interpolate
    Rc (float): Cressman radius of interpolation, calculated if not supplied
    k (int): max number of points within a radius of influence
    filter_its (int): number of filter iterations for the low-pass filter
    kernel (astropy.kernel): user defined kernel for smoothing (boxcar filter if not specified)
    corr_lens (tuple): correlation lengths for smoothing filter in vert. and horiz. dims resp.
    multiprocess (logical): True enables multiprocessing on KDtree query
    """
    t0 = time.time()

    if type(field_names) != list:
        field_names = [field_names,]

    
    dims = [len(coord) for coord in coords]

    if Rc is None:
        Rc = get_leroy_roi(radar, coords, frac=0.55)
        if verbose:
            print("Radius of influence set to {} m.".format(Rc))
            
    dmask = get_data_mask(radar, field_names)
    Rc = get_leroy_roi(radar, coords, frac=0.55)
    weights, idxs, model_idxs, sw, model_lens = setup_interpolate(radar, coords, dmask, Rc, k, multiprocess=multiprocess)
    Z, Y, X = np.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
    ppi_height = calculate_ppi_heights(radar, coords, Rc, ground_elevation = ground_elevation)
    
    if ppi_height.mask.sum() > 0:
        warnings.warn("""\n There are invalid height values which will 
        ruin the linear interpolation. This most likely means the radar
        doesnt cover the entire gridded domain""")

    fields = {}
    for field in field_names:
        ppis  = np.zeros((radar.nsweeps, dims[1]*dims[2]))
        mask  = np.ones((radar.nsweeps, dims[1]*dims[2]))
        sort_idx = np.argsort(radar.fixed_angle['data'])
        for i in sort_idx:
            slc = radar.get_slice(i)
            data = radar.fields[field]['data'].filled(0)[slc][~dmask[slc]]
            if len(data) > 0:
                ppis[i, model_idxs[i, :model_lens[i]]] = np.sum(data[idxs[i]] * weights[i], axis=1)/ sw[i, model_idxs[i, :model_lens[i]]]
                mask[i, model_idxs[i, :model_lens[i]]] = 0
                out = np.ma.masked_array(ppis.reshape((radar.nsweeps, dims[1], dims[2])), mask.reshape((radar.nsweeps, dims[1], dims[2])))
        
        grid = interp_along_axis(out.filled(np.nan), ppi_height, Z, axis=0, method="linear")

        if filter_its > 0:
                grid = smooth_grid(grid, coords,kernel,corr_lens,filter_its, verbose)
    
        #add to output dictionary
        fields[field] = {'data': np.ma.masked_array(grid, mask=np.isnan(grid))}
        # copy the metadata from the radar to the grid
        for key in radar.fields[field].keys():
            if key == 'data':
                continue
            fields[field][key] = radar.fields[field][key]
    
    if verbose:
        print('Took: ', time.time()-t0)
    
    return fields

def build_pyart_grid(radar, fields, gs, gb):
    #build pyart grid object
    
    # time dictionaries
    time = get_metadata('grid_time')
    time['data'] = np.array([radar.time['data'][0]])
    time['units'] = radar.time['units']
    # metadata dictionary
    metadata = dict(radar.metadata)
    # grid origin location dictionaries
    origin_latitude = get_metadata('origin_latitude')
    origin_longitude = get_metadata('origin_longitude')
    origin_latitude['data'] = radar.latitude['data']
    origin_longitude['data'] = radar.longitude['data']
    origin_altitude = get_metadata('origin_altitude')
    origin_altitude['data'] = radar.altitude['data']
    # grid coordinate dictionaries
    nz, ny, nx = gs
    (z0, z1), (y0, y1), (x0, x1) = gb
    x = get_metadata('x')
    x['data'] = np.linspace(x0, x1, nx)
    y = get_metadata('y')
    y['data'] = np.linspace(y0, y1, ny)
    z = get_metadata('z')
    z['data'] = np.linspace(z0, z1, nz)
    # create radar_ dictionaries
    radar_latitude = get_metadata('radar_latitude')
    radar_latitude['data'] = radar.latitude['data']
    radar_longitude = get_metadata('radar_longitude')
    radar_longitude['data'] = radar.longitude['data']
    radar_altitude = get_metadata('radar_altitude')
    radar_altitude['data'] = radar.altitude['data']
    radar_time = time
    radar_name = get_metadata('radar_name')
    name_key = 'instrument_name'
    radar_name['data'] = radar.metadata[name_key]
    if name_key in radar.metadata:
        radar_name['data'] = np.array([radar.metadata[name_key]])
    else:
        radar_name['data'] = np.array([''])
        
    return pyart.core.Grid(
        time, fields, metadata,
        origin_latitude, origin_longitude, origin_altitude, x, y, z,
        radar_latitude=radar_latitude, radar_longitude=radar_longitude,
        radar_altitude=radar_altitude, radar_name=radar_name,
        radar_time=radar_time, projection=None)