import numpy as np
import pyart
import warnings
from scipy.spatial import cKDTree
import multiprocessing as mp
from scipy.signal import savgol_filter
from scipy.interpolate import RegularGridInterpolator

def cressman_ppi_interp(radar, coords, Rc, field_names, k = 100, filter_its = 0, verbose = True,**kwargs):#mu =5, poly =3
    """
    Interpolate multiple fields from a radar object to a grid. This 
    is an implementation of the method described in Dahl et. al. (2019).
    
    Inputs:
    radar (Pyart object): radar to be interpolated
    coords (tuple): tuple of 3 1d arrays containing z, y, x of grid
    Rc (float): Cressman radius of interpolation
    field_names (list or string): field names in radar to interpolate
    k (int): max number of points within a radius of influence
    filter_its (int): number of filter iterations for the low-pass filter
    kwargs passed to the scipy.savgol_filter function
    
    """
    if type(field_names) != list:
        field_names = [field_names,]
    
    fields = []
    dims = [len(coord) for coord in coords]
    Z, Y, X = np.meshgrid(coords[0], coords[1], coords[2], indexing = 'ij')
    
    ppi_height = interpolate_to_ppi(radar, coords, Rc, 'height', k=k)
    

    
    if ppi_height.mask.sum() > 0:
        warnings.warn("""\n There are invalid height values which will 
        ruin the linear interpolation. This most likely means the radar
        doesnt cover the entire gridded domain""")
    
    if verbose:
        print('Interpolating...')
    for field in field_names:
        ppis = interpolate_to_ppi(radar, coords, Rc, field, k=k)
        grid = interp_along_axis(ppis.filled(np.nan), ppi_height, Z, axis=0, method='linear')
        
        
        if filter_its > 0:
            if verbose:
                print('Filtering...')
            
            interpolator = RegularGridInterpolator(coords, grid)
            smooth = interpolator(np.c_[Z.ravel(),Y.ravel(),X.ravel()]).reshape(dims)
            
            for i in range(filter_its):
#                 smooth = filter3D(smooth, mu, poly, **kwargs)
                smooth = savgol_filter(smooth, axis=0, **kwargs)
                smooth = savgol_filter(smooth, axis=1, **kwargs)
                smooth = savgol_filter(smooth, axis=2, **kwargs)
                
            grid = smooth.copy()
            
        fields.append(np.ma.masked_array(grid, mask = np.isnan(grid)))
    if verbose:
        print('Done!')
    if len(fields) > 1:
        return fields
    else:
        return fields[0]


def interpolate_to_ppi(radar, coords, Rc, field, k=100, fill_ground = True):
    """
    A function for interpolating radar fields to ppi surfaces in 
    Cartesian coordinates. 
    
    Inputs:
    radar (Pyart object): radar to be interpolated
    coords (tuple): tuple of 3 1d arrays containing z, y, x of grid
    Rc (float): Cressman radius of interpolation
    field (string): field name in radar or 'height' for altitude
    k (int): max number of points within a radius of influence
    fill_ground (bool): If 0 elevation scan, set lowest ppi to the ground (z=0)
    """
    # setup stuff
    nsweeps=radar.nsweeps
    slices = []
    elevations = radar.fixed_angle['data']
    Y,X = np.meshgrid(coords[1], coords[2], indexing='ij')
    
    # loop through grid and define data, no mask for height data
    for i in range(nsweeps):
        x,y,z = radar.get_gate_x_y_z(i)
        if field=='height':
            data = z.ravel()
            dmask = np.zeros(data.shape).astype('bool')
        else:
            dmask = radar.fields[field]['data'].mask[radar.get_slice(i)].ravel()
            data = radar.get_field(i, field).ravel()[~dmask]
        
        # define a lookup tree for the horizontal coordinates
        tree = cKDTree(np.c_[y.ravel()[~dmask],x.ravel()[~dmask]])
        d, idx = tree.query(np.c_[Y.ravel(),X.ravel()], k=k, distance_upper_bound=Rc, workers = mp.cpu_count())
        
        # set invalid indicies to 0 to avoid errors, they are masked out by the weights anyway
        idx[idx==len(data)] = 0
        
        # check that there aren't more than k points within radius on lowest sweep
        if i == 0:
            ball_idx = tree.query_ball_point(np.c_[Y.ravel(),X.ravel()], Rc, workers = mp.cpu_count())
            lens = np.array([len(x) for x in ball_idx])
            if (lens>k).sum() > 0:
                warnings.warn("\n Some points are being left out of radius of influence, make 'k' bigger!")
                
        # do all of the weighting stuff based on kdtree distance
        d[np.isinf(d)] = Rc+1e3
        d2, r2 = d**2, Rc**2
        w = (r2 - d2) / (r2 + d2)
        w[w<0] = 0
        sw = np.sum(w, axis=1)
        valid = sw!=0
        
        # put valid data into a resultant array and reshape to model grid
        slce = np.zeros(sw.shape)           
        if ((field == 'height') and (fill_ground) and (elevations[i] == 0)):
            pass
        elif len(data)==0:
            pass
        else:
            slce[valid] = np.sum(data[idx]*w, axis = 1)[valid] / sw[valid]
        slce = np.ma.masked_array(slce, mask = ~valid)
        slices.append(slce.reshape((len(coords[1]), len(coords[2]))))
    
    # stack ppis into model grid shape
    ppis = np.ma.asarray(slices)
    return ppis

def interp_along_axis(y, x, newx, axis, inverse=False, method='linear'):
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
        warnings.warn("Some values are outside the interpolation range. "
                      "These will be filled with NaN")
    if np.any(np.diff(_x, axis=0) < 0):
        raise ValueError('x should increase monotonically')
    if np.any(np.diff(_newx, axis=0) < 0):
        raise ValueError('newx should increase monotonically')
    # Cubic interpolation needs the gradient of y in addition to its values
    if method == 'cubic':
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
        needs_update = (xi > x_upper) & (i_upper+1<len(_x))
        # print x_upper.max(), np.any(needs_update)
        while np.any(needs_update):
            i_lower = np.where(needs_update, i_lower+1, i_lower)
            i_upper = i_lower + 1
            x_lower = _x[[i_lower]+ind]
            x_upper = _x[[i_upper]+ind]
            # Check again
            needs_update = (xi > x_upper) & (i_upper+1<len(_x))
        # Express the position of xi relative to its neighbours
        xj = (xi-x_lower)/(x_upper - x_lower)
        # Determine where there is a valid interpolation range
        within_bounds = (_x[0, ...] < xi) & (xi < _x[-1, ...])
        if method == 'linear':
            f0, f1 = _y[[i_lower]+ind], _y[[i_upper]+ind]
            a = f1 - f0
            b = f0
            newy[i, ...] = np.where(within_bounds, a*xj+b, np.nan)
        elif method=='cubic':
            f0, f1 = _y[[i_lower]+ind], _y[[i_upper]+ind]
            df0, df1 = ydx[[i_lower]+ind], ydx[[i_upper]+ind]
            a = 2*f0 - 2*f1 + df0 + df1
            b = -3*f0 + 3*f1 - 2*df0 - df1
            c = df0
            d = f0
            newy[i, ...] = np.where(within_bounds, a*xj**3 + b*xj**2 + c*xj + d, np.nan)
        else:
            raise ValueError("invalid interpolation method"
                             "(choose 'linear' or 'cubic')")
    if inverse:
        newy = newy[::-1, ...]
    return np.moveaxis(newy, 0, axis)