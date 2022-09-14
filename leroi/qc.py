import pyart
import numpy as np
from astropy.convolution import convolve
import warnings

def smooth_ppi(radar, field, sweep, c_len):
    """
    Function that returns a smooth ppi for a radar field. Smoothing
    is performed using a convolution with a boxcar kernel, special
    care is taken to interpolate nans and deal with ray edges.
    
    radar (pyart.radar): radar object
    field (string): radar field name
    sweep (int): radar sweep no.
    c_len (float): length in meters for the boxcar kernel
    """
    window = int(np.ceil(c_len / np.mean(np.diff(radar.range["data"]))) // 2 * 2 + 1)
    data = radar.get_field(sweep, field).filled(np.nan)
    kernel = np.ones((1, window)) / float(window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smooth = convolve(data, kernel, boundary="extend")
    mask = gate_range_mask(smooth, window)
    return np.ma.masked_array(smooth, np.logical_or(np.isnan(smooth), mask))


def gate_range_mask(data, window):
    """
    Return a mask for the edges of a ppi scar
    
    data (2d array): ppi data to be masked
    window (int): number of gates to mask at 
        beggining and end of ray
    """
    window -= int(window / 2)
    end = data.shape[1]
    mask = ~np.isnan(data)
    starts = np.argmax(mask, axis=1)
    starts[starts > window] += window - 1
    ends = np.argmax(mask[:, ::-1], axis=1)
    ends = end - ends
    ends[ends < (end - window)] += 1 - window
    mask = np.ones(data.shape)
    for j, i in enumerate(zip(starts, ends)):
        mask[j, i[0] : i[1]] = 0
    return mask.astype("bool")


def _clear_small_echoes_ppi(label_image, areas, min_area):
    """
    Despeckle filter worker, gets rid of objects with less than area
    """
    small_echoes = []
    for i in range(1, np.amax(label_image) + 1):
        area = np.sum(areas[label_image == i])
        if area < min_area:
            small_echoes.append(i)
    small = np.array(small_echoes)
    for obj in small:
        label_image[label_image == obj] = 0
        label_image[label_image > obj] -= 1
        small[small > obj] -= 1
    return label_image


def mask_invalid_data(radar, field, add_to=None, correlation_length=2000, 
                      min_field=0, min_area=10, return_smooth=False):
    """
    Apply a mask to an existing or new radar field. The mask attempts to filter out 
    bad values by masking data outside of contiguous objects within ppi scans.
    
    radar (pyart object): pyar radar object
    field (string): radar field name
    add_to (list): list of field names to add the new mask to
    correlation_length (float): greater length, greater convolutional smoothing window
    min_field (float): field value threshold for valid data
    min_area (float): minimum area of cells in km^2
    return_smooth (bool): whether to create a new field with the smoothed data used for despeckling
    """
    dR = np.mean(np.diff(radar.range["data"])) / 1e3
    elevations = radar.fixed_angle['data']
    smooth_data, mask = np.ma.zeros((2, radar.nrays, radar.ngates))
    smooth_fn = field + "_smooth"
    nsweeps = radar.nsweeps
    for sweep in range(nsweeps):
        smooth_data[radar.get_slice(sweep)] = smooth_ppi(radar, field, sweep, correlation_length)
    radar.add_field_like(field, smooth_fn, smooth_data, replace_existing=True)
    for sweep in range(nsweeps):
        if elevations[sweep] == 90.0:
            continue # dont mask vertical scans
        dA = np.radians(np.mean(np.diff(np.sort(radar.azimuth["data"][radar.get_slice(sweep)]))))
        ranges = np.repeat(radar.range["data"][np.newaxis, :], radar.rays_per_sweep["data"][sweep], axis=0)
        corr = np.cos(np.radians(radar.fixed_angle["data"][sweep])) ** 2
        areas = ranges * dA * dR * corr / 1e3
        obj_dict = pyart.correct.find_objects(radar, smooth_fn, min_field, sweeps=sweep)
        label = _clear_small_echoes_ppi(obj_dict["data"].filled(0), areas, min_area)
        mask[radar.get_slice(sweep)] = label == 0
    if add_to is None:
        add_to = [
            field,
        ]
    for f in add_to:
        radar.fields[f]["data"] = np.ma.masked_array(radar.fields[f]["data"], mask)
    if not return_smooth:
        radar.fields.pop(smooth_fn)
    return radar