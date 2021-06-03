import re
import pyart
from astropy.convolution import convolve
import numpy as np

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
    window = int(np.ceil(c_len/np.mean(np.diff(radar.range['data']))) // 2 * 2 + 1)
    data = radar.get_field(sweep, field).filled(np.nan)
    kernel =  np.ones((1, window))/np.float(window)
    smooth = convolve(data, kernel, boundary = 'extend')
    mask = gate_range_mask(smooth, window)
    return np.ma.masked_array(smooth, np.logical_or(np.isnan(smooth), mask))

def gate_range_mask(data, window):
    """
    Return a mask for the edges of a ppi scar
    
    data (2d array): ppi data to be masked
    window (int): number of gates to mask at 
        beggining and end of ray
    """
    window -= int(window/2)
    end = data.shape[1]
    mask = ~np.isnan(data)
    starts = np.argmax(mask, axis=1) 
    starts[starts>window] += window -1
    ends = np.argmax(mask[:,::-1], axis=1)
    ends = end-ends
    ends[ends<(end-window)] += 1 - window
    mask = np.ones(data.shape)
    for j,i in enumerate(zip(starts,ends)):
        mask[j,i[0]:i[1]] = 0
    return mask.astype('bool')

def clear_small_echoes_ppi(label_image, areas, min_area):
    small_echoes = []
    for i in range(1,np.amax(label_image)+1):
        area = np.sum(areas[label_image == i])
        if area < min_area:
            small_echoes.append(i)
    small = np.array(small_echoes)
    for obj in small:
        label_image[label_image == obj] = 0
        label_image[label_image>obj] -= 1
        small[small>obj] -= 1
    return label_image

def mask_invalid_data(radar, field, add_to = None, correlation_length = 2000, min_field = 0, min_area = 10, 
                      return_smooth = False):
    """
    Apply a mask to an existing or new radar field. The mask attempts to filter out 
    bad values by masking data outside of contiguous objects within ppi scans.
    
    radar (pyart object): pyar radar object
    field (string): radar field name
    correlation_length (float): greater length, greater convolutional smoothing window
    min_field (float): field value threshold for valid data
    min_area (float): minimum area of cells in km^2
    dilate (int): rough distance to dilate the object masks by
    replace_exsiting (bool): whether to create a new field with the mask or to add it to original
    """
    
    dR = np.mean(np.diff(radar.range['data']))/1e3
    
    smooth_data, mask = np.ma.zeros((2,radar.nrays, radar.ngates))
    smooth_fn = field+'_smooth'
    nsweeps=radar.nsweeps
    for sweep in range(nsweeps):
        smooth_data[radar.get_slice(sweep)] = smooth_ppi(radar, field, sweep, correlation_length)
    radar.add_field_like(field, smooth_fn, smooth_data, replace_existing = True)
    for sweep in range(nsweeps):
        dA = np.radians(np.mean(np.diff(np.sort(radar.azimuth['data'][radar.get_slice(sweep)]))))
        ranges = np.repeat(radar.range['data'][np.newaxis,:],radar.rays_per_sweep['data'][sweep],axis=0)
        corr = np.cos(np.radians(radar.fixed_angle['data'][sweep]))**2
        areas = ranges*dA*dR*corr/1e3
        obj_dict = pyart.correct.find_objects(radar, smooth_fn, min_field, sweeps=sweep)
        label = clear_small_echoes_ppi(obj_dict['data'].filled(0), areas, min_area)
        mask[radar.get_slice(sweep)] = label == 0
    if add_to is None:
        add_to = [field,]
    for f in add_to:
        radar.fields[f]['data'] = np.ma.masked_array(radar.fields[f]['data'], mask)        
    if not return_smooth:
        radar.fields.pop(smooth_fn)
    return radar

def dt_from_fn(fns):
    """
    Naive function to try and get the datetimes from a list filenames.
    It works by detecting 8 digit numbers (dates) then 6 or 4 digit numbers (times)
    assumed to be hhmmss or hhmm. There's a lot of cases where this will fail
    beware!!
    
    __Inputs__
    fns: List of filenames with unique 8 digit numbers for dates and unique
    6 or 4 digit numbers for times.
    """
    # Keep track of whether four or six time digits are in filename
    four = False
    
    #check if full path in fns, if so remove just filenames
    if '/' in fns[0]:
        fns = [fn.split('/')[-1] for fn in fns]
        
    # Get lists of 8 and 6 digit numbers assumed to be times in filenames
    dates = sum([re.findall(r'(?<!\d)\d{8}(?!\d)', st) for st in fns],[])
    times = sum([re.findall(r'(?<!\d)\d{6}(?!\d)', st) for st in fns],[])
    
    # If there are no 6 digit ones, check for 4 digits
    if len(times) == 0:
        times = sum([re.findall(r'(?<!\d)\d{4}(?!\d)', st) for st in fns],[])
        if len(times) != 0:
            four = True
    if len(dates) != len(times):
        raise ValueError('Filename comprehension failed! Check instructions in dt_from_fn')
        
    # Make into datetime strings
    if four:
        dt = ['-'.join((i[:4],i[4:6],i[6:])) + 'T' + ':'.join((j[:2],j[2:],'00')) for i, 
              j in zip(dates, times)]
    else:
        dt = ['-'.join((i[:4],i[4:6],i[6:])) + 'T'+ ':'.join((j[:2],j[2:4],j[4:])) for i, 
              j in zip(dates, times)]
    if len(dt) != len(fns):
        raise ValueError('Filename comprehension failed! Check instructions in dt_from_fn')
    return np.array(dt).astype('datetime64')