# LEROI (Linear Elevation and Radius Of Influence) 🤴

A Python3 implementation of a radar gridding algorithm that uses linear interpolation in elevation and radius of influence on the PPI.

## Notes and User Parameters

### Quality Control:

Use the `qc.mask_invalid_data` function to mask radar fields. This function is effectively a despeckle filter in ppi space and is useful for cleaning up radar fields before they enter gridding algorithms. Here are some of the import parameters:

- `correlation_length`: the length of the boxcar smoothing filter along each ray, bigger length > more smoothing
- `min_field`: minimum field threshold used to identify contiguous objects in radar data 
- `min_area`: contiguous objects with areas smaller than this will be masked

### Gridding:

Use the `leroi.cressman_ppi_interp` function to grid radar data onto a cartesian grid. This function takes a radar object and returns a list of numpy.masked_arrays for each required field. Important user parameters are:

- `Rc`: Cressman radius of influence for ppi's, leave as None to be calculated as a little over half of the maximum data seperation as recommended by Dahl et. al., (2019)
- `k`: number of nearest neighbours to search for around each grid point. This should be set higher than the maximum number of data points within a radius of influence, a warning will be issued if this is too low. 
- `filter_its`: number of filter iterations after the gridding is done. Dahl et. al., (2019) recommends 2 passes of a Leise filter after gridding for best results.
- `kernel`: option to add a user-defined kernel for the `astropy.convolve` smoothing function. The default is set to a boxcar filter with user-defined correlation lengths.
- `corr_lens`: tuple containing correlation lengths in vertical and horizontal directions. 

### Potential Improvements:
 - [x] Output a PyART grid with the interpolated fields instead of numpy arrays
 - [ ] Confirm the benefits of post-gridding smoothing outlined by Dahl et. al. (2019) (refer to `post_smoothing.ipynb` for example). 
 - [ ] Make a kernel similar to the Leise filter they describe (I've only been able to find 1970's fortran code for this filter). The `scipy.savgol_filter` looks very similar (refer to `filtering.ipynb` for example). 
 - [x] Optimise

## References:
- Dahl, N. A., Shapiro, A., Potvin, C. K., Theisen, A., Gebauer, J. G., Schenkman, A. D., & Xue, M. (2019). High-Resolution, Rapid-Scan Dual-Doppler Retrievals of Vertical Velocity in a Simulated Supercell, Journal of Atmospheric and Oceanic Technology, 36(8), 1477-1500. Retrieved Jun 5, 2021, from https://journals.ametsoc.org/view/journals/atot/36/8/jtech-d-18-0211.1.xml
