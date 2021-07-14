#!/usr/bin/env python
# coding: utf-8

import os
import glob
import uuid
import datetime
import warnings
import collections
import traceback
import argparse

import leroi
import pyart
import cftime
import netCDF4
import numpy as np
import matplotlib.pyplot as pl

from concurrent.futures import TimeoutError
from pebble import ProcessPool, ProcessExpired


def chunks(l, n: int):
    """
    Yield successive n-sized chunks from l.
    From http://stackoverflow.com/a/312464
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]
        
        
def get_dtype():
    keytypes = {
        "air_echo_classification": np.int16,
        "radar_echo_classification": np.int16,
        "corrected_differential_phase": np.float32,
        "corrected_differential_reflectivity": np.float32,
        "corrected_reflectivity": np.float32,
        "corrected_specific_differential_phase": np.float32,
        "corrected_velocity": np.float32,
        "cross_correlation_ratio": np.float32,
        "normalized_coherent_power": np.float32,
        "radar_estimated_rain_rate": np.float32,
        "signal_to_noise_ratio": np.float32,
        "spectrum_width": np.float32,
        "total_power": np.float32,
    }

    return keytypes


def update_metadata(radar, longitude: np.ndarray, latitude: np.ndarray):
    """
    Update metadata of the gridded products.

    Parameter:
    ==========
    radar: pyart.core.Grid
        Radar data.

    Returns:
    ========
    metadata: dict
        Output metadata dictionnary.
    """
    today = datetime.datetime.utcnow()
    dtime = cftime.num2pydate(radar.time["data"], radar.time["units"])

    maxlon = longitude.max()
    minlon = longitude.min()
    maxlat = latitude.max()
    minlat = latitude.min()

    metadata = {
        "comment": "Gridded radar volume using Barnes et al. ROI",
        "field_names": ", ".join([k for k in radar.fields.keys()]),
        "geospatial_bounds": f"POLYGON(({minlon:0.6} {minlat:0.6},{minlon:0.6} {maxlat:0.6},{maxlon:0.6} {maxlat:0.6},{maxlon:0.6} {minlat:0.6},{minlon:0.6} {minlat:0.6}))",
        "geospatial_lat_max": f"{maxlat:0.6}",
        "geospatial_lat_min": f"{minlat:0.6}",
        "geospatial_lat_units": "degrees_north",
        "geospatial_lon_max": f"{maxlon:0.6}",
        "geospatial_lon_min": f"{minlon:0.6}",
        "geospatial_lon_units": "degrees_east",
        "geospatial_vertical_min": np.int32(radar.origin_altitude["data"][0]),
        "geospatial_vertical_max": np.int32(20000),
        "geospatial_vertical_positive": "up",
        "history": f"created by Valentin Louf on gadi.nci.org.au at {today.isoformat()} using Py-ART",
        "processing_level": "b2",
        "time_coverage_start": dtime[0].isoformat(),
        "time_coverage_end": dtime[-1].isoformat(),
        "uuid": str(uuid.uuid4()),
    }

    return metadata


def update_variables_metadata(grid):
    """
    Update metadata of the gridded variables.

    Parameter:
    ==========
    grid: pyart.core.Grid
        Gridded radar data.

    Returns:
    ========
    grid: pyart.core.Grid
        Gridded radar data with updated variables metadata.
    """
    try:
        grid.fields["corrected_velocity"]["standard_name"] = "radial_velocity_of_scatterers_away_from_instrument"
    except KeyError:
        pass

    grid.radar_latitude["standard_name"] = "latitude"
    grid.radar_latitude["coverage_content_type"] = "coordinate"
    grid.radar_longitude["standard_name"] = "longitude"
    grid.radar_longitude["coverage_content_type"] = "coordinate"
    grid.radar_altitude["standard_name"] = "altitude"
    grid.radar_altitude["coverage_content_type"] = "coordinate"
    grid.radar_time["standard_name"] = "time"
    grid.radar_time["coverage_content_type"] = "coordinate"

    grid.point_latitude["standard_name"] = "latitude"
    grid.point_latitude["coverage_content_type"] = "coordinate"
    grid.point_longitude["standard_name"] = "longitude"
    grid.point_longitude["coverage_content_type"] = "coordinate"
    grid.point_altitude["standard_name"] = "altitude"
    grid.point_altitude["coverage_content_type"] = "coordinate"

    return grid


def mkdir(dirpath: str) -> None:
    """
    Create directory. Check if directory exists and handles error.
    """
    if not os.path.exists(dirpath):
        # Might seem redundant, but the multiprocessing creates error.
        try:
            os.mkdir(dirpath)
        except FileExistsError:
            pass

    return None


def leroy_grid(radar, coords, field):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        return leroi.cressman_ppi_interp(
            radar,
            coords, field,
            k=200,
            verbose=False,
            filter_its = 0,
            corr_lens = (800, 2000),
            multiprocessing=False
        )


def gridfile(infile) -> None:
    good_keys = GOOD_KEYS
    output_directory = OUTDIR
    radar = pyart.io.read(infile)

    # Update radar dtype:
    radar.altitude["data"] = radar.altitude["data"].astype(np.float64)
    radar.longitude["data"] = radar.longitude["data"].astype(np.float64)
    radar.latitude["data"] = radar.latitude["data"].astype(np.float64)

    keys = [*radar.fields.keys()]
    for k in keys:
        if k not in good_keys:
            _ = radar.fields.pop(k)

    radar_date = cftime.num2pydate(radar.time["data"][0], radar.time["units"])
    year = str(radar_date.year)
    datestr = radar_date.strftime("%Y%m%d")
    datetimestr = radar_date.strftime("%Y%m%d_%H%M")

    # radar.fields["corrected_reflectivity"]["data"] += 1.1

    outpath = os.path.join(output_directory, year)
    mkdir(outpath)
    outpath = os.path.join(outpath, datestr)
    mkdir(outpath)

    outfilename = f"502_{datetimestr}00_grid.nc"
    outfilename = os.path.join(outpath, outfilename)
    if os.path.exists(outfilename):
        print(f"Output file already exists {outfilename}")
        return None

    gs = (41, 301, 301)
    gb = ((0, 20000), (-150000, 150000), (-150000, 150000))
    center_pos = (0, 0, 0)
    lon0, lat0 = radar.longitude['data'][0], radar.latitude['data'][0]

    x = np.linspace(gb[2][0],gb[2][1], gs[2])
    y = np.linspace(gb[1][0],gb[1][1], gs[1])
    z = np.linspace(gb[0][0],gb[0][1], gs[0])
    coords = (z - center_pos[0], y - center_pos[1], x - center_pos[2])

    gridded_fields = leroy_grid(radar, coords, [*radar.fields.keys()])

    grid = leroi.build_pyart_grid(radar, gridded_fields, gs, gb)

    keytypes = get_dtype()
    for k, v in keytypes.items():
        try:
            if grid.fields[k]["data"].dtype != v:
                grid.fields[k]["data"] = grid.fields[k]["data"].astype(v)
        except KeyError:
            pass

    # Metadata
    lon_data, lat_data = grid.get_point_longitude_latitude(0)
    metadata = update_metadata(grid, longitude=lon_data, latitude=lat_data)
    for k, v in metadata.items():
        grid.metadata[k] = v
    grid.metadata["title"] = f"Gridded radar volume on a 300x300x20km grid"
    grid = update_variables_metadata(grid)
    # A-Z order.
    metadata = grid.metadata
    grid.metadata = collections.OrderedDict(sorted(metadata.items()))

    # Saving data.
    if outfilename is not None:
        pyart.io.write_grid(outfilename, grid, arm_time_variables=True, write_point_lon_lat_alt=False)
        print(f"{outfilename} written.")
        # append ROI and lat/long 2D grids and update metadata
        with netCDF4.Dataset(outfilename, "a") as ncid:
            nclon = ncid.createVariable("longitude", np.float32, ("y", "x"), zlib=True, least_significant_digit=2)
            nclon[:] = lon_data
            nclon.units = "degrees_east"
            nclon.standard_name = "longitude"
            nclon.long_name = "longitude_degrees_east"
            nclat = ncid.createVariable("latitude", np.float32, ("y", "x"), zlib=True, least_significant_digit=2)
            nclat[:] = lat_data
            nclat.units = "degrees_north"
            nclat.standard_name = "latitude"
            nclat.long_name = "latitude_degrees_north"

        del grid
    return None


def buffer(infile):
    try:
        gridfile(infile)
    except Exception:
        traceback.print_exc()

    return None


def main():    
    inpath = os.path.join(INDIR, "**", "*.nc")
    flist = glob.glob(inpath)
    if len(flist) == 0:
        print(f"No file found in {inpath}. Doing nothing")
        return None
    
    for flist_chunk in chunks(flist, 32):
        with ProcessPool() as pool:
            future = pool.map(buffer, flist_chunk, timeout=120)
            iterator = future.result()

            while True:
                try:
                    _ = next(iterator)
                except StopIteration:
                    break
                except TimeoutError as error:
                    print("function took longer than %d seconds" % error.args[1])
                except ProcessExpired as error:
                    print("%s. Exit code: %d" % (error, error.exitcode))
                except Exception:
                    traceback.print_exc()

    return None


if __name__ == "__main__":
    GOOD_KEYS = [
#         "air_echo_classification",
#         "corrected_differential_phase",
#         "corrected_differential_reflectivity",
        "corrected_reflectivity",
#         "corrected_specific_differential_phase",
        "corrected_velocity",
        "velocity",
#         "cross_correlation_ratio",
#         "normalized_coherent_power",
#         "radar_echo_classification",
#         "radar_estimated_rain_rate",
#         "signal_to_noise_ratio",
#         "spectrum_width",
        "total_power",
    ]
    parser_description = """Grid content of directory using LeROI"""
    parser = argparse.ArgumentParser(description=parser_description)
    parser.add_argument("-i", "--input", dest="indir", type=str, help="Input directory", required=True)
    parser.add_argument("-o", "--output", dest="outdir", type=str, help="Output directory.", default="/scratch/kl02/vhl548/opol/")
    args = parser.parse_args()
    OUTDIR = args.outdir
    INDIR = args.indir

    main()






