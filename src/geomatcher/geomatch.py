# -*- coding: utf-8 -*-
"""

@author: Adrien Wehrlé, EO-IO

"""

import functools
import math
import time
import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
from scipy.spatial import cKDTree
from typing import Callable
import xarray as xr
from osgeo import gdal, gdalconst
from osgeo_utils import gdal_merge


def timer(function: Callable):
    """Get the processing time of a decorated function.

    :param function: Function to monitor.
    :type function: Callable

    :return: Wrapper timer.
    :rtype: Callable
    """

    @functools.wraps(function)
    def wrapper_timer(*args, **kwargs):
        """Wrap the timer"""

        # start the counter
        start_time = time.perf_counter()

        # run the function
        result = function(*args, **kwargs)

        # stop the counter
        end_time = time.perf_counter()

        # compute run time in seconds
        run_time = end_time - start_time

        # print run time with the approriate unit
        if run_time > 60:
            run_time /= 60
            print(f"Finished {function.__name__!r} in {run_time:.4f} minutes")

        else:
            print(f"Finished {function.__name__!r} in {run_time:.4f} seconds")

        return result

    return wrapper_timer


@timer
def get_grid_from_geotiff_file(geotiff_filename: str) -> np.ndarray:
    """Create a grid from a GeoTIFF file, to be used thereafter for data
    matching.

    :param geotiff_filename: Path to the GeoTIFF file.
    :type geotiff_filename: str

    :return: A 3D array of superimposed 2D arrays for x, y and pixel
      values.
    :rtype: numpy.ndarray
    """

    # open and read raster file
    data_reader = rasterio.open(geotiff_filename)
    data = data_reader.read(1)

    # create associated row and column matrices
    cols, rows = np.meshgrid(np.arange(np.shape(data)[1]), np.arange(np.shape(data)[0]))

    # extract pixel positions at centroids
    x_coords, y_coords = data_reader.xy(rows, cols)

    # compile coordinates and pixel values in 3D array
    grid = np.dstack((x_coords, y_coords, data))

    return grid


@timer
def get_grid_from_nc_file(netcdf_filename: str, variable_name: str) -> np.ndarray:
    """Create a grid from a NetCDF file, to be used thereafter for data
    matching.

    :param netcdf_filename: Path to the NetCDF file.
    :type netcdf_filename: str

    :param variable_name: Variable from which to extract pixel values in `grid`.
    :type variable_name: str

    :return: A 3D array of superimposed 2D arrays for x, y and pixel
      values.
    :rtype: numpy.ndarray

    """

    # candidate names for x and y variables
    x_variables = ["x", "lat", "latitude"]
    y_variables = ["y", "lon", "longitude"]

    # open data set
    dataset = xr.open_dataset(netcdf_filename)

    #
    for x_variable, y_variable in zip(x_variables, y_variables):

        variable_not_found = False

        try:
            x_coords = dataset[x_variable].values
            y_coords = dataset[y_variable].values
            break

        except KeyError:
            variable_not_found = True

    if variable_not_found:
        raise KeyError("Spatial variables not found in nc file")

    x_coords_tD, y_coords_tD = np.meshgrid(x_coords, y_coords)

    grid = np.dstack((x_coords_tD, y_coords_tD, dataset[variable_name][0, :, :].values))

    return grid


@timer
def convert_grid_coordinates(
    grid: np.ndarray, in_projection: str, out_projection: str
) -> np.ndarray:
    """Convert grid coordinates to a projection in meters that can be used
      for data matching.

    :param grid: A 3D array of two superimposed 2D arrays with x and y
      coordinates.
    :type grid: numpy.ndarray

    :param in_projection: Current EPSG code of `grid`.
    :type in_projection : str

    :param out_projection: EPSG code to use for the reprojection of
      `grid`.
    :type out_projection: str

    :return: A 3D array of the same size as `grid` containing the
      reprojected x and y coordinates.
    :rtype: numpy.ndarray

    """

    # format projections for pyproj
    inProj = f"epsg:{in_projection}"
    outProj = f"epsg:{out_projection}"

    # create in-out transformation
    transformation = Transformer.from_crs(inProj, outProj, always_xy=True)

    # transform grid coordinates
    x_coords, y_coords = transformation.transform(grid[:, :, 0], grid[:, :, 1])

    # compile coordinates and pixel values back in 3D array
    reprojected_grid = np.dstack((x_coords, y_coords, grid[:, :, 2]))

    return reprojected_grid


@timer
def match_p2m(
    reference_grid: np.ndarray, points: np.ndarray
) -> pd.core.frame.DataFrame:
    """Data matching between a set of points and a matrix, mapping points
    to reference_grid.

    Warning: reference_grid should be geoferenced in a projection with
    meter unit. If this is not the case, consider using
    geomap.convert_grid_coordinates().

    :param points: A 2D array [n rows * 3 columns] with n the number
      of points and three columns for x, y, and pixel values (x and
      y must be in meters).
    :type grid: numpy.ndarray

    :param reference_grid: A 3D array of two superimposed 2D arrays
      with x and y coordinates, used as reference.
    :type grid: numpy.ndarray

    :return: A DataFrame containing the x, y coordinates of the points
      and the matching matrix locations as well as the corresponding
      point and pixel values.
    :rtype: pandas.core.frame.DataFrame

    """

    point_coords = points[:, :2]
    matrix_coords = np.column_stack(
        (reference_grid[:, :, 0].ravel(), reference_grid[:, :, 1].ravel())
    )

    btree = cKDTree(matrix_coords)
    distances, indexes = btree.query(point_coords, k=1)

    matching_matrix_values = reference_grid[:, :, 2].ravel()[indexes]

    p2m_results = pd.DataFrame(
        {
            "x_p": points[:, 0],
            "y_p": points[:, 1],
            "val_p": points[:, 2],
            "x_m": matrix_coords[:, 0][indexes],
            "y_m": matrix_coords[:, 1][indexes],
            "val_m": matching_matrix_values,
        }
    )

    return p2m_results


@timer
def match_m2m(
    reference_grid: np.ndarray, match_grid: np.ndarray, only_indexes: bool = False
) -> np.ndarray:
    """Data matching between two matrices, mapping match_grid to
    reference_grid.

    Warning: grids should be geoferenced in a projection with meter
    unit. If this is not the case, consider using
    geomap.convert_grid_coordinates().

    :param reference_grid: A 3D array of two superimposed 2D arrays
      with x and y coordinates, used as reference.
    :type grid: numpy.ndarray

    :param match_grid: A 3D array of two superimposed 2D arrays
      with x and y coordinates, to be transformed.
    :type match_grid: numpy.ndarray

    :param only_indexes: If True, only output matching indexes of
      reference_grid when flattened, and not raster values. Default is
      False.
    :param only_indexes: boolean, optional

    :return: A 3D array of two superimposed 2D arrays containing the
      pixels values associated to reference_grid and the matching
      pixel values of match_grid.
    :rtype: numpy.ndarray

    """

    reference_coords = np.column_stack(
        (reference_grid[:, :, 0].ravel(), reference_grid[:, :, 1].ravel())
    )

    match_coords = np.column_stack(
        (match_grid[:, :, 0].ravel(), match_grid[:, :, 1].ravel())
    )

    btree = cKDTree(match_coords)
    distances, indexes = btree.query(reference_coords, k=1)

    matching_sg_values = match_grid[:, :, 2].ravel()[indexes]

    tD_matching_sg_values = matching_sg_values.reshape(
        np.shape(reference_grid)[0], np.shape(reference_grid)[1]
    )

    if only_indexes:
        m2m_results = indexes.reshape(
            np.shape(reference_grid)[0], np.shape(reference_grid)[1]
        )
    else:
        m2m_results = np.dstack((reference_grid[:, :, 2], tD_matching_sg_values))

    return m2m_results


@timer
def match_gt2gt(reference_gtiff_file, match_gtiff_file):
    """Data matching between two GeoTIFF files using Rasterio, mapping
    match_gtiff_file to reference_gtiff_file.

    Parameters
    ----------
    reference_gtiff_file : TYPE
        Path to the GeoTIFF file to be used as reference.
    match_gtiff_file : TYPE
        Path to the GeoTIFF file to be matched to reference_gtiff_file

    Returns
    -------
    gt2gt_results: numpy.ndarray
        A 3D array of two superimposed 2D arrays containing the pixels values
        associated to reference_gtiff_file and the matching pixel
        values of match_gtiff_file.

    """

    rt_reader = rasterio.open(match_gtiff_file)
    rt_data = rt_reader.read(1)

    cols, rows = np.meshgrid(
        np.arange(np.shape(rt_data)[1]), np.arange(np.shape(rt_data)[0])
    )

    rt_x_coords, rt_y_coords = rt_reader.xy(rows.flatten(), cols.flatten())

    mt_reader = rasterio.open(reference_gtiff_file)
    mt_data = mt_reader.read(1)

    matching_cols, matching_rows = rasterio.transform.rowcol(
        mt_reader.transform, rt_x_coords, rt_y_coords
    )

    tD_matching_values = mt_data[matching_cols, matching_rows].reshape(
        np.shape(rt_data)[0], np.shape(rt_data)[1]
    )

    gt2gt_results = np.dstack((rt_data, tD_matching_values))

    return gt2gt_results


@timer
def resample_m2m(
    reference_grid, match_grid, reference_grid_resolution, match_grid_resolution
):
    """

    Data matching combined with a spatial resampling. To use when
    reference_grid has a lower resolution than match_grid.

    Warning: grids should be geoferenced in a projection with meter unit. If
    this is not the case, consider using geomap.convert_grid_coordinates().

    Parameters
    ----------
    reference_grid : numpy.ndarray
        A 3D array of superimposed 2D arrays for x, y and pixel values.
    match_grid : numpy.ndarray
        A 3D array of superimposed 2D arrays for x, y and pixel values.
    reference_grid_resolution : int or flat
        The spatial resolution of reference_grid, in meters.
    match_grid_resolution : int or flat
        The spatial resolution of match_grid, in meters.

    Returns
    -------
    m2m_resampling_results:
        A 3D array of two superimposed 2D arrays containing the pixels values
        associated to reference_grid and the matching pixel values of
        the resampled match_grid.

    """

    m2m_results = match_m2m(reference_grid, match_grid, only_indexes=True)

    indexes = m2m_results.ravel()

    # compute the maximum distance between two pixel centroids (45-degree
    # angle) in the new grid resolution
    distance_threshold = np.sqrt(2) * match_grid_resolution

    # compute the ratio of resolution between grid_resolution and
    # new_grid_resolution and round it up if non-integer (giving a
    # conservative number of neighbours in that case)
    resolution_ratio = math.ceil(match_grid_resolution / reference_grid_resolution)

    # compute the number of neighbors to search based on resolution_ratio
    # for eight-connected pixels
    number_of_neighbours = np.sum(
        [8 * connect_order for connect_order in range(1, resolution_ratio + 1)]
    )

    # prepare inputs for CKDTree
    reference_coords = np.column_stack(
        (
            reference_grid[:, :, 0].ravel()[indexes],
            reference_grid[:, :, 1].ravel()[indexes],
        )
    )

    # run the kd-tree for quick nearest-neighbor lookup
    btree = cKDTree(reference_coords)
    distances_rs, indexes_rs = btree.query(reference_coords, k=number_of_neighbours)

    # extract pixel values for each neighbour sets
    raw_values = reference_grid[:, :, 2].ravel()[indexes_rs]

    # mask values associated to distances higher than distance_threshold
    raw_values[distances_rs > distance_threshold] = np.nan

    # compute mean pixel values in the kernel of diamete distance_threshold
    resampled_values = np.nanmean(raw_values, axis=1)

    # reshape to 2D
    tD_resampled_values = resampled_values.reshape(
        np.shape(reference_grid)[0], np.shape(reference_grid)[1]
    )

    # stack to 3D
    m2m_resampling_results = np.dstack((reference_grid[:, :, 2], tD_resampled_values))

    return m2m_resampling_results


@timer
def resample_m(grid, grid_resolution, new_grid_resolution):
    """

    Spatial resampling of a single grid to a given resolution.

    ----------
    grid : numpy.ndarray
        A 3D array of superimposed 2D arrays for x, y and pixel values.
    grid_resolution : int or flat
        The current spatial resolution of grid, in meters.
    grid_resolution : int or flat
        The spatial resolution of grid to use for spatial resampling, in
        meters.

    Returns
    -------
    m2m_resampling_results:
        A 3D array of two superimposed 2D arrays containing the raw and
        resampled pixel values of grid.

    """

    # compute the maximum distance between two pixel centroids (45-degree
    # angle) in the new grid resolution
    distance_threshold = np.sqrt(2) * new_grid_resolution

    # compute the ratio of resolution between grid_resolution and
    # new_grid_resolution and round it up if non-integer (giving a
    # conservative number of neighbours in that case)
    resolution_ratio = math.ceil(new_grid_resolution / grid_resolution)

    # compute the number of neighbors to search based on resolution_ratio
    # for eight-connected pixels
    number_of_neighbours = np.sum(
        [8 * resolution for resolution in range(1, resolution_ratio + 1)]
    )

    # prepare inputs for CKDTree
    reference_coords = np.column_stack(
        (
            grid[:, :, 0].ravel(),
            grid[:, :, 1].ravel(),
        )
    )

    # run the kd-tree for quick nearest-neighbor lookup
    btree = cKDTree(reference_coords)
    distances_rs, indexes_rs = btree.query(reference_coords, k=number_of_neighbours)

    # extract pixel values for each neighbour sets
    raw_values = grid[:, :, 2].ravel()[indexes_rs]

    # mask values associated to distances higher than distance_threshold
    raw_values[distances_rs > distance_threshold] = np.nan

    # compute mean pixel values in the kernel of diameter distance_threshold
    resampled_values = np.nanmean(raw_values, axis=1)

    # reshape to 2D
    tD_resampled_values = resampled_values.reshape(np.shape(grid)[0], np.shape(grid)[1])

    # stack in 3D
    m2m_resampling_results = np.dstack((grid[:, :, 2], tD_resampled_values))

    return m2m_resampling_results


def raster_match(source, target, output):

    # source
    src_filename = source
    src = gdal.Open(src_filename, gdalconst.GA_ReadOnly)
    src_proj = src.GetProjection()

    # raster to match
    match_filename = target
    match_ds = gdal.Open(match_filename, gdalconst.GA_ReadOnly)
    match_proj = match_ds.GetProjection()
    match_geotrans = match_ds.GetGeoTransform()
    wide = match_ds.RasterXSize
    high = match_ds.RasterYSize

    # output/destination
    dst_filename = output
    dst = gdal.GetDriverByName("Gtiff").Create(
        dst_filename, wide, high, 1, gdalconst.GDT_Float32
    )
    dst.SetGeoTransform(match_geotrans)
    dst.SetProjection(match_proj)
    dst.GetRasterBand(1).SetNoDataValue(0)

    # run
    gdal.ReprojectImage(src, dst, src_proj, match_proj, gdalconst.GRA_NearestNeighbour)

    del dst  # Flus

    return None


def raster_merge(input_files_path, output_file_path):

    parameters = (
        ["", "-o", output_file_path]
        + ["-n", "0.0"]
        + input_files_path
        + ["-co", "COMPRESS=LZW"]
    )

    gdal_merge.main(parameters)

    return None


def raster_reproject(input_file, output_file, output_crs):

    with rasterio.open(input_file, mode="r") as src:
        transform, width, height = calculate_default_transform(
            src.crs, output_crs, src.width, src.height, *src.bounds
        )  # features transform
        kwargs = src.meta.copy()  # create features for dst
        kwargs.update(
            {
                "crs": output_crs,
                "transform": transform,
                "width": width,
                "height": height,
            }
        )  # update dst features

        # write new file: new extension, projection, compression
        with rasterio.open(output_file, "w", **kwargs, compress="deflate") as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=output_crs,
                resampling=Resampling.nearest,
            )

    return None
