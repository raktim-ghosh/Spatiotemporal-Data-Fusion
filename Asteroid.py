import gdal
from numpy import *
import numpy as np
import math


def __main():
    lst_b6 = gdal.Open('D:/RaktimPorjectData/ProjectData/Subset_LC0814004320180125_DN/lstsubset_b10_dn_20180125.tif')
    b6_arr = lst_b6.GetRasterBand(1).ReadAsArray().astype(float)

    # rad_tmp = gdal.Open('D:/Output_temperature_Map/radiant_temp_map_b7.tiff')
    # temp_arr = rad_tmp.GetRasterBand(1).ReadAsArray().astype(float)
    print(b6_arr[:3, :3])
    spct_rad = spectral_radiance(b6_arr)
    temp_arr = radiant_temperature(spct_rad)
    outfile = 'Landsat_b10_temperature_map_latest'
    arr_to_img(temp_arr, lst_b6, outfile)


def spectral_radiance(b6_arr):

    M = 3.3420E-04
    A = 0.10000
    spct_rad = M * b6_arr + A
    rows = np.shape(b6_arr)[0]
    cols = np.shape(b6_arr)[1]
    # for i in range(rows):
    #     for j in range(cols):
    #         if  temp_arr[i, j] >= 255 and temp_arr[i, j] <= 257:
    #             print(b6_arr[i, j], spct_rad[i, j], temp_arr[i, j])
    #
    return spct_rad


def radiant_temperature(spct_rad):
    h = 6.627 * 10 ** (-34)
    c = 3 * 10 ** 8
    k = 1.38 * 10 ** (-23)
    w = 10.895 * 10 ** (-6)
    c1 = ((h * c) / (k * w))
    c2 = 2 * h * c ** 2 * w ** (-5) * 10 ** (-6)
    # k1 = 480.8883
    # k2 = 1201.1442
    rows = np.shape(spct_rad)[0]
    cols = np.shape(spct_rad)[1]

    temp_arr = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            try:
                temp_arr[i, j] = (c1 / (math.log((c2 / spct_rad[i, j]) + 1))) / (0.9 ** 0.25) - 273

            except:
                temp_arr[i, j] = 23.00

    print(np.amin(temp_arr))

    return temp_arr


def arr_to_img(img, lst_b7, outfile):
    """
    This function will return a GeoTiff file by converting an array to image

        Args:
            img : this is a given array to be converted to image
            lst_b7 : this is the array of surface reflectance of band 7
            outfile : this is the string name

        Returns:
            return a GeoTiff file using by conversion

    """

    cols, rows = img.shape
    driver = gdal.GetDriverByName("GTiff")
    outfile += '.tiff'
    out_data = driver.Create(outfile, rows, cols, 1, gdal.GDT_Float32)
    out_data.SetProjection(lst_b7.GetProjection())
    out_data.SetGeoTransform(lst_b7.GetGeoTransform())
    out_data.GetRasterBand(1).WriteArray(img)
    out_data.FlushCache()


if __name__ == '__main__':
    __main()
