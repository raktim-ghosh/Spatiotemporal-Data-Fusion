import gdal
import numpy as np
import math


def __main():

    """
    Frame of Landsat 8 OLI data for visualisation
    """

    b1 = gdal.Open('D:/RaktimPorjectData/ProjectData/Subset_LC0814004320180125_DN_sr/lstsubset_20180125_sr_b1.tif')
    b1_arr = b1.GetRasterBand(1).ReadAsArray().astype(float)
    b1_sr = b1_arr * 0.0001

    b4 = gdal.Open('D:/RaktimPorjectData/ProjectData/Subset_LC0814004320180125_DN_sr/lstsubset_20180125_sr_b4.tif')
    b4_arr = b4.GetRasterBand(1).ReadAsArray().astype(float)
    b4_sr = b4_arr * 0.0001

    b5 = gdal.Open('D:/RaktimPorjectData/ProjectData/Subset_LC0814004320180125_DN_sr/lstsubset_20180125_sr_b5.tif')
    b5_arr = b5.GetRasterBand(1).ReadAsArray().astype(float)
    b5_sr = b5_arr * 0.0001

    b6 = gdal.Open('D:/RaktimPorjectData/ProjectData/Subset_LC0814004320180125_DN_sr/lstsubset_20180125_sr_b6.tif')
    b6_arr = b6.GetRasterBand(1).ReadAsArray().astype(float)
    b6_sr = b6_arr * 0.0001

    b7 = gdal.Open('D:/RaktimPorjectData/ProjectData/Subset_LC0814004320180125_DN_sr/lstsubset_20180125_sr_b7.tif')
    b7_arr = b7.GetRasterBand(1).ReadAsArray().astype(float)
    b7_sr = b7_arr * 0.0001

    dn_b4 = gdal.Open('D:/RaktimPorjectData/ProjectData/Subset_LC0814004320180125_DN/lstsubset_b4_dn_20180125.tif')
    b4_dn = dn_b4.GetRasterBand(1).ReadAsArray().astype(float)

    dn_b5 = gdal.Open('D:/RaktimPorjectData/ProjectData/Subset_LC0814004320180125_DN/lstsubset_b5_dn_20180125.tif')
    b5_dn = dn_b5.GetRasterBand(1).ReadAsArray().astype(float)

    dn_b6 = gdal.Open('D:/RaktimPorjectData/ProjectData/Subset_LC0814004320180125_DN/lstsubset_b6_dn_20180125.tif')
    b6_dn = dn_b6.GetRasterBand(1).ReadAsArray().astype(float)

    dn_b7 = gdal.Open('D:/RaktimPorjectData/ProjectData/Subset_LC0814004320180125_DN/lstsubset_b7_dn_20180125.tif')
    b7_dn = dn_b7.GetRasterBand(1).ReadAsArray().astype(float)

    dn_b10 = gdal.Open('D:/RaktimPorjectData/ProjectData/Subset_LC0814004320180125_DN/lstsubset_b10_dn_20180125.tif')
    b10_dn = dn_b10.GetRasterBand(1).ReadAsArray().astype(float)

    """
    This function will determine the confidence map of all the pixels inside of an algorithm
    """
    shrt_tmp = short_wave_temperature(b7_dn)
    thermal_tmp = thermal_infrared_temperature(b10_dn)
    ndfi_arr = normalized_difference_fire_index(b5_sr, b6_sr, b7_sr)
    schroeder_arr = coal_fire_pixels(b7_sr, b6_sr, b5_sr)

    new_img = probability_map(shrt_tmp, thermal_tmp, ndfi_arr, schroeder_arr)
    outfile = 'Confidence_map'
    arr_to_img(new_img, b1, outfile)


def short_wave_temperature(b7_dn):

    """
    This function will generate the pixel integrated radiant temperature map as well as the kinetic temperature
    map of the each and every pixel of Landsat 8 OLI data. It will help identify the thermally anomalous ambiguous
    pixel

         Args:
             b7_dn : This is the array of at-satellite digital number of channel 7 (Landsat 8 OLI  data)

         Returns:
             It will return the radiance of band 7

    """
    M = 5.2875E-04
    A = -2.64377
    b7_rad = M * b7_dn + A

    rows = np.shape(b7_dn)[0]
    cols = np.shape(b7_dn)[1]

    h = 6.626 * (10 ** (-34))
    c = 3 * (10 ** 8)
    k = 1.381 * (10 ** (-23))
    w = 2.2005 * (10 ** (-6))
    c1 = ((h * c) / (k * w))
    c2 = 2 * h * c ** 2 * w ** (-5) * 10 ** (-6)

    temp_arr = np.zeros((rows, cols))
    shrt_thrshld = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            try:
                temp_arr[i, j] = (c1 / (math.log((c2 / b7_rad[i, j]) + 1))) / (0.97 ** 0.25) - 273
                if temp_arr[i, j] > 230.0:
                    shrt_thrshld[i, j] = 1.0
                else:
                    shrt_thrshld[i, j] = 0.0

            except:
                temp_arr[i, j] = 120.00
                shrt_thrshld[i, j] = 0.0

    print(b7_rad[:3, :3], ' radiance')
    return shrt_thrshld


def thermal_infrared_temperature(b10_dn):

    """
       This function will generate the pixel integrated radiant temperature map as well as the kinetic temperature
       map of the each and every pixel of Landsat 8 OLI data. It will help identify the thermally anomalous ambiguous
       pixel

            Args:
                b10_dn : This is the array of at-satellite digital number of channel 7 (Landsat 8 OLI  data)

            Returns:
                It will return the radiance of band 7

    """
    M = 3.3420E-04
    A = 0.10000
    spct_rad = M * b10_dn + A

    h = 6.627 * 10 ** (-34)
    c = 3 * 10 ** 8
    k = 1.38 * 10 ** (-23)
    w = 10.895 * 10 ** (-6)
    c1 = ((h * c) / (k * w))
    c2 = 2 * h * c ** 2 * w ** (-5) * 10 ** (-6)
    rows = np.shape(spct_rad)[0]
    cols = np.shape(spct_rad)[1]

    temp_arr = np.zeros((rows, cols))
    tirs_thrshld = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            try:
                temp_arr[i, j] = (c1 / (math.log((c2 / spct_rad[i, j]) + 1))) / (0.9 ** 0.25) - 273
                if temp_arr[i, j] > 40.0:
                    tirs_thrshld[i, j] = 1.0
                else:
                    tirs_thrshld[i, j] = 0.0

            except:
                temp_arr[i, j] = 23.00
                tirs_thrshld[i, j] = 0.0

    print(np.amin(temp_arr))

    return tirs_thrshld


def normalized_difference_fire_index(b5_sr, b6_sr, b7_sr):

    """
    This is the fire index personally developed by me after seeing an exponential trend of fire affected pixels

        Args:
            b5_sr : The surface reflectance value of band 5 of Landsat 8 OLI data
            b6_sr : The surface reflectance value of band 6 of Landsat 8 OLI data
            b7_sr : The surface reflectance value of band 7 of Landsat 8 OLI data

        Returns:
            return a binary classified image using NDFI index

    """
    rows = np.shape(b6_sr)[0]
    cols = np.shape(b6_sr)[1]

    ndfi_arr = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if b6_sr[i, j] + b7_sr[i, j] == 0:
                ndfi_arr[i, j] = 1
            elif b5_sr[i, j] + b6_sr[i, j] == 0:
                ndfi_arr[i, j] = 1
            else:
                i1 = (b7_sr[i, j] - b6_sr[i, j]) / (b7_sr[i, j] + b6_sr[i, j])
                i2 = (b6_sr[i, j] - b5_sr[i, j]) / (b6_sr[i, j] + b5_sr[i, j])
                if i1 > 0 and i2 > 0 and i1 - i2 > 0:
                    ndfi_arr[i, j] = 1.0
                elif i1 > 0 and i2 > 0 and (i1 > 0.85 or i2 > 0.85):
                    ndfi_arr[i, j] = 1.0
                else:
                    ndfi_arr[i, j] = 0.0

    return ndfi_arr


def coal_fire_pixels(b7_sr, b6_sr, b5_sr):
    """
    This function will typically extract out potential coal fire pixels using the background statistics
    depicted in the equation (6) of Active fire detection using Landsat 8 OLI data

        Args:
            b7_sr : This is the surface reflectance array of band 7 of landsat 8 OLI data
            b6_sr : This is the surface reflectance array of band 6 of landsat 8 OLI data
            b5_sr : This is the surface reflectance array of band 5 of landsat 8 OLI data

        Returns:
            This function will return the binary classified map of potentially fire affected pixels

    """
    rows = np.shape(b7_sr)[0]
    cols = np.shape(b7_sr)[1]

    ratio_arr = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            ratio_arr[i, j] = b7_sr[i, j] / b5_sr[i, j]

    x = 0
    fire_arr = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            '''
            if b7_sr[i, j] / b5_sr[i, j] > 2.5 and b7_sr[i, j] - b5_sr[i, j] > 0.3 and b7_sr[i, j] > 0.5:
                fire_arr[i, j] = 0
            else:
                fire_arr[i, j] = 20000
            '''
            if b7_sr[i, j] / b5_sr[i, j] > 1.8 and b7_sr[i, j] - b5_sr[i, j] > 0.17:
                if i - 30 >= 0 and j - 30 >= 0 and rows - i >= 30 and cols - j >= 30:
                    temp_mean = np.mean(ratio_arr[i - 30:i + 31, j - 30:j + 31])
                    temp_std = np.std(ratio_arr[i - 30:i + 31, j - 30:j + 31])
                    temp_max = [(3 * temp_std), 0.8]
                    if ratio_arr[i, j] > temp_mean + max(temp_max):
                        b7_mean = np.mean(b7_sr[i - 30:i + 31, j - 30:j + 31])
                        b7_std = np.std(b7_sr[i - 30:i + 31, j - 30:j + 31])
                        b7_max = [(3 * b7_std), 0.08]
                        if b7_sr[i, j] > b7_mean + max(b7_max):
                            if b7_sr[i, j] / b6_sr[i, j] > 1.6:
                                fire_arr[i, j] = 1.0
                                x += 1
                    print('the number of potential fire pixels', x)

                else:
                    fire_arr[i, j] = 0.0
            else:
                fire_arr[i, j] = 0.0

    return fire_arr


def probability_map(shrt_tmp, thermal_tmp, ndfi_arr, schroeder_arr):

    """
    This is the fire index personally developed by me after seeing an exponential trend of fire affected pixels

        Args:
            shrt_tmp : the shortwave temperature map by putting a temperature threshold
            thermal_tmp: the tirs temperature map by putting a temperature threshold
            ndfi_arr : ndfi_arr temperature threshold map hby putting a ndfi threshold
            schroeder_arr : the algorithm has been applied to develop the particular threshold

        Returns:
            return a classified map of the pixel

    """
    rows = np.shape(shrt_tmp)[0]
    cols = np.shape(shrt_tmp)[1]
    prob_arr = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            p1, p2 = shrt_tmp[i, j], thermal_tmp[i, j]
            p3, p4 = ndfi_arr[i, j], schroeder_arr[i, j]

            if p1 == 1.0 and p2 == 1.0 and p3 == 1.0 and p4 == 1.0:
                prob_arr[i, j] = 100.0

            elif p1 == 1.0 and p2 == 1.0 and p3 == 1.0 and p4 == 0.0:
                prob_arr[i, j] = 75.0
            elif p1 == 0.0 and p2 == 1.0 and p3 == 1.0 and p4 == 1.0:
                prob_arr[i, j] = 75.0
            elif p1 == 1.0 and p2 == 0.0 and p3 == 1.0 and p4 == 1.0:
                prob_arr[i, j] = 75.0
            elif p1 == 1.0 and p2 == 1.0 and p3 == 0.0 and p4 == 1.0:
                prob_arr[i, j] = 75.0

            elif p1 == 1.0 and p2 == 1.0 and p3 == 0.0 and p4 == 0.0:
                prob_arr[i, j] = 50.0
            elif p1 == 1.0 and p2 == 0.0 and p3 == 1.0 and p4 == 0.0:
                prob_arr[i, j] = 50.0
            elif p1 == 1.0 and p2 == 0.0 and p3 == 0.0 and p4 == 1.0:
                prob_arr[i, j] = 50.0
            elif p1 == 0.0 and p2 == 1.0 and p3 == 1.0 and p4 == 0.0:
                prob_arr[i, j] = 50.0
            elif p1 == 0.0 and p2 == 1.0 and p3 == 0.0 and p4 == 1.0:
                prob_arr[i, j] = 50.0
            elif p1 == 0.0 and p2 == 0.0 and p3 == 1.0 and p4 == 1.0:
                prob_arr[i, j] = 50.0

            elif p1 == 0.0 and p2 == 1.0 and p3 == 0.0 and p4 == 0.0:
                prob_arr[i, j] = 25.0
            elif p1 == 0.0 and p2 == 0.0 and p3 == 1.0 and p4 == 0.0:
                prob_arr[i, j] = 25.0
            elif p1 == 0.0 and p2 == 0.0 and p3 == 0.0 and p4 == 1.0:
                prob_arr[i, j] = 25.0

    return prob_arr


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