import gdal
import numpy as np
import math


def __main():

    """
    Frame of Landsat 8 OLI data for visualisation
    """

    lst_b1 = gdal.Open('erdas_subset_b1_140043.tif')
    print('hello')
    b1_array = lst_b1.GetRasterBand(1).ReadAsArray().astype(float)
    b1_sr = 0.0001 * b1_array
    print(b1_sr[:3, :3])

    lst_b6 = gdal.Open('erdas_subset_b6_140043.tif')
    print('hello')
    b6_array = lst_b6.GetRasterBand(1).ReadAsArray().astype(float)
    b6_sr = 0.0001 * b6_array
    print(b6_sr[:3, :3])

    lst_b7 = gdal.Open('erdas_subset_b7_140043.tif')
    print('Hello')
    b7_array = lst_b7.GetRasterBand(1).ReadAsArray().astype(float)
    b7_sr = 0.0001 * b7_array
    print(b7_sr[:3, :3])

    lst_b4 = gdal.Open('erdas_subset_b4.tif')
    print('Hello')
    b4_array = lst_b4.GetRasterBand(1).ReadAsArray().astype(float)
    # rad_cal(b4_array)

    lst_b5 = gdal.Open('erdas_subset_b5_140043.tif')
    print('Hello')
    b5_array = lst_b5.GetRasterBand(1).ReadAsArray().astype(float)
    b5_sr = 0.0001 * b5_array

    "This function will set a threshold criteria to effectively distinguish coal fire affected pixels"
    outfile = 'fire'
    fire = coal_fire_pixels(b7_sr, b6_sr, b5_sr)
    arr_to_img(fire, lst_b7, outfile)

    outfile = 'kinetic_temp_b7'
    emit = ndvi(b4_array, b5_array)
    b7_rad = radiometric_calibration(b7_sr)
    kinetic_arr = pixel_integrated_temperature(b7_rad, emit)
    arr_to_img(kinetic_arr, lst_b7, outfile)

    outfile = 'ndfi'
    ndfi_arr = normalized_difference_fire_index(b5_sr, b6_sr, b7_sr)
    arr_to_img(ndfi_arr, lst_b7, outfile)

    "This function will see the saturation of fire affected pixels in band 7"
    outfile = 'unambiguous'
    unambiguous_arr = unambiguous_fire_affected_pixels(b1_sr, b5_sr, b6_sr, b7_sr)
    arr_to_img(unambiguous_arr, lst_b7, outfile)


def ndvi(b4_arr, b5_arr):

    """
    This function will estimate the NDVI value from surface reflectance product and also it will discard the
    negative value and replace its value by zero

        Args:
            b4_arr : This is the array of band4 of Landsat 8 OLI
            b5_arr : This is the array band 5 of Landsat 8 OLI data

        Returns:
            will return the NDVI array
    """

    rows = np.shape(b5_arr)[0]
    cols = np.shape(b5_arr)[1]

    ndvi_arr = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if b5_arr[i, j] + b4_arr[i, j] == 0.0:
                print('the value of NDVI is zero')
                ndvi_arr[i, j] = 0.0
            else:
                ndvi_arr[i, j] = ((b5_arr[i, j] - b4_arr[i, j]) / (b5_arr[i, j] + b4_arr[i, j]))

    print(ndvi_arr[:3, :3])
    x = emissivity_map(ndvi_arr)
    return x


def emissivity_map(ndvi_arr):

    """
    This function will generate the emissivity map using NDVI value. There is an empirical relationship between
    ndvi_arr.

        Args:
            ndvi_arr : This is an array of NDVI map of the study area

        Returns
            using the NDVI map, it will generate a pixel-integrated emissivity map.

    """
    rows = np.shape(ndvi_arr)[0]
    cols = np.shape(ndvi_arr)[1]

    print(ndvi_arr, 'The NDVI map')
    emis_arr = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if ndvi_arr[i, j] > 0:
                emis_arr[i, j] = 1.0094 + 0.047 * math.log(ndvi_arr[i, j])
            else:
                emis_arr[i, j] = 0.9
    return emis_arr


def radiometric_calibration(b7):

    """
    This function will extract out the radiance value from surface reflectance value using the gain and offset
    value of metadata file

        Args:
            b7 : This is the array of surface reflectance of band 7

        Returns:
            It will return the radiance of band 7

    """

    b7_dn = b7 * 50000 + 50000
    print(b7_dn[:3, :3])

    b7_radiance = 0.00052993 * b7_dn - 2.64965
    print(b7_radiance[:3, :3], 'the radiance value of the band 7')
    return b7_radiance


def pixel_integrated_temperature(b7_rad, emit):

    """
    This function will generate the pixel integrated radiant temperature map as well as the kinetic temperature
    map of the each and every pixel of Landsat 8 OLI data. It will help identify the thermally anomalous ambiguous
    pixel

         Args:
             b7_rad : This is the array of at-satellite radiance of band 7
             emit : This is the emissivity map

         Returns:
             It will return the radiance of band 7

    """

    rows = np.shape(b7_rad)[0]
    cols = np.shape(b7_rad)[1]

    # h = 6.626 * (10 ** (-34))
    # c = 3 * (10 ** 8)
    # k = 1.381 * (10 ** (-23))
    wl = 2.2005 * (10 ** (-6))
    radiant_arr = np.zeros((rows, cols))
    kinetic_arr = np.zeros((rows, cols))

    # c1 = ((h * c) / k)
    # c2 = ((2 * h * (c ** 2)) * (wl ** (-5)))
    # c1 = 1260.56
    # c2 = 666.09
    x = 3.742 * (10 ** (-16))

    # temp_arr = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            # radiant_arr[i, j] = (c1 / (wl * math.log(1 + (c2 / b7_rad[i, j]))))
            # kinetic_arr[i, j] = ((radiant_arr[i, j]) / (emit[i, j] ** 0.25))
            # radiant_arr[i, j] = (c1 / (math.log(1 + (c2 / b7_rad[i, j]))))
            # kinetic_arr[i, j] = ((radiant_arr[i, j]) / (emit[i, j] ** 0.25))
            y = math.pi * (wl ** 5) * b7_rad[i, j]
            radiant_arr[i, j] = ((1.44 * 10 ** (-2)) / wl) / (math.log(1 + (x / y)))
            kinetic_arr[i, j] = ((radiant_arr[i, j]) / (emit[i, j] ** 0.25)) - 273

    print(radiant_arr[:3, :3], 'radiant temperature')
    print(kinetic_arr[:3, :3], 'kinetic temperature')
    print(b7_rad[:3, :3], ' radiance')
    print(emit, 'emissivity')
    return radiant_arr


def normalized_difference_fire_index(b5_sr, b6_sr, b7_sr):

    """
    This is the fire index personally developed by me after seeing an exponential trend of fire affected pixels

        Args:
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
                    ndfi_arr[i, j] = 0
                elif i1 > 0 and i2 > 0 and (i1 > 0.85 or i2 > 0.85):
                    ndfi_arr[i, j] = 0
                else:
                    ndfi_arr[i, j] = 1

    return ndfi_arr


def unambiguous_fire_affected_pixels(b1_sr, b5_sr, b6_sr, b7_sr):

    """
    This function will detect the potentially unambiguous fire affected pixels to understand the relationship
    to understand the potentiality of all the affected pixels to securely identify the affected pixels where
    the saturation in band 7 will be detected.

        Args:
            b1_sr (arry): This is the surface reflectance array of band 1 of landsat 8 OLI data
            b5_sr (arry): This is the surface reflectance array of band 5 of landsat 8 OLI data
            b6_sr (arry): This is the surface reflectance array of band 6 of landsat 8 OLI data
            b7_sr (arry):  This is the surface reflectance array of band 7 of landsat 8 OLI data

        Returns:
            This will return a binary classified map to detect the number of unambiguous fire affected pixels

    """

    rows = np.shape(b7_sr)[0]
    cols = np.shape(b7_sr)[1]

    unambiguous_arr = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            if b6_sr[i, j] > 0.8 and b1_sr[i, j] < 0.2 and (b5_sr[i, j] > 0.4 or b7_sr[i, j] < 0.1):
                unambiguous_arr[i, j] = 0
            else:
                unambiguous_arr[i, j] = 1

    return unambiguous_arr


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
                                fire_arr[i, j] = 1
                                x += 1
                    print('the number of potential fire pixels', x)
               
                else:
                    fire_arr[i, j] = 0
            else:
                fire_arr[i, j] = 0

    return fire_arr


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