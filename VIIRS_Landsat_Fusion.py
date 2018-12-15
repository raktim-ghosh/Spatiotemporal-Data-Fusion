import gdal
import numpy as np
import math

''' List of all the scenes of landsat 8 OLI data with a central meridian of 87 of Jharia Coalfield India'''


def __main():
    """ Frame of landsat 8 OLI data for the visualisation """

    ''' List of all the extracted stacked layer of Landsat 8 OLI data in the region of Jharia Coalfield
    lst is defined as Landsat 8 OLI data
    lt = List '''

    lst_all = gdal.Open('erdas_subset_140043_lst_b6.tif')

    lt = list()
    for i in range(1, lst_all.RasterCount + 1):
        temp1 = lst_all.GetRasterBand(i).ReadAsArray().astype(float)
        y = temp1
        lt.append(y)
    # print((lt[0]))

    " viirs band I_3 with a resolution of 463.00 m has been extracted. Let us assume that the viirs layer is within "
    " the bound of Landsat 8 OLI data"
    " viirs_t1 is the image "

    viirs_t1 = gdal.Open('resampled_viirs_i3_day09.tif')
    # viirs_t1 = gdal.Open('erdas_subset_140043_lst_b6.tif')

    vt1 = list()
    for i in range(1, viirs_t1.RasterCount + 1):
        temp2 = viirs_t1.GetRasterBand(i).ReadAsArray().astype(float)
        y = temp2
        vt1.append(y)

    viirs_t2 = gdal.Open('subset_resampled_viirs_i3_day25.tif')
    # viirs_t2 = gdal.Open('erdas_subset_140043_lst_b6.tif')

    vt2 = list()
    " vt2 is the list of all the bands stored in an array"
    for i in range(1, viirs_t2.RasterCount + 1):
        temp3 = viirs_t2.GetRasterBand(i).ReadAsArray().astype(float)
        y = temp3
        vt2.append(y)
    # a = []
    # b = []
    # c = []
    # a.append(lt[0][:200, :200])
    # b.append(vt1[0][:200, :200])
    # c.append(vt2[0][:200, :200])
    # print(np.shape(a))
    window_creation_viirs(lt, vt1, vt2, lst_all)
    # window_creation_viirs(a, b, c, lst_all)


# " This function will create the window for viirs data"


def window_creation_viirs(lst, vrs1, vrs2, lst_all):
    # this list will append all the windows of viirs bands which will be used to design teh weight function
    """
    This function will create windows from the given input image

        Args:
            lst (list): the bands of Landsat 8 OLI data are stored as a list
            vrs1 (list): the bands of first day of VIIRS are stored as a list
            vrs2 (list): the second day (predicted day), all the bands are stored as a list

        Returns:
            The function will create list of windows for all the input, and the windows will be used
            to extract spectral similarity in the next function (spectral similarity).

    """

    " here r_m1 and c_m1 denotes the rows and columns of the viirs image respectively "

    " in this function we will be using a list to create an image "
    img_as_lst = list()
    # print(len(vrs1))
    r_m1 = np.shape(vrs1[0])[0]
    c_m1 = np.shape(vrs1[0])[1]

    # print(len(vrs1), 'number of viirs window in a stacked image')
    for i in range(1, r_m1 - 1):
        for j in range(1, c_m1 - 1):
            vrs1_of_windows = list()
            vrs2_of_windows = list()
            lst_of_windows = list()
            for k in range(len(vrs1)):
                window_vst1 = vrs1[k][i - 1:i + 2, j - 1:j + 2]
                window_vst2 = vrs2[k][i - 1:i + 2, j - 1:j + 2]
                window_lst1 = lst[k][i - 1:i + 2, j - 1:j + 2]
                vrs1_of_windows.append(window_vst1)
                vrs2_of_windows.append(window_vst2)
                lst_of_windows.append(window_lst1)
                " lst_of_windows: represents the list of windows of landsat "
            print(np.shape(lst_of_windows[0]), 'the shape of the window', i, j)

            if len(vrs1_of_windows) >= 1:
                if np.shape(vrs1_of_windows[0])[0] * np.shape(vrs1_of_windows[0])[1] == 9:
                    " here we are checking the length of windows and dimension of windows "
                    a = spectral_similarity(lst_of_windows, vrs1_of_windows, vrs2_of_windows)
                    img_as_lst.append(a)
                    # print(a)
            else:
                continue
    # print(len(img_as_lst), 'the length of the image list')
    c_new = c_m1 - 1
    creation_of_image(img_as_lst, c_new, lst_all)

    # " this function will send the square window defined over the viirs data (for time1 and time2 along with the
    # Landsat"
    # " 8 OLI band"

# '''
# def window_creation_lst(lst_bands, vrs1_wndw, vrs2_wndw):
#     """ This function will dynamically create a square window and operate over the whole image and also it will help
#     to create the mean and variance image of the input image having multiple bands
#     Initial Window = Square window """
#
#     lst_of_windows = list()
#     vrs1_of_windows = vrs1_wndw
#     vrs2_of_windows = vrs2_wndw
#     " here r_b1, c_b1 denotes the rows and columns of the landsat image respectively "
#
#     r_b1 = np.shape(lst_bands[0])[0]
#     c_b1 = np.shape(lst_bands[0])[1]
#
#     for i in range(1, r_b1):
#         for j in range(1, c_b1):
#             for k in range(len(lst_bands)):
#                 " creation of square window "
#                 window_lst = lst_bands[k][i - 1:i + 2, j - 1:j + 2]
#                 lst_of_windows.append(window_lst)
#                 # print(lst_of_windows[0])
#
#     # " The window has been sent to extract out spectrally similar pixels for post-analysis "
#     # print('continue')
#             z = spectral_similarity(lst_of_windows, vrs1_of_windows, vrs2_of_windows)
#             return z
#
# '''


def spectral_similarity(l_w, v1_w, v2_w):
    # print('Test for inserting into the function')
    # print(l_w, 'list of window of landsat')
    # print(v1_w, 'list of window of first viirs')
    # print(v2_w, 'list of window of second viirs')
    """  This function will extract out the spectrally similar homogeneous pixels within a given window
    sd = standard deviation  cls denotes the number of probable classes to extract out the similar pixels
    l_w denotes the list of windows of all the bands being sent to the spectral similarity test

        Args:
            l_w (list): consist of the list of windows extracted from each band of Landsat 8 OLI data.
            v1_w (list): consist of the list of windows extracted from each band of VIIRS.
            v2_w (list): consist of the list of windows extracted from each  band of VIIRS on teh day of prediction.

        Returns:
            The list of windows are sent to the next function which will design weight from the extracted pixels.

    """

    cls = 10
    sum2 = 0
    " w_r, w_c denotes the number of rows and columns of the defined window of landsat window "

    w_r = np.shape(l_w[0])[0]
    w_c = np.shape(l_w[0])[1]

    " this will represent the window of viirs data"

    # vrs_r = np.shape(v1_w[0])[0]
    # vrs_c = np.shape(v2_w[0])[1]
    " The standard deviation has been estimated in order to distinguish the homogeneous pixels"
    " lst_sd is the list of windows "

    lst_sd = list()
    # print(len(l_w))
    for i in range(len(l_w)):
        lst_sd.append(np.std(l_w[i]))
        # sum1 = 0
        '''
        for j in range(w_r):
            for k in range(w_c):
                " the standard deviation has been estimated below "
                sum1 += math.sqrt(((l_w[i][j, k] - np.mean(l_w[i])) ** 2) / ((w_r * w_c) - 1))
                # print(sum1)
            sum2 += sum1
        " The standard deviation has been estimated below "
        '''
        # sd = math.sqrt((sum2 - np.mean(l_w)) ** 2 / (w_r * w_c))
        # lst_sd.append(sum2)

    " c_w_c, c_w_r denotes the middle point of the square window with odd number of rows and columns "
    " This for loop is designed to extract the spectrally similar pixels "

    c_w_r = w_r // 2
    c_w_c = w_c // 2

    " the spectrally similar pixels are kept into a list to design the weight function list: similar pixels "

    " this loop is to "
    similar_pixels_lst = list()
    similar_pixels_vrs1 = list()
    similar_pixels_vrs2 = list()

    ' the dijk is a list of distance between the spectrally similar pixel and the central pixel of the window'

    dst_lst = list()

    " t_c denotes the number of times it has been tested as a potential homogeneous pixels "
    t_c = 0

    'A list has been created to store the index value of homogeneous pixel of landsat 8 OLI data'

    " loop for num of rows in a window "

    for i in range(w_r):
        ' loop for num of cols in a window '
        for j in range(w_c):
            ' total number of standard deviation of every window of each band of Landsat 8 OLI data '
            for k in range(len(lst_sd)):
                if l_w[k][i, j] != l_w[k][c_w_r, c_w_c]:
                    " conditions for testing a pixel to be a potentially homogeneous pixels "
                    if abs(l_w[k][i, j] - l_w[k][c_w_r, c_w_c]) <= abs(2 * lst_sd[k] / cls):
                        # print(y, 'something')
                        # print(x, 'everything')
                        # print('true')
                        # print(l_w[k][c_w_r, c_w_c])
                        t_c += 1

            lst_pic_vec = list()
            vrs1_pic_vec = list()
            vrs2_pic_vec = list()
            # print(t_c)
            if t_c == len(lst_sd):
                ' list is created to store homogeneous pixel as a vector in the form of list '

                for k in range(len(l_w)):
                    lst_pic_vec.append(l_w[k][i, j])
                ' similar_pixels is a list consists of pixel-vectors (as a list) having the property of homogeneity '
                ' for landsat 8 OLI data '

                similar_pixels_lst.append(lst_pic_vec)
                # print(similar_pixels_lst)

                ' the same pixel location from the window of viirs1 and viirs2, have been extracted by indexing with'
                'the similar pixel location of landsat 8 OLI data '
                # print(v1_w, 'window of viirs')
                # print(v1_w[0], 'the first window')
                for n in range(len(v1_w)):
                    vrs1_pic_vec.append(v1_w[n][i, j])
                    vrs2_pic_vec.append(v2_w[n][i, j])

                similar_pixels_vrs1.append(vrs1_pic_vec)
                similar_pixels_vrs2.append(vrs2_pic_vec)

                # print(similar_pixels_vrs1, 'similar pixel of first image of viirs')
                # print(similar_pixels_vrs2, 'similar pixel of second image of viirs')
                euclidean_distance = 1 + math.sqrt(((i - c_w_r) ** 2) + ((j - c_w_c) ** 2)) / 1.5
                dst_lst.append(euclidean_distance)
                # print(dst_lst)

            t_c = 0

    # The list of similar_pixels consist of all the similar pixels and the last element is a list consist of central 
    # pixel of the window
    # print('The code is good')
    # print(lst_sd)
    mean_lst = list()
    for i in range(len(l_w)):
        mean_lst.append(l_w[i][c_w_r, c_w_c])

    similar_pixels_lst.append(mean_lst)

    # print(similar_pixels_lst)
    " The list of all sm pixel has been sent to the next function which is acting as a weight, and also the viirs data "
    # print(similar_pixels_lst, 'spectrally similar pixels of landsat')
    # print(similar_pixels_vrs1, 'spectrally similar pixels of first viirs')
    # print(similar_pixels_vrs2, 'spectrally similar pixels of second viirs')
    if len(similar_pixels_lst) == 1:
        return mean_lst
    else:
        y = design_weight(similar_pixels_lst, similar_pixels_vrs1, similar_pixels_vrs2, dst_lst)
        return y


def design_weight(lst_sm_pxl, vrs1_sm_pxl, vrs2_sm_pxl, dst):

    """ This function will design weight for the two viirs input data and one Landsat 8 OLI data."
    The spectral bias, spatial bias and temporal bias will be considered to design the end weight function

        Args:
            lst_sm_pxl (list): The list of similar pixels of Landsat 8 OLI data.
            vrs1_sm_pxl (list): The list of similar pixels of VIIRS first day
            vrs2_sm_pxl (list): The list of similar pixels of VIIRS predicted day.
            dst: list of spatial distances from the central to the corresponding location of similar pixels within a
            window.

        Returns:
            List of similar pixels of Landsat 8 OLI data.

    """

    " sijk: spectral bias, dijk: temporal bias, tijk: temporal bias "

    # print('entering into the weight function')
    sijk = list()
    dijk = dst
    tijk = list()

    # print(lst_sm_pxl, 'list of similar pixel of landsat')
    # print(vrs1_sm_pxl, 'list of similar pixel of first viirs image')
    # print(vrs2_sm_pxl, 'list of similar pixel of sencond viirs image')
    # print(dst, 'list of Euclidean distance')

    for i in lst_sm_pxl:
        temp_syn_lst_sijk = list()
        temp_syn_lst_tijk = list()
        if lst_sm_pxl.index(i) != len(lst_sm_pxl) - 1:
            for j in range(len(i)):
                temp_syn_lst_sijk.append(abs(i[j] - vrs1_sm_pxl[lst_sm_pxl.index(i)][j]))
                t = abs(vrs1_sm_pxl[lst_sm_pxl.index(i)][j] - vrs2_sm_pxl[lst_sm_pxl.index(i)][j])
                temp_syn_lst_tijk.append(t)
            sijk.append(temp_syn_lst_sijk)
            tijk.append(temp_syn_lst_tijk)

    " Here spb: spatial bias, sptb: spectral bias, tb: temporal bias"

    wijk = list()
    # print('entering into the normalization')
    # print(len(sijk[0]), 'the length of the spectral bias')
    for m in range(len(sijk[0])):
        sum1 = 0
        temp_lst = list()
        for n in range(len(tijk)):
            if sijk[n][m] == 0 or tijk[n][m] == 0:
                sum1 = 1
                break
            else:
                # print(sum1)
                sum1 += 1 / (sijk[n][m] * tijk[n][m] * dijk[n])
        for l in range(len(tijk)):
            if sum1 == 1:
                temp2 = 1
                temp_lst.append(temp2)
            else:
                # x = 1 / (sijk[l][m] * tijk[l][m] * dijk[l])
                # print(x)
                temp2 = ((1 / (sijk[l][m] * tijk[l][m] * dijk[l])) / sum1)
                temp_lst.append(temp2)
        wijk.append(temp_lst)
        # print(wijk)
    # print(wijk, 'weight list')

    x = central_window_pixel(lst_sm_pxl, vrs1_sm_pxl, vrs2_sm_pxl, wijk)
    return x

# " This function will create the central window pixel and image thereafter "


def central_window_pixel(landsat_sm_pxl, viirs1_sm_pxl, viirs2_sm_pxl, weight_lst):

    # print(landsat_sm_pxl)
    # print(viirs1_sm_pxl)
    # print(viirs2_sm_pxl)
    # print(weight_lst)

    """ The central pixel of the window is taken as a list or a vector of all hte bands"""
    central_pxl_vec = list()
    for i in range(len(viirs2_sm_pxl[0])):
        sum1 = 0
        for j in range(len(weight_lst[0])):
            sum1 += weight_lst[i][j] * (landsat_sm_pxl[j][i] + viirs2_sm_pxl[j][i] - viirs1_sm_pxl[j][i])
        central_pxl_vec.append(abs(sum1))

    # print(central_pxl_vec, 'central pixel vector')
    # print('hello')
    return central_pxl_vec


def creation_of_image(image_vector_lst, col, lst_all):

    img_as_lst = list()
    for values in zip(*image_vector_lst):
        img_as_lst.append(values)

    # column = list()
    outfile = 'viirs_landsat_fusion_day25'
    # print(img_as_lst)
    # print(len(img_as_lst))
    # print(len(img_as_lst[0]))
    # print(img_as_lst[0])
    temp_lst = list()
    print(col)

    for (index, image) in enumerate(img_as_lst):
        for i in range(0, len(image), col - 1):
            x = image[i:i + col - 1]
            y = list(x)
            if len(y) == col - 1:
                temp_lst.append(y)

        arr = np.array(temp_lst)
        cols, rows = arr.shape
        driver = gdal.GetDriverByName("GTiff")
        outfile += '.tiff'
        out_data = driver.Create(outfile, rows, cols, 1, gdal.GDT_Float32)
        out_data.SetProjection(lst_all.GetProjection())
        out_data.SetGeoTransform(lst_all.GetGeoTransform())
        out_data.GetRasterBand(1).WriteArray(arr)
        out_data.FlushCache()

    '''
    for i in range(len(img_as_lst)):
        for j in range(0, col, len((img_as_lst[0]))):
            x = img_as_lst[i][j: j + col]
            list(x)
            column.append(x)
        arr = np.array(column)
        '
    for (index, image) in enumerate(img_as_lst):
        for i in range(0, len(image), col):
            x = image[i:i + col]
            y = list(x)
            if len(y) == col:
                temp_lst.append(y)
            
        arr = np.array(temp_lst)
        cols, rows = arr.shape
        driver = gdal.GetDriverByName("GTiff")
        outfile += '.tiff'
        out_data = driver.Create(outfile, rows, cols, 1, gdal.GDT_Float32)
        out_data.GetRasterBand(1).WriteArray(arr)
        out_data.FlusgCache()
        '''


if __name__ == '__main__':
    __main()