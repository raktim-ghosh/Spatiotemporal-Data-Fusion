import gdal
from numpy import *
import math
import numpy as np


def __main():

    """
    subset image of Landsat 8 OLI data.
    """

    'lt is the list of all bands of Landsat 8 OLI data appended as array for the day 1'

    lst1_all = gdal.Open('erdas_subset_140043_lst_b6.tif')
    lt1 = list()
    lt1_std = list()
    for i in range(1, lst1_all.RasterCount + 1):
        temp1 = lst1_all.GetRasterBand(i).ReadAsArray().astype(float)
        lt1.append(temp1)
        lt1_std.append(np.std(temp1))

    'It is the list of all bands of Landsat 8 OLI data appended as array for the day 2'

    lst3_all = gdal.Open('erdas_subset_140043_lst_b6.tif')
    lt3 = list()
    lt3_std = list()
    for i in range(1, lst3_all.RasterCount + 1):
        temp3 = lst3_all.GetRasterBand(i).ReadAsArray().astype(float)
        lt3.append(temp3)
        lt3_std.append(np.std(temp3))

    ' vt1 (list): list of all bands of VIIRS captured at day 1'

    viirs_all_t1 = gdal.Open('resampled_viirs_i3_day09.tif')
    vt1 = list()
    for i in range(1, viirs_all_t1.RasterCount + 1):
        tmp1 = viirs_all_t1.GetRasterBand(i).ReadAsArray().astype(float)
        vt1.append(tmp1)

    viirs_all_t2 = gdal.Open('resampled_viirs_i3_day16_tif')

    ' vt2 (list): list of all bands of VIIRS captured at day 2'
    vt2 = list()
    for i in range(1, viirs_all_t2.RasterCount + 1):
        tmp2 = viirs_all_t2.GetRasterBand(i).ReadAsArray().astype(float)
        vt2.append(tmp2)

    viirs_all_t3 = gdal.Open('subset_resampled_viirs_i3_day25.tif')

    ' vt3 (list): list of all bands of VIIRS captured at day 3'
    vt3 = list()
    for i in range(1, viirs_all_t3.RasterCount + 1):
        tmp3 = viirs_all_t3.GetRasterBand(i).ReadAsArray().astype(float)
        vt3.append(tmp3)

    ' sliding window function will return a vector list of all synthetic pixels '
    image_vector_lst = sliding_windows(lt1, lt3, vt1, vt2, vt3, lt1_std, lt3_std)

    ' the function will create image using image_vector_lst, the projection system and number of columns of the input'
    col = np.shape(lt1[0])[1]
    creation_of_image(image_vector_lst, lst1_all, col)


def window_size():
    """
    The window size will assign the height and width of the window. The dynamic assignment will be based
    on the study.
    """
    w = 3
    h = 3

    return [w, h]


def sliding_windows(lt1, lt3, vt1, vt2, vt3, lt1_std, lt3_std):

    """
    This function will create the dynamic sliding window from a given input image

        Args:
            lt1 (list): list of landsat 8 OLI band in array format
            lt3 (list): list of landsat 8 OLI band in array format
            vt1 (list): list of viirs band at first day stored in array format
            vt2 (list): list of viirs band at first day stored in array format
            vt3 (list): list of viirs band at predicted day stored in array format
            lt1_std (list): list of standard deviation value of all images
            lt3_std (list): list of standard deviation value of all images

        Returns:
            this function will return the synthetic spectral list
    """

    ' synthetic_central_pxl_lst: the list is created to store all the synthetic pixels '
    synthetic_central_pxl_lst = list()

    rows = np.shape(lt1[0])[0]
    cols = np.shape(lt1[0])[1]

    w_s = window_size()
    w = w_s[0]
    h = w_s[1]
    ' here we need to check the whether a window is uneven or even '
    for i in range(rows):
        for j in range(cols):
            'lst1_wndw (list): is a list appending individual window from each band of landsat'
            'lst3_wndw (list): is a list appending individual window from each band of landsat'
            'vrs1_wndw (list): is a list appending individual window from each band of VIIRS'
            'vrs2_wndw (list): is a list appending individual window from each band of VIIRS'
            'vrs3_wndw (list): is a list appending individual window from each band of VIIRS'
            'f1, f2, f3, f4 are the indicators of the pixels'
            'dmy_r1, dmy_r2, dmy_r3, dmy_r4 are the variables to find out the central pixel of non-square window'

            lst1_wndw, lst3_wndw, vrs1_wndw, vrs2_wndw, vrs3_wndw = list(), list(), list(), list(), list()
            f1, f2, f3, f4 = False, False, False, False
            dmy_r1, dmy_r2, dmy_c1, dmy_c2 = 0, 0, 0, 0
            flag_lst, dummy_lst = list(), list()

            for k in range(len(lt1)):
                ' the boundary conditions are checked to set the window size according to the edge of an image '
                if i - (h // 2) <= 0:
                    r1 = 0
                    f1 = True
                    dmy_r1 = (h // 2) - i
                else:
                    r1 = i - (h // 2)
                if j - (w // 2) <= 0:
                    c1 = 0
                    f2 = True
                    dmy_c1 = (w // 2) - j
                else:
                    c1 = j - (w // 2)
                if i + (h // 2) >= rows:
                    r2 = rows
                    dmy_r2 = i + (h // 2) - rows + 1
                    f3 = True
                else:
                    r2 = i + (h // 2)
                if j + (w // 2) >= cols:
                    c2 = cols
                    dmy_c2 = j + (w // 2) - cols + 1
                    f4 = True
                else:
                    c2 = j + (w // 2)

                flag_lst = [f1, f2, f3, f4]
                dummy_lst = [dmy_r1, dmy_r2, dmy_c1, dmy_c2]

                lst1_wndw.append(lt1[k][r1:r2 + 1, c1:c2 + 1])
                lst3_wndw.append(lt3[k][r1:r2 + 1, c1:c2 + 1])
                vrs1_wndw.append(vt1[k][r1:r2 + 1, c1:c2 + 1])
                vrs2_wndw.append(vt2[k][r1:r2 + 1, c1:c2 + 1])
                vrs3_wndw.append(vt3[k][r1:r2 + 1, c1:c2 + 1])

            synthetic_central_pxl_lst.append(spectral_similarity(lst1_wndw, lst3_wndw,
                                                                 vrs1_wndw, vrs2_wndw, vrs3_wndw,
                                                                 lt1_std, lt3_std, flag_lst, dummy_lst))

    return synthetic_central_pxl_lst


def central_window_pixel_locator(flag_list, dummy_lst, lst1_wndw):

    """
    This function will be able to locate the dummy central pixels of a virtual window. The function will dynamically
    select the central pixels of the window.

        Args:
            flag_list (list): The flag list will be able to dynamically assign the window
            dummy_lst (list): The dummy list will be able to dynamically assign the window
            lst1_wndw (list): consist of the list of windows extracted from each band of Landsat 8 OLI data on day 1

        Returns:
            it will return the location of central window pixel

    """

    rows = np.shape(lst1_wndw[0])[0]
    cols = np.shape(lst1_wndw[0])[1]
    w_s = window_size()
    w, h = w_s[0], w_s[1]

    if flag_list[0] is True and flag_list[1] is True and flag_list[2] is False and flag_list[3] is False:
        for i in range(rows):
            for j in range(cols):
                if (h // 2) - dummy_lst[0] == i and (w // 2) - dummy_lst[1] == j:
                    r_w_c = i
                    c_w_c = j
                    return [r_w_c, c_w_c]

    elif flag_list[0] is True and flag_list[1] is False and flag_list[2] is False and flag_list[3] is False:
        for i in range(rows):
            for j in range(cols):
                if (h // 2) - dummy_lst[0] == i and j == (w // 2):
                    r_w_c = i
                    c_w_c = j
                    return [r_w_c, c_w_c]

    elif flag_list[0] is True and flag_list[1] is False and flag_list[2] is False and flag_list[3] is True:
        for i in range(rows):
            for j in range(cols):
                if (h // 2) - dummy_lst[0] == i and j == (w // 2):
                    r_w_c = i
                    c_w_c = j
                    return [r_w_c, c_w_c]

    elif flag_list[0] is False and flag_list[1] is True and flag_list[2] is False and flag_list[3] is False:
        for i in range(rows):
            for j in range(cols):
                if (h // 2) == i and (w // 2) - dummy_lst[1] == j:
                    r_w_c = i
                    c_w_c = j
                    return [r_w_c, c_w_c]

    elif flag_list[0] is False and flag_list[1] is False and flag_list[2] is False and flag_list[3] is True:
        for i in range(rows):
            for j in range(cols):
                if (h // 2) == i and (w // 2) == j:
                    r_w_c = i
                    c_w_c = j
                    return [r_w_c, c_w_c]

    elif flag_list[0] is False and flag_list[1] is True and flag_list[2] is True and flag_list[3] is False:
        for i in range(rows):
            for j in range(cols):
                if (h // 2) == i and (w // 2) - dummy_lst[2] == j:
                    r_w_c = i
                    c_w_c = j
                    return [r_w_c, c_w_c]

    elif flag_list[0] is False and flag_list[1] is False and flag_list[2] is True and flag_list[3] is False:
        for i in range(rows):
            for j in range(cols):
                if (h // 2) == i and (w // 2) == j:
                    r_w_c = i
                    c_w_c = j
                    return [r_w_c, c_w_c]

    elif flag_list[0] is False and flag_list[1] is False and flag_list[2] is True and flag_list[3] is True:
        for i in range(rows):
            for j in range(cols):
                if (h // 2) == i and (w // 2) == j:
                    r_w_c = i
                    c_w_c = j
                    return [r_w_c, c_w_c]


def spectral_similarity(lst1_wndw, lst3_wndw, vrs1_wndw, vrs2_wndw, vrs3_wndw, lt1_std, lt3_std, flag_lst, dummy_lst):

    """
    This function will extract out the spectrally similar homogeneous pixels within a given window
    sd = standard deviation  cls denotes the number of probable classes to extract out the similar pixels
    l_w denotes the list of windows of all the bands being sent to the spectral similarity test

        Args:
            lst1_wndw (list): consist of the list of windows extracted from each band of Landsat 8 OLI data on day 1.
            lst3_wndw (list): consist of the list of windows extracted from each band of Landsat 8 OLI data on day 3.
            vrs1_wndw (list): consist of the list of windows extracted from each band of VIIRS on day 1
            vrs2_wndw (list): consist of the list of windows extracted from each band of VIIRS on day 2
            vrs3_wndw (list): consist of the list of windows extracted from each band of VIIRS on day 3
            lt1_std  (list1): consist of standard deviation of all the bands of Landsat 8 OLI data
            lt3_std (list1): consist of standard deviation of all the bands of Landsat 8 OLI data
            flag_lst (list): is a list to locate the central pixel for non-square window by considering a dummy
                             square window.
            dummy_lst (list): is a list of absolute difference for locating the central pixel

        Returns:
            The list of windows are sent to the next function which will design weight from the extracted pixels.

    """

    rows = np.shape(lst1_wndw[0])[0]
    cols = np.shape(lst1_wndw[0])[1]

    cls = 10
    w_s = window_size()

    if rows * cols == w_s[0] * w_s[1]:
        r_w_c = rows // 2
        c_w_c = cols // 2
    else:
        x = central_window_pixel_locator(flag_lst, dummy_lst, lst1_wndw)
        r_w_c = x[0]
        c_w_c = x[1]

    ' central pixel vector extracted for day 1 of landsat 8 OLI data '
    central_pic_vec_lst1 = list()
    for i in range(len(lst1_wndw)):
        central_pic_vec_lst1.append(lst1_wndw[i][r_w_c, c_w_c])

    central_pic_vec_lst3 = list()
    for j in range(len(lst3_wndw)):
        central_pic_vec_lst3.append(lst3_wndw[j][r_w_c, c_w_c])

    smlr_lst1 = list()
    smlr_lst3 = list()
    smlr_vrs1 = list()
    smlr_vrs1_vrs2 = list()
    smlr_vrs3 = list()
    smlr_vrs3_vrs2 = list()

    ' dst_lst1 , dst_lst3 denotes the list of Euclidean distance for all the similar pixels '
    dst_lst1 = list()
    dst_lst3 = list()

    t_c_lst1 = 0
    t_c_lst3 = 0
    for i in range(rows):
        for j in range(cols):
            for k in range(len(lst1_wndw)):
                if lst1_wndw[k][i, j] != lst1_wndw[k][r_w_c, c_w_c]:
                    if abs(lst1_wndw[k][i, j] - lst1_wndw[k][r_w_c, c_w_c]) <= 2 * (lt1_std[k] / cls):
                        t_c_lst1 += 1
                if lst3_wndw[k][i, j] != lst3_wndw[k][r_w_c, c_w_c]:
                    if abs(lst3_wndw[k][i, j] - lst3_wndw[k][r_w_c, c_w_c]) <= 2 * (lt3_std[k] / cls):
                        t_c_lst3 += 1

            smlr_pic_vec_lst1 = list()
            smlr_pic_vec_lst3 = list()
            smlr_pic_vec_vrs1 = list()
            smlr_pic_vec_vrs1_vrs2 = list()
            smlr_pic_vec_vrs3 = list()
            smlr_pic_vec_vrs3_vrs2 = list()

            if t_c_lst1 == len(lst1_wndw):
                for l in range(len(lst1_wndw)):
                    smlr_pic_vec_lst1.append(lst1_wndw[l][i, j])
                    smlr_pic_vec_vrs1.append(vrs1_wndw[l][i, j])
                    smlr_pic_vec_vrs1_vrs2.append(vrs2_wndw[l][i, j])

                smlr_lst1.append(smlr_pic_vec_lst1)
                smlr_vrs1.append(smlr_pic_vec_vrs1)
                smlr_vrs1_vrs2.append(smlr_pic_vec_vrs1_vrs2)

                euclidean_distance_lst1 = 1.0 + math.sqrt(((i - r_w_c) ** 2) + ((j - c_w_c) ** 2)) / 1.5
                dst_lst1.append(euclidean_distance_lst1)

            if t_c_lst3 == len(lst3_wndw):
                for m in range(len(lst3_wndw)):
                    smlr_pic_vec_lst3.append(lst3_wndw[m][i, j])
                    smlr_pic_vec_vrs3.append(vrs3_wndw[m][i, j])
                    smlr_pic_vec_vrs3_vrs2.append(vrs2_wndw[m][i, j])

                smlr_lst3.append(smlr_pic_vec_lst3)
                smlr_vrs3.append(smlr_pic_vec_vrs3)
                smlr_vrs3_vrs2.append(smlr_pic_vec_vrs3_vrs2)

                euclidean_distance_lst3 = 1.0 + math.sqrt(((i - r_w_c) ** 2) + ((j - c_w_c) ** 2)) / 1.5
                dst_lst3.append(euclidean_distance_lst3)

    lst1_smlr = list()
    for values in zip(*smlr_lst1):
        lst1_smlr.append(values)

    lst3_smlr = list()
    for values in zip(*smlr_lst3):
        lst3_smlr.append(values)

    vrs1_smlr = list()
    for values in zip(*smlr_vrs1):
        vrs1_smlr.append(values)

    vrs3_smlr = list()
    for values in zip(*smlr_vrs3):
        vrs3_smlr.append(values)

    vrs1_vrs2_smlr = list()
    for values in zip(*smlr_vrs1_vrs2):
        vrs1_vrs2_smlr.append(values)

    vrs3_vrs2_smlr = list()
    for values in zip(*smlr_vrs3_vrs2):
        vrs3_vrs2_smlr.append(values)

    central_pic_vec = [central_pic_vec_lst1, central_pic_vec_lst3]
    if len(smlr_lst1) == 0:
        return [central_pic_vec_lst1, central_pic_vec_lst3]
    else:
        l_w = get_weight(lst1_smlr, lst3_smlr, vrs1_smlr, vrs3_smlr, dst_lst1, dst_lst3)
        v = get_conversion_coefficient(lst1_smlr, lst3_smlr, vrs1_smlr, vrs3_smlr)
        t = temporal_difference(vrs1_smlr, vrs1_vrs2_smlr, vrs3_vrs2_smlr, vrs3_smlr)
        c_pxl = create_central_pixel(l_w, v, t, central_pic_vec, vrs1_smlr, vrs1_vrs2_smlr, vrs3_vrs2_smlr, vrs3_smlr)

        return c_pxl


def get_weight(lst1_smlr, lst3_smlr, vrs1_smlr, vrs3_smlr, dst_lst1, dst_lst3):

    """
    This function will design weight for the two viirs input data and one Landsat 8 OLI data."
    The spectral bias, spatial bias and temporal bias will be considered to design the end weight function

        Args:
            lst1_smlr (list): The list of similar pixels of Landsat 8 OLI data on day 1
            lst3_smlr (list): The list of similar pixels of Landsat 8 OLI data on day 3
            vrs1_smlr (list): The list of similar pixels of VIIRS on day 1
            vrs3_smlr (list): The list of similar pixels of VIIRS on day 3
            dst_lst1 (list): list of spatial distances from the central
                             to the corresponding location of similar pixels within a window for day 1.
            dst_lst3 (list): list of spatial distances from the central
                             to the corresponding location of similar pixels within a window for day 3.
        Returns:
            List of weights for each similar pixels contributing towards the measure of synthetic pixel within a
            same window.

    """
    lst_weight1 = list()
    lst_weight3 = list()

    dst_lst1_lst3 = dst_lst1 + dst_lst3
    print(len(vrs3_smlr))
    for sp1 in lst1_smlr:
        temp1_weight = list()
        temp3_weight = list()
        mean13_fi = np.mean(sp1 + lst3_smlr[lst1_smlr.index(sp1)])
        mean13_ci = np.mean(vrs1_smlr[lst1_smlr.index(sp1)] + vrs3_smlr[lst1_smlr.index(sp1)])
        var13_fi = np.var(sp1 + lst3_smlr[lst1_smlr.index(sp1)])
        var13_ci = np.var(vrs1_smlr[lst1_smlr.index(sp1)] + vrs3_smlr[lst1_smlr.index(sp1)])
        sum13 = 0

        temp_lst1_lst3 = sp1 + lst3_smlr[lst1_smlr.index(sp1)]
        temp_vrs1_vrs3 = vrs1_smlr[lst1_smlr.index(sp1)] + vrs3_smlr[lst1_smlr.index(sp1)]
        for i in range(len(temp_lst1_lst3)):

            fi = temp_lst1_lst3[i]
            ci = temp_vrs1_vrs3[i]
            temp_ri = (((fi - mean13_fi) * (ci - mean13_ci)) / (var13_fi * var13_ci))
            temp_di = (1 - temp_ri) * dst_lst1_lst3[i]
            sum13 += 1 / temp_di

        for j in range(len(temp_lst1_lst3)):
            fj = temp_lst1_lst3[j]
            cj = temp_vrs1_vrs3[j]
            temp_rj = (((fj - mean13_fi) * (cj - mean13_ci)) / (var13_fi * var13_ci))
            temp_dj = (1 - temp_rj) * dst_lst1_lst3[j]
            inv_temp_dj = 1 / temp_dj
            if j < len(sp1):
                temp1_weight.append(inv_temp_dj / sum13)
            else:
                temp3_weight.append(inv_temp_dj / sum13)

        lst_weight1.append(temp1_weight)
        lst_weight3.append(temp3_weight)

    return [lst_weight1, lst_weight3]


def get_conversion_coefficient(lst1_smlr, lst3_smlr, vrs1_smlr, vrs3_smlr):

    """
    This function will retrieve the conversion coefficient. The conversion coefficient is measured by the retrieved
    regression parameter between the fine resolution similar pixels with the corresponding location of
    coarse resolution pixels within a same window size. The slope of the regression line can be taken as a
    conversion coefficient

        Args:
            lst1_smlr (list): The list of similar pixels of Landsat 8 OLI data for day 3.
            lst3_smlr (list): The list of similar pixels of Landsat 8 OLI data for day 3.
            vrs1_smlr (list): The list of similar pixels of VIIRS first day
            vrs3_smlr (list): The list of similar pixels of VIIRS last day.

        Returns:
            Returns the conversion coefficient list

    """

    v1 = list()
    for sp1 in lst1_smlr:
        temp_lst1_lst3 = sp1 + lst3_smlr[lst1_smlr.index(sp1)]
        temp_vrs1_vrs3 = vrs1_smlr[lst1_smlr.index(sp1)] + vrs3_smlr[lst1_smlr.index(sp1)]

        for l1_l3_smlr, v1_v3_smlr in temp_lst1_lst3, temp_vrs1_vrs3:
            cor = corrcoef(l1_l3_smlr, v1_v3_smlr)[0, 1]
            std_v1_v3 = np.std(v1_v3_smlr)
            std_l1_l3 = np.std(l1_l3_smlr)
            slope = cor * (std_l1_l3 / std_v1_v3)
            v1.append(slope)

    return v1


def temporal_difference(vrs1_smlr, vrs1_vrs2_smlr, vrs3_vrs2_smlr, vrs3_smlr):

    """
    This function will measure teh temporal difference between the coarse resolution pixel at day 1 and day 3 with the
    prediction date. The prediction date is day 2.

        Args:
            vrs1_smlr (list): List of coarse resolution pixels having location corresponding to similar pixels of
                              Landsat 8 OLI data.
            vrs1_smlr (list): List of coarse resolution pixels will be chosen according to the similar location of
                              pixels.
            vrs1_vrs2_smlr (list): List of coarse resolution pixels will be chosen according to the similar location
                                   of corresponding fine resolution pixels.
            vrs3_vrs2_smlr (list): List of coarse resolution pixels will be chosen according to the similar location
                                   of corresponding fine resolution pixels.
            vrs3_smlr (list): List of coarse resolution pixels having location corresponding to similar pixels of
                              Landsat 8 OLI data.

        Returns:
            it will return a temporal difference weight

    """

    tmp_diff_vrs1_vrs2_smlr = list()
    tmp_diff_vrs3_vrs2_smlr = list()
    for sp1 in vrs1_smlr:
        temp_diff = 0
        for i in range(len(sp1)):
            temp_diff += 1 / (sp1[i] - vrs1_vrs2_smlr[vrs1_smlr.index(sp1)][i])
        for j in range(len(sp1)):
            inv_diff = 1 / (sp1[j] - vrs1_vrs2_smlr[vrs1_smlr.index(sp1)][j])
            tmp_diff_vrs1_vrs2_smlr.append(inv_diff / temp_diff)

    for sp2 in vrs3_smlr:
        temp_diff = 0
        for i in range(len(sp2)):
            temp_diff += 1 / (sp2[i] - vrs3_vrs2_smlr[vrs3_smlr.index(sp2)][i])
        for j in range(len(sp2)):
            inv_diff = 1 / (sp2[j] - vrs3_vrs2_smlr[vrs3_smlr.index(sp2)][j])
            tmp_diff_vrs3_vrs2_smlr.append(inv_diff / temp_diff)

    return [tmp_diff_vrs1_vrs2_smlr, tmp_diff_vrs3_vrs2_smlr]


def create_central_pixel(l_w, v, t, central_pic_vec, vrs1_smlr, vrs1_vrs2_smlr, vrs3_vrs2_smlr, vrs3_smlr):
    """
    This function will create a synthetic pixel for each sliding window which will be aggregated to
    generate a synthetic predicted image

        Args:
            l_w (list): list of weights of similar pixels extracted from fine resolution Landsat 8 OLI data
            v (list): List of weights assigned conversion coefficient
            t (list): List of temporal difference
            central_pic_vec (list): central pixel vector for a specific moving window
            vrs1_smlr (list): The VIIRS pixels with a similar location corresponding to Landsat 8 OLI pixels
            within a same window for day 1
            vrs1_vrs2_smlr (list): The VIIRS pixels with a similar location corresponding to Landsat 8 OLI pixels
            within a same window for day 2
            kept as day 2
            vrs3_vrs2_smlr (list): The VIIRS pixels with a similar location corresponding to Landsat 8 OLI pixels
            within a same window for day 2
            kept as day 2
            vrs3_smlr (list): The VIIRS pixels with a similar location corresponding to Landsat 8 OLI pixels
            within a same window for day 3

        Returns:
            the function will return the synthetic central pixel as a vector

    """
    central_pxl_vec_lst = list()
    for weights in l_w:
        pxl_vec = list()
        for l in weights:
            sum1 = 0
            for j in range(len(l)):
                temp_t = t[weights.index(l)][j]
                temp_v = v[weights.index(l)]
                temp_w = l[j]
                if l_w.index(weights) == 0:
                    temp_diff12 = vrs1_vrs2_smlr[weights.index(l)][j] - vrs1_smlr[weights.index(l)][j]
                    sum1 += temp_t * (central_pic_vec[weights.index(l)] + (temp_v * temp_w * temp_diff12))
                else:
                    temp_diff13 = vrs3_vrs2_smlr[weights.index(l)][j] - vrs3_smlr[weights.index(l)][j]
                    sum1 += temp_t * (central_pic_vec[weights.index(l)] + (temp_v * temp_w * temp_diff13))
            pxl_vec.append(sum1)
        central_pxl_vec_lst.append(pxl_vec)

    c_cntrl_pxl_lst = list()
    for values in zip(*central_pxl_vec_lst):
        c_cntrl_pxl_lst.append(values)

    final_central_pxl_lst = list()
    for values in zip(*c_cntrl_pxl_lst):
        final_central_pxl_lst.append(sum(values))

    return final_central_pxl_lst


def creation_of_image(image_vector_lst, lst_all, col):
    """
    This function is explicitly designed for creating a TiFF image from a list consist of all the
    synthetic pixels.

       Args:
           image_vector_lst (list): The list of all the synthetic pixel vector after image fusion.
           col (int): The number of column in the actual subset is less
           lst_all (<class 'osgeo.gdal.Dataset'>): used for getting the projection system.
       Returns:
           It returns a Tiff image

   """
    img_as_lst = list()
    for values in zip(*image_vector_lst):
        img_as_lst.append(values)

    outfile = 'viirs_landsat_fusion_day17'
    temp_lst = list()
    print(col)

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
        out_data.SetProjection(lst_all.GetProjection())
        out_data.SetGeoTransform(lst_all.GetGeoTransform())
        out_data.GetRasterBand(1).WriteArray(arr)
        out_data.FlushCache()


if __name__ == '__main__':
    __main()