import gdal
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from numpy.random import multivariate_normal
from scipy.stats import gaussian_kde


def __main():
    """ Frame of Landsat 8 OLI data for visualisation """
    lst_b6 = gdal.Open('D:/RaktimPorjectData/erdas_subset_lst_140043_2018025_sr.tif')
    print('hello')
    b6_arr = lst_b6.GetRasterBand(1).ReadAsArray().astype(float)
    b6_sr = b6_arr * 0.0001
    print(b6_sr[:3, :3])

    synthetic = gdal.Open(r'D:/Outputof_VIIRS_Landsat_Fusion/viirs_landsat_fusion_day25_window41.tiff')
    print('hello')
    synt_arr = synthetic.GetRasterBand(1).ReadAsArray().astype(float)
    synt_sr = synt_arr * 0.0001
    print(synt_sr[:3, :3])

    get_covariance(b6_sr, synt_sr)
    get_correlation(b6_sr, synt_sr)
    get_rmse(lst_b6, b6_sr, synt_sr)


def get_covariance(inp, predicted):
    """
    This function will determine the variability of predicted/synthetic image with respect to the input image


        Args:
            inp (array): the input image with
            predicted : This is the predicted image of the particular date

        Return:
            the function will return the covariance

    """
    row_inp = np.shape(inp)[0]
    col_inp = np.shape(inp)[1]

    row_prd = np.shape(predicted)[0]
    col_prd = np.shape(predicted)[1]

    refined_input = inp[23:row_inp - 23, 23:col_inp - 23]

    refined_output = predicted[3: row_prd - 3, 3:col_prd - 3]

    x = refined_input.flatten()
    y = refined_output.flatten()

    fig1 = plt.figure()
    plt.plot(x, y, '.r')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')

    # Estimate the 2D histogram
    nbins = 200
    H, xedges, yedges = np.histogram2d(x, y, bins=nbins)

    # H needs to be rotated and flipped
    H = np.rot90(H)
    H = np.flipud(H)

    # Mask zeros
    Hmasked = np.ma.masked_where(H == 0, H)  # Mask pixels with a value of zero

    # Plot 2D histogram using pcolor
    fig2 = plt.figure()
    plt.pcolormesh(xedges, yedges, Hmasked)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Counts')
    plt.show()

    """new testing"""
    """
    data = np.vstack([refined_input, refined_output])

    gammas = [0.8, 0.5, 0.3]

    fig, axes = plt.subplots(nrows=2, ncols=2)

    axes[0, 0].set_title('Linear normalization')
    axes[0, 0].hist2d(data[:, 0], data[:, 1], bins=100)
    print(np.shape(data[:, 0]))

    for ax, gamma in zip(axes.flat[1:], gammas):
        ax.set_title(r'Power law $(\gamma=%1.1f)$' % gamma)
        ax.hist2d(data[:, 0], data[:, 1],
                  bins=100, norm=mcolors.PowerNorm(gamma))

    fig.tight_layout()

    plt.show()
    """

    sum1 = 0
    mean_inp = np.mean(refined_input)
    mean_prd = np.mean(refined_output)

    for i in range(np.shape(refined_input)[0]):
        for j in range(np.shape(refined_input)[1]):
            sum1 += (refined_input[i, j] - mean_inp) * (refined_output[i, j] - mean_prd)

    cov = sum1 / (row_prd * col_prd - 1)

    print(cov, 'the covariance between two images')
    return cov


def get_correlation(inp, predicted):
    """
    This function will determine the correlation coefficient between predicted/synthetic image with respect to the
    input image.


        Args:
            inp (array): the input image with
            predicted : This is the predicted image of the particular date

        Return:
            the function will return the coefficient of determination.

    """
    row_inp = np.shape(inp)[0]
    col_inp = np.shape(inp)[1]

    row_prd = np.shape(predicted)[0]
    col_prd = np.shape(predicted)[1]

    refined_input = inp[23:row_inp - 23, 23:col_inp - 23]

    refined_output = predicted[1: row_prd - 1, 1:col_prd - 1]

    sum1 = 0
    sum2 = 0
    sum3 = 0
    mean_inp = np.mean(refined_input)
    mean_prd = np.mean(refined_output)

    for i in range(np.shape(refined_input)[0]):
        for j in range(np.shape(refined_input)[1]):
            sum1 += (refined_input[i, j] - mean_inp) * (refined_output[i, j] - mean_prd)
            sum2 += (refined_input[i, j] - mean_inp) ** 2
            sum3 += (refined_output[i, j] - mean_prd) ** 2

    r = sum1 / (math.sqrt(sum2) * math.sqrt(sum3))

    print(r ** 2, 'the correlation coefficient of two images')
    return r ** 2


def get_rmse(lst_b6, inp, predicted):
    """
    This function will determine the root mean square error between predicted/synthetic image with respect to the
            input image.

        Args:
            inp (array): the input image with
            predicted : This is the predicted image of the particular date

        Return:
            the function will return the coefficient of determination.

    """
    row_inp = np.shape(inp)[0]
    col_inp = np.shape(inp)[1]

    row_prd = np.shape(predicted)[0]
    col_prd = np.shape(predicted)[1]

    refined_input = inp[23:row_inp - 23, 23:col_inp - 23]

    refined_output = predicted[4: row_prd - 4, 4:col_prd - 4]

    mean_inp = np.mean(refined_input)
    mean_prd = np.mean(refined_output)

    plt.scatter(refined_input, refined_output)
    plt.show()
    sum1 = 0
    sum2 =0
    for i in range(np.shape(refined_input)[0]):
        for j in range(np.shape(refined_input)[1]):
            sum1 += (refined_input[i, j] - refined_output[i, j]) ** 2
            sum2 += abs(refined_input[i, j] - refined_output[i, j]) / (row_prd * col_prd)

    rmse = sum1 / ((row_prd * col_prd) - 1)

    print(rmse, 'the root mean square error of two images')
    print(sum2, 'absolute average difference')
    img = refined_input - refined_output
    cols, rows = img.shape
    driver = gdal.GetDriverByName("GTiff")
    outfile = 'difference_img_window45'
    outfile += '.tiff'
    out_data = driver.Create(outfile, rows, cols, 1, gdal.GDT_Float32)
    out_data.SetProjection(lst_b6.GetProjection())
    out_data.SetGeoTransform(lst_b6.GetGeoTransform())
    out_data.GetRasterBand(1).WriteArray(img)
    out_data.FlushCache()
    return rmse


if __name__ == '__main__':
    __main()