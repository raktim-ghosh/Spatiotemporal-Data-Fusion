import gdal
import numpy as np

sr = gdal.Open('D:/RaktimPorjectData/ProjectData/Subset_LC0814004320180125_DN_sr/lstsubset_20180125_sr_b7.tif')
sr_arr = sr.GetRasterBand(1).ReadAsArray().astype(float)
scaled_sr_arr = sr_arr * 0.0001
print(scaled_sr_arr[:3, :3])

radiance = gdal.Open('D:/Output_temperature_Map/spectral_radiance_of_b7.tiff')
radiance_arr = radiance.GetRasterBand(1).ReadAsArray().astype(float)
print(radiance_arr[:3, :3])

print(np.mean(radiance_arr[225:275, 815:865]))

" The emiited radiance wiil be estimated based on the subtractive component from reflected + emitted radiance"

rows = np.shape(sr_arr)[0]
cols = np.shape(sr_arr)[1]

emtd_rad_arr = np.zeros((rows, cols))

outfile = 'emitted_radiance_b7'

for i in range(rows):
    for j in range(cols):
        emtd_rad_arr[i, j] = radiance_arr[i, j] - 7.3


"""
cols1, rows1 = emtd_rad_arr.shape
driver = gdal.GetDriverByName("GTiff")
outfile += '.tiff'
out_data = driver.Create(outfile, rows1, cols1, 1, gdal.GDT_Float32)
out_data.SetProjection(sr.GetProjection())
out_data.SetGeoTransform(sr.GetGeoTransform())
out_data.GetRasterBand(1).WriteArray(emtd_rad_arr)
out_data.FlushCache()
"""