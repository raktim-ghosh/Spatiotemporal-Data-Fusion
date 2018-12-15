import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gdal

b7_radiance = gdal.Open('D:/Output_temperature_Map/spectral_radiance_of_b7.tiff')
radiance_arr = b7_radiance.GetRasterBand(1).ReadAsArray()

b7_temperature = gdal.Open('D:/Output_temperature_Map/spectral_radiance_of_b6.tiff')
temperature_arr = b7_temperature.GetRasterBand(1).ReadAsArray().astype(float)

b7_sr = gdal.Open('D:/RaktimPorjectData/ProjectData/Subset_LC0814004320180125_DN_sr/lstsubset_20180125_sr_b7.tif')
sr_arr = b7_sr.GetRasterBand(1).ReadAsArray().astype(float)

# import statsmodels.api as sm
# from scipy.stats.stats import pearsonr
#
# get_ipython().run_line_magic('matplotlib', 'inline')
matplotlib.rcParams['axes.labelsize'] = 20
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.rcParams['text.color'] = 'k'
matplotlib.rcParams['font.family'] = 'arial'
matplotlib.rcParams['figure.figsize'] = 16, 8

# In[12]:

radiance_245_line = radiance_arr[245: 246, 813: 863].flatten()
reflectance_245_line = sr_arr[245: 246, 813: 863].flatten()
# temperature_245_line = temperature_arr[245: 246, 813: 863].flatten()
# print(np.shape(radiance_245_line))
print(radiance_245_line)
print(reflectance_245_line)
# print(temperature_245_line)
# print(np.amax(radiance_245_line))
# print(np.amin(radiance_245_line))
# print(np.amax(temperature_245_line))
# print(np.amin(temperature_245_line))

l2 = np.array([i for i in range(50)])
print(np.shape(l2))

color = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080',
         '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
         '#ffffff', '#000000']

# plt.subplot(1,1,1)
# plt.title('Pixel location vs Spectral radiance', fontsize=22, fontweight='bold')
# color1 = '#'
color2 = 'tab:red'
color3 = 'yellow'
# color4 = 'cyan'
# color5 = 'magenta'
# plt.xlabel('Pixel Location', fontweight= 'bold')
# plt.ylabel('Spectral Radiance (W/ m2 * srad * μm)', fontweight= 'bold')
# plt.plot(l2, radiance_245_line,'--o', color=color2, markersize= 7,linewidth=3)
# plt.show()

fig, ax1 = plt.subplots()
plt.title('Scanline of a coal fire affected area', fontsize=22, fontweight='bold')

ax1.set_xlabel('Landsat 8 OLI (band 6 - 1.6 μm)', fontweight= 'bold')
# ax1.set_xticklabels(l2, rotation=60)
ax1.set_ylabel('Kinetic Temperature (ºC)', color=color2, fontweight= 'bold')
ax1.plot(l2, radiance_245_line, '--ok', color=color2, markersize = 5,linewidth=3)
ax1.tick_params(axis='y', labelcolor=color2)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color4 = 'tab:blue'
ax2.set_ylabel('Spectral Radiance (W/ m2 * srad * μm)', color=color4,fontweight= 'bold')
# we already handled the x-label with ax1
ax2.plot(l2, reflectance_245_line, '--ok', color=color4, markersize = 5)
ax2.tick_params(axis='y', labelcolor=color4)


fig.tight_layout()  # otherwise the right y-label is slightly clipped

# fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes, fontsize ='xx-large')
props = dict(boxstyle='square', facecolor='white', alpha = 0.4)
textstr1 = ('Pixel (min): ' + str(1) + '; DN (max): ' + str(50))
textstr2 = ('Temperature (min): ' + str(334.36) + '; Temperature (max): ' + str(399.36))
textstr3 = ('Radiance (min): ' + str(9.08) + '; Radiance (max): ' + str(21.60))
ax1.text(0.6,0.7, textstr1, transform=ax1.transAxes, fontsize=16,verticalalignment='center', bbox=props)
ax1.text(0.6,0.6, textstr2, transform=ax1.transAxes, fontsize=16,verticalalignment='center', bbox=props)
ax1.text(0.6,0.5, textstr3, transform=ax1.transAxes, fontsize=16,verticalalignment='center', bbox=props)
plt.show()