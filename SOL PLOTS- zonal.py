
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import gdal

b7_radiance = gdal.Open('D:/Output_temperature_Map/spectral_radiance_of_b7.tiff')
radiance_arr = b7_radiance.GetRasterBand(1).ReadAsArray().astype(float)

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

radiance_245_line = radiance_arr[245: 246, 815: 865]
print(np.shape(radiance_245_line))
print(radiance_245_line)

l2 = np.array([i for i in range(50)])
print(l2)

l1 = np.array([0.4, 0.6, 0.8, 1.0, 1.6, 2.0, 2.2, 4.0, 6.0, 8.0, 10.0, 12.0])

radiance_100 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0006, 0.0029])

radiance_150 = np.array([0, 0, 0, 0, 0.006, 0.14, 0.43, 23.37, 52.74, 52.74, 40.93, 29.80])

radiance_200 = np.array([0, 0, 0, 0, 0.06, 0.90, 2.24, 57.49, 96.37, 82.69, 59.56, 41.12])

radiance_250 = np.array([0, 0, 0, 0.0001, 0.37, 3.88, 8.44, 119.11, 157.18, 120.18, 81.06, 53.68])

radiance_300 = np.array([0, 0, 0, 0.001, 1.70, 12.93, 25.19, 217.42, 235.83, 164.21, 105.03, 67.26])

radiance_350 = np.array([0, 0, 0.0001, 0.010, 6.01, 35.48, 63.03, 360.51, 332.14, 214.09, 131.09, 81.67])

radiance_400 = np.array([0, 0, 0.0008, 0.060, 17.60, 83.77, 137.63, 554.85, 445.46, 269.17, 158.94, 96.78])

radiance_450 = np.array([0, 0.005, 0.26, 44.40, 175.62, 269.77, 805.08, 574.79, 328.83, 112.47])


dn = np.array([4935, 7965, 10995, 14025, 17064, 20062, 23413, 26901, 29887, 32330, 35163, 38453, 41989, 44699,
               48578, 53422, 57899, 58311, 62623, 65535])

radiance1 = np.array([2.60, 4.21, 5.81, 7.42, 9.022, 10.607, 12.37, 14.22, 15.80, 17.094, 18.59, 20.33, 22.20,
                      23.63, 25.68, 28.24, 30.614, 30.831, 33.111, 34.65])

temperature = np.array([150.75, 185.137, 209.23, 224.36, 235.68, 244.69, 253.13, 260.63, 266.31, 270.54, 275.07,
                        279.91, 284.68, 288.094, 292.65, 297.90, 302.37, 302.77, 306.77, 309.34])

dn2 = np.array([5180, 6180, 7180, 8180, 9180, 10180, 11180, 12180, 13180, 14180, 15180, 16180, 17182, 18183, 19211,
                20237])

radiance2 = np.array([0.282, 1.851, 3.415, 4.989, 6.557, 8.128, 9.69, 11.264, 12.833, 14.401, 15.970, 17.539, 19.111,
                      20.681, 22.294, 23.904])

temperature2 = np.array([243.17, 305.51, 328.97, 344.58, 356.36, 365.92, 374.00, 381.03, 387.36, 392.89, 398.01,
                         402.71, 407.08, 411.16, 415.07, 418.75])

zones = np.array(['MEERUT','GHAZIABAD','SAHARANPUR','NOIDA','MORADABAD','BAREILLY','LUCKNOW','FAIZABAD','LESA','AGRA',
                'ALIGARH','KANPUR','JHANSI','BANDA','ALLAHABAD','GORAKHPUR','VARANASI','AZAMGARH','KESCO'])
print(np.shape(zones))
# con = np.array([3651.496,7147.589,5109.097,3801.506,4436.581,2645.416,2809.8,3571.417,5148.481,5848.044,3498.357,
#                 3582.59,2040.587,1459.824,4992.233,3566.099,5940.666,2296.914,3572.728])
# V_mean = np.array([33652.188,61216.43,44949.62,70590.4,59331.016,52461.465,64985.293,72170.07,65445.65,104096.29,
#                    44578.766,52693.7,39876.684,27307.385,74508.49,58282.785,84484.74,31296.473,30154.771])
# V_med = np.array([29884.95,56927.453,40509.06,71481,56073.297,49835.56,63843.016,71778.67,65193.04,99308.33,42878.15,
#                   54152.33,38420.555,27108.754,69632.336,57393.316,87107.94,30909.344,30280.02])
# V_max = np.array([82348.875,137484.11,99743.13,127953.87,125234.02,121946.2,144144.78,154771.62,154466.38,220779.42,
#                   94917.64,102429.88,84874.92,58596.85,176909,131349.9,175997.4,63139.703,71992.17])
# V_90P = np.array([52391.33,91505.17,71819.49,98488.78,93767.98,82851.06,105371.49,117424.01,99471.22,159173.92,
#                   70626.85,80796.42,64525.19,45221.77,134421.13,94698.98,132254.59,49565.85,44121.58])
# V_meanT = np.array([36163.72,63436.29,47323.37,72035.89,62015.88,55094.35,71653.12,79318.77,68051.46,108117.38,47211.98,
#                     56420.35,40326.45,29021.07,77494.29,64994.97,92448.09,35897.36,31256.64])
# V_medT = np.array([36916.98,63928.80,48197.38,71715.30,62641.21,55891.40,73266.52,80488.81,67536.55,108848.59,47867.04,
#                    56190.02,40124.64,29235.15,76159.28,64750.66,91860.30,35505.02,30962.74])
# V_maxT = np.array([40222.82,69619.40,52423.75,80132.66,69251.62,62275.45,84209.37,92485.07,76987.95,118714.97,52465.70,
#                    65336.22,45972.00,33602.35,92279.78,76508.77,107235.55,42956.88,34908.04])
# V_pca = np.array([128676.625,232755.06,165480.02,260677.52,221514.75,192080.52,238008.14,269418.3,259616.67,391975.38,
#                   166786.38,196556.11,143230.36,96783.59,236124.05,218892.61,313711,117728.016,119028.69])
# DMSP_un = np.array([62354,93262,80112,67390,95967,57992,75188,86076,67065,177518,66311,85385,46659,32375,117782,79452,
#                     108707,33218,33722])

# In[72]:

fig, ax1 = plt.subplots()
plt.title('Relationship between Spectral Radiance, Kinetic Temperature and DN value', fontsize=22, fontweight='bold')


color = 'tab:red'
ax1.set_xlabel('DN Landsat 8 OLI (band 7 - 2.20 μm)', fontweight= 'bold')
ax1.set_xticklabels(dn, rotation=60)
ax1.set_ylabel('Spectral Radiance (W/ m2 * srad * μm)', color=color, fontweight= 'bold')
ax1.plot(dn, radiance1,'--ok', color=color, markersize = 10,linewidth=3)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Kinetic Temperature (ºC)', color=color,fontweight= 'bold')
# we already handled the x-label with ax1
ax2.plot(dn, temperature, '--ok', color=color, markersize = 10)
ax2.tick_params(axis='y', labelcolor=color)


fig.tight_layout()  # otherwise the right y-label is slightly clipped

# fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes, fontsize ='xx-large')
props = dict(boxstyle='square', facecolor='white', alpha = 0.4)
textstr1 = ('DN (min): ' + str(4935) + '; DN (max): ' + str(65535))
textstr2 = ('Temperature (min): ' + str(150.75) + '; Temperature (max): ' + str(309.34))
textstr3 = ('Radiance (min): ' + str(2.60) + '; Radiance (max): ' + str(34.65))
ax1.text(0.5,0.3, textstr1, transform=ax1.transAxes, fontsize=18,verticalalignment='center', bbox=props)
ax1.text(0.5,0.2, textstr2, transform=ax1.transAxes, fontsize=18,verticalalignment='center', bbox=props)
ax1.text(0.5,0.1, textstr3, transform=ax1.transAxes, fontsize=18,verticalalignment='center', bbox=props)
plt.show()


