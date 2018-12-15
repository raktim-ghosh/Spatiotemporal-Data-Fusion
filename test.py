import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from numpy.random import multivariate_normal

data = np.vstack([
    multivariate_normal([10, 10], [[3, 2], [2, 3]], size=100000),
    multivariate_normal([30, 20], [[2, 3], [1, 3]], size=1000)
])

print(np.shape(data)[0])
gammas = [0.8, 0.5, 0.3]

fig, axes = plt.subplots(nrows=2, ncols=2)

axes[0, 0].set_title('Linear normalization')
axes[0, 0].hist2d(data[:, 0], data[:, 1], bins=100)

for ax, gamma in zip(axes.flat[1:], gammas):
    ax.set_title(r'Power law $(\gamma=%1.1f)$' % gamma)
    ax.hist2d(data[:, 0], data[:, 1],
              bins=100, norm=mcolors.PowerNorm(gamma))

fig.tight_layout()

plt.show()









import math
import gdal

h = 6.627 * 10 ** (-34)
c = 3 * 10 ** 8
l1 = [0.4, 0.6, 0.8, 1.0, 1.6, 2.0, 2.2, 4.0, 6.0, 8.0, 10.0, 12.0]
for i in range(len(l1)):
    k = 1.38 * 10 ** (-23)
    w = l1[i] * 10 ** (-6)
    c1 = math.exp((h * c) / (k * (w) * 100)) - 1
    c2 = 2 * h * c ** (2) * w ** (-5) * 10 ** (-6)
    print(c2 / c1)

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
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

wavelength = np.array([0.4, 0.6, 0.8, 1.0, 1.6, 2.0, 2.2, 4.0, 6.0, 8.0, 10.0, 12.0])

radiance_100 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0006, 0.0029])

radiance_150 = np.array([0, 0, 0, 0, 0.006, 0.14, 0.43, 23.37, 52.74, 52.74, 40.93, 29.80])

radiance_200 = np.array([0, 0, 0, 0, 0.06, 0.90, 2.24, 57.49, 96.37, 82.69, 59.56, 41.12])

radiance_250 = np.array([0, 0, 0, 0.0001, 0.37, 3.88, 8.44, 119.11, 157.18, 120.18, 81.06, 53.68])

radiance_300 = np.array([0, 0, 0, 0.001, 1.70, 12.93, 25.19, 217.42, 235.83, 164.21, 105.03, 67.26])

radiance_350 = np.array([0, 0, 0.0001, 0.010, 6.01, 35.48, 63.03, 360.51, 332.14, 214.09, 131.09, 81.67])

radiance_400 = np.array([0, 0, 0.0008, 0.060, 17.60, 83.77, 137.63, 554.85, 445.46, 269.17, 158.94, 96.78])

radiance_450 = np.array([0, 0, 0.005, 0.26, 44.40, 175.62, 269.77, 805.08, 574.79, 328.83,188.30, 112.47])

color = ['#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#46f0f0', '#f032e6', '#bcf60c', '#fabebe', '#008080',
         '#e6beff', '#9a6324', '#fffac8', '#800000', '#aaffc3', '#808000', '#ffd8b1', '#000075', '#808080',
         '#ffffff', '#000000']
fig, ax1 = plt.subplots()
plt.title('Relationship between Radiant Temperature, Spectral Radiance and Wavelength', fontsize=22, fontweight='bold')
color1 = '#'
color2 = 'tab:blue'
color3 = 'yellow'
color4 = 'cyan'
color5 = 'magenta'
ax1.set_xlabel('Wavelength (μm)', fontweight= 'bold')
ax1.set_ylabel('Spectral Radiance (W/ m2 * srad * μm)', fontweight= 'bold')
ax1.plot(wavelength, radiance_100, '--ok', color=color[9], markersize = 10,linewidth=3)
ax1.plot(wavelength, radiance_150, '--ok', color=color[6], markersize = 10,linewidth=3)
ax1.plot(wavelength, radiance_200, '--ok', color=color[5], markersize = 10,linewidth=3)
ax1.plot(wavelength, radiance_250, '--ok', color=color[4], markersize = 10,linewidth=3)
ax1.plot(wavelength, radiance_300, '--ok', color=color[3], markersize = 10,linewidth=3)
ax1.plot(wavelength, radiance_350, '--ok', color=color[2], markersize = 10,linewidth=3)
ax1.plot(wavelength, radiance_400, '--ok', color=color[1], markersize = 10,linewidth=3)
ax1.plot(wavelength, radiance_450, '--ok', color=color[0], markersize = 10,linewidth=3)
props = dict(boxstyle='square', facecolor='white', alpha = 0.4)
textstr1 = ('DN (min): ' + str(4935) + '; DN (max): ' + str(65535))
textstr2 = ('Temperature (min): ' + str(150.75) + '; Temperature (max): ' + str(309.34))
textstr3 = ('Radiance (min): ' + str(2.60) + '; Radiance (max): ' + str(34.65))
ax1.text(11, 800, '------',verticalalignment='bottom', horizontalalignment='right',
        color=color[0], fontsize=20)
ax1.text(12, 800, '450 ºC', verticalalignment='bottom', horizontalalignment='right', fontsize=15)

ax1.text(11, 750, '------', verticalalignment='bottom', horizontalalignment='right',
        color=color[1], fontsize=20)
ax1.text(12, 750, '400 ºC', verticalalignment='bottom', horizontalalignment='right', fontsize=15)
ax1.text(11, 700, '------', verticalalignment='bottom', horizontalalignment='right',
        color=color[2], fontsize=20)
ax1.text(12, 700, '350 ºC', verticalalignment='bottom', horizontalalignment='right', fontsize=15)
ax1.text(11, 650, '------', verticalalignment='bottom', horizontalalignment='right',
        color=color[3], fontsize=20)
ax1.text(12, 650, '300 ºC', verticalalignment='bottom', horizontalalignment='right', fontsize=15)
ax1.text(11, 600, '------', verticalalignment='bottom', horizontalalignment='right',
        color=color[4], fontsize=20)
ax1.text(12, 600, '250 ºC', verticalalignment='bottom', horizontalalignment='right', fontsize=15)
ax1.text(11, 550, '------', verticalalignment='bottom', horizontalalignment='right',
        color=color[5], fontsize=20)
ax1.text(12, 550, '200 ºC', verticalalignment='bottom', horizontalalignment='right', fontsize=15)
ax1.text(11, 500, '------', verticalalignment='bottom', horizontalalignment='right',
        color=color[6], fontsize=20)
ax1.text(12, 500, '150 ºC', verticalalignment='bottom', horizontalalignment='right', fontsize=15)
ax1.text(11, 450, '------', verticalalignment='bottom', horizontalalignment='right',
        color=color[9], fontsize=20)
ax1.text(12, 450, '100 ºC', verticalalignment='bottom', horizontalalignment='right', fontsize=15)
plt.show()