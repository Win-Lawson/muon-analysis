import numpy as np
import pandas
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

data1 = pandas.read_csv('data/3-23-23-CUplates-magnet.csv',
                        usecols=['STDC_8_0', 'STDC_8_1', 'STDC_8_2', 'STDC_8_3', 'STDC_8_4', 'STDC_8_5', 'STDC_8_6',
                                 'STDC_8_7', 'STDC_8_8', 'STDC_8_SUM'])
data2 = pandas.read_csv('data/3-30-23-CUplates-magnet.csv',
                        usecols=['STDC_8_0', 'STDC_8_1', 'STDC_8_2', 'STDC_8_3', 'STDC_8_4', 'STDC_8_5', 'STDC_8_6',
                                 'STDC_8_7', 'STDC_8_15', 'STDC_8_SUM'])
data2.rename(columns={'STDC_8_15': 'STDC_8_8'}, inplace=True)
data3 = pandas.read_csv('data/4-4-23-CUplates-magnet.csv',
                        usecols=['STDC_8_0', 'STDC_8_1', 'STDC_8_2', 'STDC_8_3', 'STDC_8_4', 'STDC_8_5', 'STDC_8_6',
                                 'STDC_8_7', 'STDC_8_15', 'STDC_8_SUM'])
data3.rename(columns={'STDC_8_15': 'STDC_8_8'}, inplace=True)
data = pandas.concat([data1, data2, data3])
data = data[data['STDC_8_SUM'] > 1]

up_list = []
down_list = []
for index, row in data.iterrows():
    row = row[row != 0]
    row = row[row != 1]
    row = row[row != 4095]
    if row.argmax() < row.argmin():
        up_list.append(row.min())
    elif row.argmax() > row.argmin():
        down_list.append(row.min())

n_bins = 20
n_up, bin_edges1 = np.histogram(up_list, n_bins)  # Make a histogram
n_down, bin_edges2 = np.histogram(down_list, n_bins)

up_minus_down = n_up - n_down


def damped_sine(x, b, c, d, e, f):
    return b * np.exp(-x / c) + f * np.sin(d * x + e)


bin_centers = 0.5 * (bin_edges1[1:] + bin_edges1[:-1])
bin_centers *= (8 / 4096)
errs = np.sqrt(n_down ^ 2 + n_up ^ 2)
initial_parameters = np.array([120,4,3,-4,20])
popt, pcov = curve_fit(damped_sine, bin_centers, up_minus_down, sigma=errs, p0=initial_parameters)

fig, ax = plt.subplots(figsize=[10, 7])
ax.errorbar(bin_centers, up_minus_down, errs, fmt='o', markersize=3.0, capsize=1.0, label='Data')
ax.set_title('Up counts - down counts vs time')
ax.set_xlabel('Time ($\mu$s)')
ax.set_ylabel('Up counts - down counts')
ax.plot(bin_centers, damped_sine(bin_centers, *popt), 'r-', label='fit')
print(*popt)
plt.show()

# up only
errs = np.sqrt(n_up)
popt_up_only, pcov_up_only = curve_fit(damped_sine, bin_centers, n_up, sigma=errs)

fig, ax = plt.subplots(figsize=[10, 7])
ax.errorbar(bin_centers, up_minus_down, errs, fmt='o', markersize=3.0, capsize=1.0, label='Data')
ax.set_title('Up counts vs time')
ax.set_xlabel('Time ($\mu$s)')
ax.set_ylabel('Up counts - down counts')
ax.plot(bin_centers, damped_sine(bin_centers, *popt), 'r-', label='fit')
print(*popt)
plt.show()
