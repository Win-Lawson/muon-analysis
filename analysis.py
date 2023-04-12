import math

import numpy as np
import pandas
from matplotlib import pyplot as plt

data = pandas.read_csv('data/weekend-3-23.csv',
                       usecols=['STDC_8_0', 'STDC_8_1', 'STDC_8_2', 'STDC_8_3', 'STDC_8_4', 'STDC_8_5', 'STDC_8_6',
                                'STDC_8_7', 'STDC_8_8', 'STDC_8_SUM'])
data2 = pandas.read_csv('data/weekend-3-23.csv',
                        usecols=['STDC_8_0', 'STDC_8_1', 'STDC_8_2', 'STDC_8_3', 'STDC_8_4', 'STDC_8_5', 'STDC_8_6',
                                 'STDC_8_7', 'STDC_8_15', 'STDC_8_SUM'])
data2.rename(columns={'STDC_8_15': 'STDC_8_8'}, inplace=True)
data = pandas.concat([data, data2])
data = data[data['STDC_8_SUM'] > 1]

up_counts = 0
down_counts = 0
up_list = []
down_list = []
for index, row in data.iterrows():
    row = row[row != 0]
    row = row[row != 1]
    row = row[row != 4095]
    if row.argmax() < row.argmin():
        up_list.append(row.min())
        up_counts += 1
    elif row.argmax() > row.argmin():
        down_list.append(row.min())
        down_counts += 1

print(up_counts)
print(down_counts)

nbins = 35
n_up, bin_edges = np.histogram(up_list, nbins)  # Make a histogram
n_down, bin_edges = np.histogram(down_list, nbins)

the_thing_to_plot = n_up - n_down

fig, ax = plt.subplots(figsize=[10, 7])
x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
x *= (8 / 4096)
errs = np.sqrt(math.sqrt(n_up ^ 2 + n_down ^ 2))

ax.errorbar(x, the_thing_to_plot, errs, fmt='o', markersize=3.0, capsize=1.0, label='Data');
ax.set_title('Up counts - down counts vs time')
ax.set_xlabel('Time ($\mu$s)')
ax.set_ylabel('Up counts - down counts');
plt.show()
