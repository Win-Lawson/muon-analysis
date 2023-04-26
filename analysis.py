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


def damped_sine(t, a, tau, omega, phi, b):
    return a * np.exp(-t / tau) + b * np.sin(omega * t + phi)


def prof_fn(t, a, tau, omega, phi, b):
    return a * np.exp(-t / tau) * (1 + b * np.sin(omega * t + phi))


bin_centers = 0.5 * (bin_edges1[1:] + bin_edges1[:-1])
bin_centers *= (8 / 4096)
errs = np.sqrt(n_down ^ 2 + n_up ^ 2)
initial_parameters = np.array([120, 4, 3, -4, 20])
popt, pcov = curve_fit(prof_fn, bin_centers, up_minus_down, sigma=errs, p0=initial_parameters)

fig, ax = plt.subplots(figsize=[10, 7])
ax.errorbar(bin_centers, up_minus_down, errs, fmt='o', markersize=3.0, capsize=1.0)
ax.set_title('Up counts - down counts vs time')
ax.set_xlabel('Time ($\mu$s)')
ax.set_ylabel('Up counts - down counts')
smooth = np.linspace(bin_centers[0], bin_centers[-1], 100)
ax.plot(smooth, prof_fn(smooth, *popt), 'r-', label='$Ae^{-t/\\tau}(1+B\sin{(\omega t+\phi)}$)')
plt.legend(fontsize=20)
plt.show()

w = popt[2] * 1e6
w_err = np.sqrt(pcov[2][2] * 1e6)
e = 1.602176634e-19
B = 4.93e-3
B_err = (.1 + .996) * 1e-3
m_muon = 1.883531627e-28

g_2 = 2 * m_muon * w / (e * B)
g_2_err = (2 * m_muon / (e * B)) * np.sqrt(w_err ** 2 + (w * B_err / B) ** 2)

print('w = ' + str(w) + ' +/- ' + str(w_err))
print('g minus 2 = ' + str(g_2) + ' +/- ' + str(g_2_err))
