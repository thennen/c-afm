
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from numpy.random import rand
from scipy.ndimage.filters import gaussian_filter

df = pd.read_pickle('all_lcafm_data.pd')
df['run'] = np.int8(df['run'])
df['cycle'] = np.int8(df['cycle'])
scans = df[df.type == 'xy']
scans = scans[scans.folder == '09-Jun-2017']
scans = scans[scans.run <= 18]
scans = scans[scans.run >= 12]
scans = scans[scans.aspect == 1]
scans = scans.sort_values('voltage')


current_scans = scans[scans.channel_name == 'I']
topo_scans = scans[scans.channel_name == 'Z']
#data = np.stack(current_scans.scan, -1)
data = np.stack(topo_scans.scan, -1)

def gauss(input):
    return gaussian_filter(input, sigma=2)
filtdata = np.stack(current_scans.scan.apply(gauss), -1)

V = current_scans.voltage
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
imnum = 4
def showim(num):
    ax1.imshow(data[:, :, num], interpolation='none')
    ax1.set_title('scan voltage: {}'.format(V.iloc[num]))
    print('lol')
showim(imnum)

def on_press(event):
    print('you pressed', event.button, event.xdata, event.ydata)
    j = int(event.xdata)
    i = int(event.ydata)
    if event.button == 1:
        # left mouse, plot IV alone
        ax2.cla()
        ax2.set_xlabel('Voltage')
        ax2.set_ylabel('Current')
        ax2.plot(V, filtdata[i, j])
    if event.button == 3:
        # right mouse, add IV to plot
        ax2.plot(V, filtdata[i, j])
    if event.button == 2:
        # Middle mouse, cycle through imshows
        print('lol?')
        global imnum
        imnum += 1
        print(imnum)
        showim(imnum % len(V))
    plt.pause(.1)

cid = fig.canvas.mpl_connect('button_press_event', on_press)

plt.show()
