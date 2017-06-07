import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Just plopping down some code fragments

df = pd.read_pickle('all_lcafm_data.pd')
# Subtract a plane from topography so you don't have to keep doing it later
def correcttopo(series):
    if series['channel_name'] == 'Z':
        series['corrscan'] = 1e9 * series['scan'] - fitplane(series['scan'])
        return series
    else: return series
df = df.apply(correcttopo, 1)

#niceones = df[df.filename.apply(lambda fn: ('21_1' in fn) and ('May' in fn))]
'''
# Aggregate histogram
figure()
allpixels = np.array([])
for im in df[df.channel_name == 'I'].data:
    allpixels = np.append(allpixels, im.flatten())

hist(allpixels, bins=50, range=(-1e-7, 2e-7))

##
# Stacked Histograms

figure()
for im in df[df.channel_name == 'I'].data:
    plt.hist(im.flatten(), alpha=.3, normed=True)

##

# Make some scatter plots of current vs height
for k,g in df.groupby('filename'):
    figure()
    current = g.channel_name == 'I'
    topo = g.channel_name == 'Z'
    if any(topo) and any(current):
        if np.shape(g[current]['data'].iloc[0])[1] > 0:
            xdata = g[current]['data'].iloc[0].flatten()
            ydata = g[topo]['data'].iloc[0].flatten()
            scatter(xdata, ydata, alpha=.1, edgecolor='none')
            xlim((min(xdata), max(xdata)))
            ylim((min(ydata), max(ydata)))
            plt.savefig(k)
'''

def plot_cafm(data):
    # Data should be a dataframe with one of type 'I' and one of type 'Z'
    # This makes a plot of topography next to current
    topo_cm = 'viridis'
    current_cm = 'inferno'
    # Pandas doesn't know this should only have one row, so you need to tell it
    # to take the 0th row
    I = data[data['channel_name'] == 'I'].iloc[0]
    Z = data[data['channel_name'] == 'Z'].iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22.5,8))
    left = 0
    right = I['width'] * 1e9
    bottom = 0
    top = I['height'] * 1e9
    Idata = I['scan']
    Zdata = Z['scan']
    # Correct the Z data
    subZ = Zdata - fitplane(Zdata)
    corrZ = 1e9 * (subZ - np.min(subZ))
    # Plot topography image
    p1, p99 = np.percentile(corrZ, (0.2, 99.8))
    im1 = ax1.imshow(corrZ, cmap=topo_cm, vmin=p1, vmax=p99, extent=(left, right, bottom, top))
    ax1.set_xlabel('X [nm]')
    ax1.set_ylabel('Y [nm]')
    ax1.set_title('Sample: {},  Folder: {},  id: {}'.format(I['sample_name'], I['folder'], I['id']))
    ax2.set_title('Tip voltage: {} V'.format(I['voltage']))
    fig.colorbar(im1, ax=ax1, label='Height [nm]')
    # Plot current image
    Idata
    p1, p99 = np.percentile(1e9 * Idata, (0.2, 99.8))
    im2 = ax2.imshow(1e9 * Idata, cmap=current_cm, vmin=p1, vmax=p99, extent=(left, right, bottom, top))
    ax2.set_xlabel('X [nm]')
    ax2.set_ylabel('Y [nm]')
    fig.colorbar(im2, ax=ax2, label='Current [nA]')
    return fig


def plot_cafm_hist(data):
    # Data should be a dataframe with one of type 'I' and one of type 'Z'
    # This makes a plot of topography next to current
    # Pandas doesn't know this should only have one row, so you need to tell it
    # to take the 0th row
    I = data[data['channel_name'] == 'I'].iloc[0]
    Z = data[data['channel_name'] == 'Z'].iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22.5,8))
    Idata = I['scan']
    Zdata = Z['scan']
    # Correct the Z data
    subZ = Zdata - fitplane(Zdata)
    corrZ = 1e9 * (subZ - np.min(subZ))
    # Make height histogram
    p1, p99 = np.percentile(corrZ, (0.2, 99.8))
    hist1 = ax1.hist(corrZ.flatten(), bins=100, range=(p1, p99), color='ForestGreen')
    ax1.set_xlabel('Height [nm]')
    ax1.set_ylabel('Pixel Count')
    ax1.set_title('Sample: {},  Folder: {},  id: {}'.format(I['sample_name'], I['folder'], I['id']))
    # Make current histogram
    p1, p99 = np.percentile(Idata, (0.2, 99.8))
    hist2 = ax2.hist(Idata.flatten(), bins=100, range=(p1, p99), color='Crimson')
    ax2.set_title('Tip voltage: {} V'.format(I['voltage']))
    ax2.set_xlabel('Current [nA]')
    ax2.set_ylabel('Pixel Count')
    return fig


def fitplane(Z):
    # Plane Regression -- basically took this from online somewhere
    # probably I messed up the dimensions as usual
    m, n = np.shape(Z)
    X, Y = np.meshgrid(np.arange(m), np.arange(n))
    XX = np.hstack((np.reshape(X, (m*n, 1)) , np.reshape(Y, (m*n, 1)) ) )
    XX = np.hstack((np.ones((m*n, 1)), XX))
    ZZ = np.reshape(Z, (m*n, 1))
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(XX.transpose(), XX)), XX.transpose()), ZZ)
    plane = np.reshape(np.dot(XX, theta), (m, n))
    return plane


if __name__ == '__main__':
    # Make nice subplots of each scan (that has aspect ratio close to 1)
    for folder, folderdata in df[df['type'] == 'xy'].groupby('folder'):
        plotfolder = os.path.join(folder, 'subplots')
        if not os.path.isdir(plotfolder):
            os.makedirs(plotfolder)
        for id, data in folderdata.groupby('id'):
            h, w = np.shape(data.iloc[0]['scan'])
            if w != 0:
                ratio = float(h)/w
            else: ratio = 0
            if 0.8 < ratio < 1.2:
                fig = plot_cafm(data)
                #fn = os.path.splitext(data.iloc[0]['filename'])[0]
                #Just use the ID as a file name
                fn = id
                savepath = os.path.join(plotfolder, fn)
                fig.savefig(savepath, bbox_inches=0)
                plt.close(fig)

    # Scatter height vs current
    for folder, folderdata in df[df['type'] == 'xy'].groupby('folder'):
        scatterfolder = os.path.join(folder, 'height_current_scatter')
        if not os.path.isdir(scatterfolder):
            os.makedirs(scatterfolder)
        for id, data in folderdata.groupby('id'):
            h, w = np.shape(data.iloc[0]['scan'])
            if w != 0:
                ratio = float(h)/w
            else: ratio = 0
            if 0.8 < ratio < 1.2:
                fig, ax = plt.subplots()
                I = data[data['channel_name'] == 'I'].iloc[0]['scan'] * 1e9
                Z = data[data['channel_name'] == 'Z'].iloc[0]['scan'] * 1e9
                Z = Z - fitplane(Z)
                ax.scatter(I.flatten(), Z.flatten(), alpha=.1)
                ax.set_xlabel('Current [nA]')
                ax.set_ylabel('Height [nm]')
                #Just use the ID as a file name
                fn = id
                ax.set_title('{}, {}'.format(folder, id))
                savepath = os.path.join(scatterfolder, fn)
                fig.savefig(savepath, bbox_inches=0)
                plt.close(fig)


    # hexbin height vs current
    for folder, folderdata in df[df['type'] == 'xy'].groupby('folder'):
        scatterfolder = os.path.join(folder, 'height_current_hexbin')
        if not os.path.isdir(scatterfolder):
            os.makedirs(scatterfolder)
        for id, data in folderdata.groupby('id'):
            h, w = np.shape(data.iloc[0]['scan'])
            if w != 0:
                ratio = float(h)/w
            else: ratio = 0
            if 0.8 < ratio < 1.2:
                fig, ax = plt.subplots()
                I = data[data['channel_name'] == 'I'].iloc[0]['scan'] * 1e9
                Z = data[data['channel_name'] == 'Z'].iloc[0]['scan'] * 1e9
                Z = Z - fitplane(Z)
                hex = ax.hexbin(I.flatten(), Z.flatten(), bins='log')
                fig.colorbar(hex, label='log10(count)')
                ax.set_xlabel('Current [nA]')
                ax.set_ylabel('Height [nm]')
                #Just use the ID as a file name
                fn = id
                ax.set_title('{}, {}'.format(folder, id))
                savepath = os.path.join(scatterfolder, fn)
                fig.savefig(savepath, bbox_inches=0)
                plt.close(fig)

    # Histograms of current
    # Make nice subplots of each scan (that has aspect ratio close to 1)
    for folder, folderdata in df[df['type'] == 'xy'].groupby('folder'):
        plotfolder = os.path.join(folder, 'histograms')
        if not os.path.isdir(plotfolder):
            os.makedirs(plotfolder)
        for id, data in folderdata.groupby('id'):
            h, w = np.shape(data.iloc[0]['scan'])
            if w != 0:
                ratio = float(h)/w
            else: ratio = 0
            if 0.8 < ratio < 1.2:
                fig = plot_cafm_hist(data)
                #fn = os.path.splitext(data.iloc[0]['filename'])[0]
                #Just use the ID as a file name
                fn = id
                savepath = os.path.join(plotfolder, fn)
                fig.savefig(savepath, bbox_inches=0)
                plt.close(fig)

    # Polar plot gradient vs current
    # These take a long time for some reason
    for folder, folderdata in df[df['type'] == 'xy'].groupby('folder'):
        scatterfolder = os.path.join(folder, 'gradient_current_polarscatter')
        if not os.path.isdir(scatterfolder):
            os.makedirs(scatterfolder)
        for id, data in folderdata.groupby('id'):
            h, w = np.shape(data.iloc[0]['scan'])
            if w != 0:
                ratio = float(h)/w
            else: ratio = 0
            if 0.8 < ratio < 1.2:
                fig = plt.figure(figsize=(10,10))
                ax = plt.subplot(polar=True)
                I = data[data['channel_name'] == 'I'].iloc[0]['scan'] * 1e9
                Z = data[data['channel_name'] == 'Z'].iloc[0]['scan'] * 1e9
                Z = Z - fitplane(Z)
                gradx, grady = np.gradient(Z)
                angle = np.arctan2(grady, gradx).flatten()
                magn = np.sqrt(gradx**2 + grady**2).flatten()
                Iflat = I.flatten()
                #sc = ax.scatter(angle, magn, c=Iflat, s=10, alpha=.2, edgecolor='none', cmap='rainbow')
                # Scattering all of the points washes out the less frequent points.  Need to normalize number of data points by the histogram
                Irange = np.percentile(I, (.1, 99.9))
                hist, bins = np.histogram(I.flatten(), range=Irange, bins=50)
                # For each bin, if the number of data points is over a certain percentage of the total data, take a random sample
                scatterangle = []
                scattermagn = []
                scatterI = []
                for i, hi in enumerate(hist):
                    p = 0.1
                    binmax = int(p * h * w)
                    mask = (Iflat <= bins[i+1]) & (Iflat >= bins[i])
                    ang = angle[mask]
                    mag = magn[mask]
                    curr = Iflat[mask]
                    inds = range(len(ang))
                    if hi > binmax:
                        inds = np.random.choice(inds, binmax, replace=False)
                    scatterangle.extend(ang[inds])
                    scattermagn.extend(mag[inds])
                    scatterI.extend(curr[inds])
                sc = ax.scatter(scatterangle, scattermagn, c=scatterI, s=10, alpha=.8, edgecolor='none', cmap='rainbow')

                ax.set_rlim(0, np.percentile(magn, 99))
                #ax.set_rlabel_position(10)
                cb = colorbar(sc, label='c-afm current [nA]')
                cb.set_alpha(1)
                cb.draw_all()
                # Save the figure
                fn = id
                ax.set_title('Current vs. Topography gradient: {}, {}'.format(folder, id), y=1.1)
                savepath = os.path.join(scatterfolder, fn)
                fig.savefig(savepath, bbox_inches=0)
                plt.close(fig)


    # Make plot of scan locations.  I don't know why, I just thought it would be cool.
    # Might be useful to annotate them with the measurement id
    from matplotlib import patches
    for folder, folderdata in df.groupby('folder'):
        topo = folderdata[folderdata.channel_name == 'Z']
        fig, ax = plt.subplots()
        miny = 1e6 * np.min(topo.y_offset)
        minx = 1e6 * np.min(topo.x_offset)
        maxx = 1e6 * np.max(topo.x_offset + topo.width)
        maxy = 1e6 * np.max(topo.y_offset + topo.height)
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        topo['patchcolor'] = list(plt.cm.jet(np.linspace(0, 1, len(topo))))
        for row, scan in topo.iterrows():
            startpt = 1e6 * scan.x_offset, 1e6 * scan.y_offset
            width = 1e6 * scan.width
            height = 1e6 * scan.height
            c = scan.patchcolor
            patch = patches.Rectangle(startpt, width, height, alpha=.4, color=c)
            ax.add_patch(patch)
        ax.set_xlabel('X [$\mu$m]')
        ax.set_ylabel('Y [$\mu$m]')
        ax.set_title('Scan areas for {}'.format(folder))

        # Also put in points where IV measurements were taken
        iv = folderdata[folderdata.type == 'iv']
        x = iv['x_offset']
        y = iv['y_offset']
        ax.scatter(x, y, alpha=1, edgecolor='none')

