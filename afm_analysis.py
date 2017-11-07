import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys
import scipy

# Not storing all the data in one file anymore
#df = pd.read_pickle('all_lcafm_data.pd')

# fitplane below for some reason does not do a perfect job every time
'''
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
'''
# Idea from https://gist.github.com/amroamroamro/1db8d69b4b65e8bc66a6
def fitplane(Z, X=None, Y=None, returncoeffs=False):
    ''' Z is 2D, can optionally pass x, y 2d arrays, if you don't want to use default meshgrid'''
    # Can easily be extended to X, Y, Z data that is not on a regular grid!
    m, n = np.shape(Z)
    if X is None:
        X, Y = np.meshgrid(np.arange(m), np.arange(n), indexing='ij')
    XX = X.flatten()
    YY = Y.flatten()
    ZZ = Z.flatten()
    A = np.c_[XX, YY, np.ones(len(XX))]
    C,_,_,_ = scipy.linalg.lstsq(A, ZZ)
    if returncoeffs:
        return C
    else:
        plane = C[0] * X + C[1] * Y + C[2]
        return plane


# Should now be saved this way already
#
# Subtract a plane from topography so you don't have to keep doing it later
#def correcttopo(series):
#    if series['channel_name'] == 'Z':
#        series['corrscan'] = 1e9 * (series['scan'] - fitplane(series['scan']))
#        series['corrscan2'] = 1e9 * (series['scan2'] - fitplane(series['scan2']))
#        return series
#    else: return series
#df = df.apply(correcttopo, 1)

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

def plot_cafm(data, n=1, scaleaxes=True):
    # Data should be a dataframe with one of type 'I' and one of type 'Z'
    # This makes a plot of topography next to current
    topo_cm = 'viridis'
    current_cm = 'inferno'
    # Pandas doesn't know this should only have one row, so you need to tell it
    # to take the 0th row
    I = data[data['channel_name'] == 'I'].iloc[0]
    Z = data[data['channel_name'] == 'Z'].iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22.5,8))
    # This is not the way the afm program plots it
    #left = 0
    #right = I['width'] * 1e9
    #bottom = 0
    #top = I['height'] * 1e9
    left = - I['width'] * 1e9 / 2
    right = I['width'] * 1e9 / 2
    bottom = -I['height'] * 1e9 / 2
    top = I['height'] * 1e9 / 2
    if n==1:
        Idata = I['scan']
        Zdata = Z['corrscan']
    elif n==2:
        Idata = I['scan2']
        Zdata = Z['corrscan2']
    if scaleaxes:
        extent = (left, right, bottom, top)
    else:
        extent = None
    # Plot topography image
    p1, p99 = np.percentile(Zdata, (0.2, 99.8))
    im1 = ax1.imshow(Zdata, cmap=topo_cm, vmin=p1, vmax=p99, extent=extent)
    ax1.invert_yaxis()
    title = 'Sample: {},  Folder: {},  id: {}'.format(I['sample_name'], I['folder'], I['id'])
    if n == 2:
        # Indicate that it's the reverse scan
        title += '_r'
    ax1.set_title(title)
    ax2.set_title('Tip voltage: {} V'.format(I['voltage']))
    fig.colorbar(im1, ax=ax1, label='Height [nm]')
    # Plot current image
    Idata
    p1, p99 = np.percentile(1e9 * Idata, (0.2, 99.8))
    im2 = ax2.imshow(1e9 * Idata, cmap=current_cm, vmin=p1, vmax=p99, extent=extent)
    if scaleaxes:
        ax2.set_xlabel('X [nm]')
        ax2.set_ylabel('Y [nm]')
        ax1.set_xlabel('X [nm]')
        ax1.set_ylabel('Y [nm]')
    else:
        ax2.set_xlabel('X [pixels]')
        ax2.set_ylabel('Y [pixels]')
        ax1.set_xlabel('X [pixels]')
        ax1.set_ylabel('Y [pixels]')
    ax2.invert_yaxis()
    fig.colorbar(im2, ax=ax2, label='Current [nA]')
    return fig

def grad_scatter(data, n=1):
    fig = plt.figure(figsize=(10,10))
    ax = plt.subplot(polar=True)
    if n == 1:
        I = data[data['channel_name'] == 'I'].iloc[0]['scan'] * 1e9
        Z = data[data['channel_name'] == 'Z'].iloc[0]['corrscan']
    elif n == 2:
        I = data[data['channel_name'] == 'I'].iloc[0]['scan2'] * 1e9
        Z = data[data['channel_name'] == 'Z'].iloc[0]['corrscan2']
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
        h, w = np.shape(I)
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
    cb = fig.colorbar(sc, label='c-afm current [nA]')
    cb.set_alpha(1)
    cb.draw_all()
    return fig, ax


def plot_cafm_hist(data):
    # Data should be a dataframe with one of type 'I' and one of type 'Z'
    # This makes a plot of topography next to current
    # Pandas doesn't know this should only have one row, so you need to tell it
    # to take the 0th row
    I = data[data['channel_name'] == 'I'].iloc[0]
    Z = data[data['channel_name'] == 'Z'].iloc[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(22.5,8))
    Idata = I['scan']
    Zdata = Z['corrscan']
    # Correct the Z data
    # Make height histogram
    p1, p99 = np.percentile(Zdata, (0.2, 99.8))
    # bins = 'auto' sometimes results in an insane number of bins and the program hangs while it fills up your memory completely
    hist1 = ax1.hist(Zdata.flatten(), bins='scott', range=(p1, p99), color='ForestGreen')
    ax1.set_xlabel('Height [nm]')
    ax1.set_ylabel('Pixel Count')
    ax1.set_title('Sample: {},  Folder: {},  id: {}'.format(I['sample_name'], I['folder'], I['id']))
    # Make current histogram
    p1, p99 = np.percentile(Idata, (0.2, 99))
    hist2 = ax2.hist(Idata.flatten(), bins='scott', range=(p1, p99), color='Crimson')
    ax2.set_title('Tip voltage: {} V'.format(I['voltage']))
    ax2.set_xlabel('Current [nA]')
    ax2.set_ylabel('Pixel Count')
    return fig


if __name__ == '__main__':
    # You can pass a set of folders to analyze, or else the script will do them all
    if len(sys.argv) > 1:
        folders = sys.argv[1:]
    else:
        folders = os.listdir(sys.path[0])
        folders = [f for f in folders if os.path.isdir(f)]
        # Don't touch folders that have no data files inside of them ...
        def anydata(folder):
            dirlist = os.listdir(folder)
            for fn in dirlist:
                if fn.endswith('mtrx'): return True
            return False
        folders = [f for f in folders if anydata(f)]

    # Construct single dataframe from all the dataframes in the folders
    # This happens to be less work at the moment
    # Also allows me to write each plot type in its own loop so it can be copy pasted
    dfs = []
    for f in folders:
        df_path = os.path.join(f, f + '.df')
        dfs.append(pd.read_pickle(df_path))
        print('Loaded {} into memory'.format(df_path))
    df = pd.concat(dfs)

    # Make nice subplots of each scan (that has aspect ratio close to 1)
    are_scans = df['type'] == 'xy'
    # Not needed anymore, we don't load data that is not in folders of interest
    #in_folders =  df['folder'].isin(folders)
    for folder, folderdata in df[are_scans].groupby('folder'):
        plotfolder = os.path.join(folder, 'topo_current_subplots')
        if not os.path.isdir(plotfolder):
            os.makedirs(plotfolder)
        for id, data in folderdata.groupby('id'):
            aspect = data.iloc[0]['aspect']
            if 0.5 < aspect < 2:
                fig = plot_cafm(data, n=1)
                fig2 = plot_cafm(data, n=2)
                #fn = os.path.splitext(data.iloc[0]['filename'])[0]
                #Just use the ID as a file name
                fn = id
                savepath = os.path.join(plotfolder, fn)
                fig.savefig(savepath, bbox_inches=0)
                print('Wrote {}.png'.format(savepath))
                fig2.savefig(savepath + '_r')
                print('Wrote {}_r.png'.format(savepath))
                plt.close(fig)
                plt.close(fig2)

    # Scatter height vs current
    for folder, folderdata in df[are_scans].groupby('folder'):
        scatterfolder = os.path.join(folder, 'topo_current_scatter')
        if not os.path.isdir(scatterfolder):
            os.makedirs(scatterfolder)
        for id, data in folderdata.groupby('id'):
            aspect = data.iloc[0]['aspect']
            if 0.6 < aspect < 1.5:
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
                print('Wrote {}.png'.format(savepath))
                plt.close(fig)


    # Not that useful
    '''
    # Forward vs backward scans
    for folder, folderdata in df[are_scans].groupby('folder'):
        topo_dir = os.path.join(folder, 'forward_vs_backward_topo')
        current_dir = os.path.join(folder, 'forward_vs_backward_current')
        if not os.path.isdir(topo_dir):
            os.makedirs(topo_dir)
        if not os.path.isdir(current_dir):
            os.makedirs(current_dir)
        for id, data in folderdata.groupby('id'):
            h, w = np.shape(data.iloc[0]['scan'])
            if w != 0:
                ratio = float(h)/w
            else: ratio = 0
            if 0.8 < ratio < 1.2:
                fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(22.5,8))
                fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(22.5,8))

                topo_cm = 'viridis'
                current_cm = 'inferno'
                # Pandas doesn't know this should only have one row, so you need to tell it
                # to take the 0th row
                I = data[data['channel_name'] == 'I'].iloc[0]
                Z = data[data['channel_name'] == 'Z'].iloc[0]
                left = 0
                right = I['width'] * 1e9
                bottom = 0
                top = I['height'] * 1e9
                Idata1 = I['scan']
                Zdata1 = Z['corrscan']
                Idata2 = I['scan2']
                Zdata2 = Z['corrscan2']
                # Plot topography images
                p1, p99 = np.percentile(Zdata1, (0.2, 99.8))
                im11 = ax11.imshow(Zdata1, cmap=topo_cm, vmin=p1, vmax=p99, extent=(left, right, bottom, top))
                im12 = ax12.imshow(Zdata2, cmap=topo_cm, vmin=p1, vmax=p99, extent=(left, right, bottom, top))
                ax11.set_xlabel('(Forward scan) X [nm]')
                ax11.set_ylabel('Y [nm]')
                ax12.set_xlabel('(Reverse scan) X [nm]')
                ax12.set_ylabel('Y [nm]')
                title = 'Sample: {},  Folder: {},  id: {}'.format(I['sample_name'], I['folder'], I['id'])
                ax11.set_title(title)
                ax12.set_title('Tip voltage: {} V'.format(I['voltage']))
                fig1.colorbar(im11, ax=ax11, label='Height [nm]')
                fig1.colorbar(im12, ax=ax12, label='Height [nm]')

                # Plot current images
                p1, p99 = np.percentile(1e9 * Idata1, (0.2, 99.8))
                im21 = ax21.imshow(1e9 * Idata1, cmap=current_cm, vmin=p1, vmax=p99, extent=(left, right, bottom, top))
                im22 = ax22.imshow(1e9 * Idata1, cmap=current_cm, vmin=p1, vmax=p99, extent=(left, right, bottom, top))
                ax21.set_xlabel('(Forward scan) X [nm]')
                ax21.set_ylabel('Y [nm]')
                ax22.set_xlabel('(Reverse scan) X [nm]')
                ax22.set_ylabel('Y [nm]')
                fig2.colorbar(im21, ax=ax21, label='Current [nA]')
                fig2.colorbar(im22, ax=ax22, label='Current [nA]')

                #Just use the ID as a file name
                fn = id
                savepath1 = os.path.join(topo_dir, fn)
                fig1.savefig(savepath1, bbox_inches=0)
                print('Wrote {}.png'.format(savepath1))
                savepath2 = os.path.join(current_dir, fn)
                fig2.savefig(savepath2)
                print('Wrote {}.png'.format(savepath2))
                plt.close(fig1)
                plt.close(fig2)
    '''

    # hexbin height vs current
    for folder, folderdata in df[are_scans].groupby('folder'):
        scatterfolder = os.path.join(folder, 'topo_current_hexbin')
        if not os.path.isdir(scatterfolder):
            os.makedirs(scatterfolder)
        for id, data in folderdata.groupby('id'):
            aspect = data.iloc[0]['aspect']
            if 0.6 < aspect < 1.5:
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
                print('Wrote {}.png'.format(savepath))
                plt.close(fig)

    # Histograms of current
    # Make nice subplots of each scan (that has aspect ratio close to 1)
    for folder, folderdata in df[are_scans].groupby('folder'):
        plotfolder = os.path.join(folder, 'topo_current_histograms')
        if not os.path.isdir(plotfolder):
            os.makedirs(plotfolder)
        for id, data in folderdata.groupby('id'):
            aspect = data.iloc[0]['aspect']
            if 0.6 < aspect < 1.5:
                fig = plot_cafm_hist(data)
                #fn = os.path.splitext(data.iloc[0]['filename'])[0]
                #Just use the ID as a file name
                fn = id
                savepath = os.path.join(plotfolder, fn)
                fig.savefig(savepath, bbox_inches=0)
                print('Wrote {}.png'.format(savepath))
                plt.close(fig)

    # Polar plot gradient vs current
    # These take a long time for some reason
    for folder, folderdata in df[are_scans].groupby('folder'):
        scatterfolder = os.path.join(folder, 'gradient_current_polarscatter')
        if not os.path.isdir(scatterfolder):
            os.makedirs(scatterfolder)
        for id, data in folderdata.groupby('id'):
            aspect = data.iloc[0]['aspect']
            if 0.6 < aspect < 1.5:
                fig1, ax1 = grad_scatter(data, n=1)
                ax1.set_title('Current vs. Topography gradient: {}, {}'.format(folder, id), y=1.1)
                fig2, ax2 = grad_scatter(data, n=2)
                ax2.set_title('Current vs. Topography gradient: {}, {}_r'.format(folder, id), y=1.1)
                # Save the figure
                fn = id
                savepath = os.path.join(scatterfolder, fn)
                fig1.savefig(savepath, bbox_inches=0)
                print('Wrote {}.png'.format(savepath))
                fig2.savefig(savepath + '_r', bbox_inches=0)
                print('Wrote {}_r.png'.format(savepath))
                plt.close(fig1)
                plt.close(fig2)


    # Make plot of scan locations.  I don't know why, I just thought it would be cool.
    # Might be useful to annotate each region with the measurement id
    # If you want, you can make a plot for every scan that shows only the previous and next scan areas with low opacity
    from matplotlib import patches
    for folder, folderdata in df.groupby('folder'):
        regionsfolder = os.path.join(folder, 'scan_regions')
        if not os.path.isdir(regionsfolder):
            os.makedirs(regionsfolder)
        topo = folderdata[folderdata.channel_name == 'Z']
        fig, ax = plt.subplots(figsize=(8,8))
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
        scan1d = folderdata[folderdata['type'].isin(['iv', 'iz'])]
        x = scan1d['x_offset']
        y = scan1d['y_offset']
        ax.scatter(x, y, alpha=1, edgecolor='none', c='black', s=15, zorder=0)
        # Save to folder
        savepath = os.path.join(regionsfolder, '{}_all_scan_regions.png'.format(folder))
        fig.savefig(savepath, bbox_inches=0)
        print('Wrote {}'.format(savepath))

        # Write spreadsheet of xy scan locations
        savepath = os.path.join(regionsfolder, '{}_all_xy_scan_regions.xls'.format(folder))
        xy_regions = folderdata[folderdata['type'] == 'xy'][['folder', 'filename', 'mystery_id', 'run', 'cycle', 'x_offset', 'y_offset', 'width', 'height']].sort_values(by=['mystery_id', 'run', 'cycle'])
        xy_regions.to_excel(savepath, index=False)
        print('Wrote {}'.format(savepath))

        # Write spreadsheet of iv/iz scan locations
        savepath = os.path.join(regionsfolder, '{}_all_iv_iz_scan_regions.xls'.format(folder))
        xy_regions = folderdata[folderdata['type'].isin(['iv', 'iz'])][['folder', 'filename', 'mystery_id', 'run', 'cycle', 'type', 'x_offset', 'y_offset']].sort_values(by=['mystery_id', 'run', 'cycle'])
        xy_regions.to_excel(savepath, index=False)
        print('Wrote {}'.format(savepath))




