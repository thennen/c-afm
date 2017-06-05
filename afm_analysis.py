import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# Just plopping down some code fragments

df = pd.read_pickle('all_lcafm_data.pd')
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
    fig.colorbar(im1, ax=ax1, label='Height [nm]')
    # Plot current image
    Idata
    p1, p99 = np.percentile(1e9 * Idata, (0.2, 99.8))
    im2 = ax2.imshow(1e9 * Idata, cmap=current_cm, vmin=p1, vmax=p99, extent=(left, right, bottom, top))
    ax2.set_xlabel('X [nm]')
    ax2.set_ylabel('Y [nm]')
    fig.colorbar(im2, ax=ax2, label='Current [nA]')
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

def frames_to_mp4(directory, prefix='Loop', outname='out'):
    # Send command to create video with ffmpeg
    cmd = (r'cd "{}" & ffmpeg -framerate 10 -i {}%03d.png -c:v libx264 '
            '-r 15 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
            '{}.mp4; pause').format(directory, prefix, outname)
    os.system(cmd)

interesting = df[(df.id == '23_1') & (df.folder == '31-May-2017')]
interestingZ = interesting.iloc[0]
Zdata = interestingZ['scan']
corrzdata = 1e9 * (Zdata - fitplane(Zdata))
def rotatingvideo(image):
    # Make a sweet ass rotating 3d surface plot with mayavi
    from mayavi import mlab
    fig1 = mlab.figure(bgcolor=(1,1,1), size=(800, 400))
    # I don't think this makes it any faster
    #fig1.scene.off_screen_rendering = True
    # Probably messing up dimensions again.
    h, w = np.shape(image)
    x = np.arange(h)
    y = np.arange(w)
    mlab.surf(x, y, image, warp_scale=1)
    mlab.view(elevation=80, distance='auto')
    for i in range(360):
        fig1.scene.camera.azimuth(1)
        fig1.scene.save_png('anim/anim{:03d}.png'.format(i))
    frames_to_mp4('anim', 'anim')



if __name__ == '__main__':
    # Make nice subplots of each scan (that has aspect ratio close to 1)
    for folder, folderdata in df.groupby('folder'):
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

    # Make plot of scan locations.  I don't know why, I just thought it would be cool.
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

