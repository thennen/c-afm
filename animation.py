# Should use dedicated mayavi environment because of its odd requirements
import pandas as pd
import numpy as np
import os
import sys

df = pd.read_pickle('all_lcafm_data.pd')

def frames_to_mp4(directory, prefix='Loop', outname='out'):
    # Send command to create video with ffmpeg
    #cmd = (r'cd "{}" & ffmpeg -framerate 10 -i {}%03d.png -c:v libx264 '
    #        '-r 15 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
    #        '{}.mp4').format(directory, prefix, outname)
    # Should be higher quality still compatible with outdated media players
    cmd = (r'cd "{}" & ffmpeg -framerate 10 -i {}%03d.png -c:v libx264 '
            '-pix_fmt yuv420p -crf 1 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
            '{}.mp4').format(directory, prefix, outname)
    # Need elite player to see this one
    #cmd = (r'cd "{}" & ffmpeg -framerate 10 -i {}%03d.png -c:v libx264 '
            #' -crf 17 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
            #'{}.mp4').format(directory, prefix, outname)
    os.system(cmd)

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

def rotatingvideo(imseries, column='scan', folder='', color=None, nrotations=180,warpscale=3):
    # Make a sweet ass rotating 3d surface plot with mayavi
    # Pass the pandas series of the scan you want to plot
    from mayavi import mlab
    image = imseries[column]
    if not os.path.isdir(folder):
        os.makedirs(folder)
    fig1 = mlab.figure(bgcolor=(1,1,1), size=(1200, 600))
    # I don't think this makes it any faster
    #fig1.scene.off_screen_rendering = True
    # Probably messing up dimensions again.
    h, w = np.shape(image)
    sh = imseries['height'] * 1e9
    sw = imseries['width'] * 1e9
    x = np.linspace(0, sh, h)
    y = np.linspace(0, sw, w)
    # Use this if you just want the surface color to represent the height
    #mlab.surf(x, y, image, warp_scale=warpscale)
    if color is None:
        color = image
    p1, p99 = np.percentile(color.flatten(), (0.1, 99.9))
    # mesh allows coloring by a different array than the height
    y, x = np.meshgrid(y, x)
    print(np.shape(x))
    print(np.shape(y))
    print(np.shape(image))
    print(np.shape(color))
    mlab.mesh(x, y, warpscale*image, scalars=color, colormap='blue-red', vmin=p1, vmax=p99)
    mlab.view(elevation=70, distance='auto')
    for i in range(nrotations):
        fig1.scene.camera.azimuth(360. / nrotations)
        fp = os.path.join(folder, 'anim{:03d}.png'.format(i))
        fig1.scene.save_png(fp)
    mlab.close()


if __name__ == '__main__':
    '''
    # Detailed movie
    #folder = '31-May-2017'
    #id = '23_1'
    #id = '21_1'
    #id = '5_1'
    #id = '16_1'
    folder = '01-Jun-2017'
    #id = '20_1'
    #id = '14_1'
    id = '21_1'
    interesting = df[(df.id == id) & (df.folder == folder)]
    interestingZ = interesting.iloc[0]
    interestingI = interesting.iloc[1]
    Idata = interestingI['scan']
    Zdata = interestingZ['scan'] # / 2 + interestingZ['scan2'] / 2
    sourcefolder = interesting['folder'].iloc[0]
    animdir = os.path.join(folder, 'animations', id)
    #rotatingvideo(interestingZ, column='corrscan', folder=animdir, color=Idata, warpscale=3)
    #frames_to_mp4(animdir, 'anim')
    '''

    # Make a few frames at warpscale 1 for all ~square topography measurements
    # You can pass a folder to analyze, or else the script will do them all
    if len(sys.argv) > 1:
        folders = sys.argv[1:]
    else:
        folders = df['folder'].unique()
    print('Making movies for folders:')
    print('\n'.join(folders))
    are_scans = df['type'] == 'xy'
    in_folders =  df['folder'].isin(folders)
    for folder, folderdata in df[are_scans & in_folders].groupby('folder'):
        for id, data in folderdata.groupby('id'):
            print('Aspect ratio: {}'.format(data.aspect.iloc[0]))
            if 0.8 < data.aspect.iloc[0] < 1.2:
                Zseries = data[data['channel_name'] == 'Z'].iloc[0]
                Idata = data[data['channel_name'] == 'I'].iloc[0]['scan']
                animdir = os.path.join(folder, 'animations', '{}_warpscale_1'.format(id))
                rotatingvideo(Zseries, column='corrscan', nrotations=27, folder=animdir, color=Idata, warpscale=1)
                frames_to_mp4(animdir, 'anim')
