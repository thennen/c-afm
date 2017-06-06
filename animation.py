import pandas as pd
import numpy as np
import os

df = pd.read_pickle('all_lcafm_data.pd')

def frames_to_mp4(directory, prefix='Loop', outname='out'):
    # Send command to create video with ffmpeg
    cmd = (r'cd "{}" & ffmpeg -framerate 10 -i {}%03d.png -c:v libx264 '
            '-r 15 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
            '{}.mp4').format(directory, prefix, outname)
    # Try this one ...
    # Should be higher quality still compatible with outdated media players
    cmd = (r'cd "{}" & ffmpeg -framerate 10 -i {}%03d.png -c:v libx264 '
            '-pix_fmt yuv420p -crf 1 -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" '
            '{}.mp4').format(directory, prefix, outname)
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

def rotatingvideo(image, folder, color, warpscale=3):
    # Make a sweet ass rotating 3d surface plot with mayavi
    from mayavi import mlab
    if not os.path.isdir(folder):
        os.makedirs(folder)
    fig1 = mlab.figure(bgcolor=(1,1,1), size=(800, 400))
    # I don't think this makes it any faster
    #fig1.scene.off_screen_rendering = True
    # Probably messing up dimensions again.
    h, w = np.shape(image)
    # Hard coding in the dimensions so that I can regret it later
    x = np.linspace(0, 1000, h)
    y = np.linspace(0, 1000, w)
    # Test line
    x, y = np.meshgrid(x, y)
    #mlab.surf(x, y, image, warp_scale=warpscale)
    p1, p99 = np.percentile(color.flatten(), (0.1, 99.9))
    mlab.mesh(x, y, warpscale*image, scalars=color, colormap='blue-red', vmin=p1, vmax=p99)
    mlab.view(elevation=70, distance='auto')
    nrotations = 270
    for i in range(nrotations):
        fig1.scene.camera.azimuth(360. / nrotations)
        fp = os.path.join(folder, 'anim{:03d}.png'.format(i))
        fig1.scene.save_png(fp)
    frames_to_mp4(folder, 'anim')


if __name__ == '__main__':
    id = '23_1'
    id = '16_1'
    folder = '31-May-2017'
    #id = '20_1'
    #folder = '01-Jun-2017'
    interesting = df[(df.id == id) & (df.folder == folder)]
    interestingZ = interesting.iloc[0]
    interestingI = interesting.iloc[1]
    Idata = interestingI['scan']
    Zdata = interestingZ['scan'] # / 2 + interestingZ['scan2'] / 2
    corrzdata = 1e9 * (Zdata - fitplane(Zdata))
    sourcefolder = interesting['folder'].iloc[0]
    animdir = os.path.join(folder, 'animations', id)
    rotatingvideo(corrzdata, animdir, color=Idata, warpscale=3)
