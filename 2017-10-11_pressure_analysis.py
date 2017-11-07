# Jonathan wants me to average some pixels
# He gave me a newly made directory of png files spit out of my program to identify the data, instead of the data files themselves.
# I will have to identify the data files like this ..

import os
from matplotlib.widgets import RectangleSelector
import pandas as pd
from afm_analysis import *

psplit = os.path.split
pjoin = os.path.join

outputdir = '2017-10-11_Pressure_Analysis'
if not os.path.isdir(outputdir):
    os.makedirs(outputdir)

# I selected the relevant data from the original data files in the past, but the code to do it again is still here
dataframefile = pjoin(outputdir, 'Scan_Data.pd')
if os.path.isfile(dataframefile):
    df = pd.read_pickle(dataframefile)
else:
    # Need to generate the data
    '''
    interest = []
    for root, dirs, files in os.walk(r'X:\emrl\Pool\Bulletin\Rupp\AFM Tyler Height Evaluation'):
        for f in files:
            if f.endswith('.png'):
                interest.append(os.path.join(root, f))
    '''
    interest = ['X:\\emrl\\Pool\\Bulletin\\Rupp\\AFM Tyler Height Evaluation\\2017 06 27\\105212_15_1.png',
                'X:\\emrl\\Pool\\Bulletin\\Rupp\\AFM Tyler Height Evaluation\\2017 06 27\\105212_21_1_r.png',
                'X:\\emrl\\Pool\\Bulletin\\Rupp\\AFM Tyler Height Evaluation\\2017 06 27\\105212_28_1.png',
                'X:\\emrl\\Pool\\Bulletin\\Rupp\\AFM Tyler Height Evaluation\\2017 06 27\\145141_7_1.png',
                'X:\\emrl\\Pool\\Bulletin\\Rupp\\AFM Tyler Height Evaluation\\2017 07 04\\135243_7_1.png',
                'X:\\emrl\\Pool\\Bulletin\\Rupp\\AFM Tyler Height Evaluation\\2017 07 11\\101901_10_1.png',
                'X:\\emrl\\Pool\\Bulletin\\Rupp\\AFM Tyler Height Evaluation\\2017 07 11\\101901_20_1.png',
                'X:\\emrl\\Pool\\Bulletin\\Rupp\\AFM Tyler Height Evaluation\\2017 07 11\\101901_25_1.png',
                'X:\\emrl\\Pool\\Bulletin\\Rupp\\AFM Tyler Height Evaluation\\2017 07 11\\134017_13_1.png',
                'X:\\emrl\\Pool\\Bulletin\\Rupp\\AFM Tyler Height Evaluation\\2017 07 11\\134017_7_2.png',
                'X:\\emrl\\Pool\\Bulletin\\Rupp\\AFM Tyler Height Evaluation\\2017 07 18\\094054_10_1.png',
                'X:\\emrl\\Pool\\Bulletin\\Rupp\\AFM Tyler Height Evaluation\\2017 08 02\\143931_16_1.png',
                'X:\\emrl\\Pool\\Bulletin\\Rupp\\AFM Tyler Height Evaluation\\2017 08 02\\143931_18_1.png',
                'X:\\emrl\\Pool\\Bulletin\\Rupp\\AFM Tyler Height Evaluation\\2017 08 02\\143931_25_1.png',
                'X:\\emrl\\Pool\\Bulletin\\Rupp\\AFM Tyler Height Evaluation\\2017 08 02\\143931_7_1.png']

    ids = [psplit(fn)[-1][:-4].strip('_r') for fn in interest]

    # Even the date string is a completely different format. Make old format from new format.
    # Christ ...
    dates = [psplit(psplit(fp)[0])[-1].split(' ') for fp in interest]
    monthnames = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    datestrings = ['-'.join((d[2], monthnames[int(d[1])], d[0])) for d in dates]

    # Or could just find the files in the original data folders by brute force...
    '''
    pngfns = [psplit(fp)[-1] for fp in interest]
    pngorigin = []
    for pngfn in pngfns:
        for root, dirs, files in os.walk('C:\\t\\LCAFM'):
            for f in files:
                if f == pngfn:
                    pngorigin.append(pjoin(root, f))
    '''

    # Load all the dataframes containing the images of interest
    datafolders = np.unique(datestrings)
    dframes = {}
    for datafolder in datafolders:
        datafile = pjoin(r'C:\t\LCAFM', datafolder, datafolder + '.df')
        dframes[datafolder] = pd.read_pickle(datafile)

    # Make a new dataframe containing only the images of interest
    dflist = []
    for id, datestring in zip(ids, datestrings):
        datedf = dframes[datestring]
        dflist.append(datedf[datedf.id == id][datedf.type == 'xy'])
    df = pd.concat(dflist)

    df.to_pickle(dataframefile)

df = df[df.type == 'xy'].dropna(how='all')

df['scan'] = df['scan'].apply(lambda s: s[5:-5])
df['scan2'] = df['scan2'].apply(lambda s: s[5:-5])

df['corrscan'] = df['scan'].apply(lambda s: 1e9 * (s - fitplane(s)))
df['corrscan2'] = df['scan2'].apply(lambda s: 1e9 * (s - fitplane(s)))


summaryfile = pjoin(outputdir, 'Region_stats.csv')
with open(summaryfile, 'w') as f:
    # Write a header
    # Data gets written afterward
    f.write('id,region,imin,imax,jmin,jmax,topo_mean,topo_max,topo_min,topo_std,current_mean,current_min,current_max,current_std\n')

# Group by measurement of interest
for k,g in df.groupby('id'):
    breakloop = False
    plot_cafm(g, scaleaxes=False)
    fig = gcf()
    topoax, currentax, _, _ = fig.get_axes()
    currentdata = g[g['channel_name'] == 'I'].iloc[0].scan
    topodata = g[g['channel_name'] == 'Z'].iloc[0].corrscan
    height_pix, width_pix = shape(topodata)
    width_nm = g.iloc[0].width
    height_nm = g.iloc[0].height

    # pixels were converted to nm, this converts back
    def convert_nm_to_pix(xnm, ynm):
        xpix = width_pix / 2 + xnm * width_pix / width_nm / 1e9
        ypix = height_pix / 2 - ynm * height_pix / height_nm / 1e9
        return (int(ypix), int(xpix))

    vertices = []
    slices = []
    n = 0
    def onselect(eclick, erelease):
        global n
        x0, y0 = int(eclick.xdata), int(eclick.ydata)
        x1, y1 = int(erelease.xdata), int(erelease.ydata)
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        xmin, xmax = min(x0, x1), max(x0, x1)
        ymin, ymax = min(y0, y1), max(y0, y1)
        # Arrays are indexed the opposite way
        # And there is a scaling factor and an offset ....
        # And the y axis is inverted...
        #imin, jmin = convert_nm_to_pix(xmin, ymin)
        #imax, jmax = convert_nm_to_pix(xmax, ymax)
        # I am now plotting directly in pixels to avoid a lot of headache
        imin, imax = int(ymin), int(ymax)
        jmin, jmax = int(xmin), int(xmax)
        vertices.append((imin, imax, jmin, jmax))
        slices.append(np.s_[imin:imax, jmin:jmax])
        # if imshow has no "extent" specified:
        #slices.append(np.s_[ymin:ymax, xmin:xmax])
        xmid = (x0 + x1) / 2
        ymid = (y0 + y1) / 2
        ax = gca()
        # Draw rectangles and put some information there
        bbox={'facecolor':'black', 'alpha':.5}
        currentax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, color='white'))
        meancurrent = np.mean(currentdata[slices[-1]]) * 1e9
        currentax.text(xmid, ymid, '{}\n{:.4f}'.format(n, meancurrent), color='white', bbox=bbox, horizontalalignment='center', verticalalignment='center')
        topoax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, color='white'))
        meantopo = np.mean(topodata[slices[-1]])
        topoax.text(xmid, ymid, '{}\n{:.4f}'.format(n, meantopo), color='white', bbox=bbox, horizontalalignment='center', verticalalignment='center')
        n += 1

    def selector(event):
        if event.key in ['N', 'n'] and selector.RS1.active:
            print('Next measurement ...')
            selector.RS1.set_active(False)
            selector.RS2.set_active(False)
        if event.key in ['Q', 'q']:
            breakloop = True
            selector.RS1.set_active(False)
            selector.RS2.set_active(False)

    selector.RS1 = RectangleSelector(topoax, onselect)
    selector.RS2 = RectangleSelector(currentax, onselect)
    connect('key_press_event', selector)
    while selector.RS1.active:
        plt.pause(.5)

    if breakloop:
        # For some reason break doesn't break here
        break

    # Write files
    id = g.id.iloc[0]
    fig.savefig(pjoin(outputdir, id + '.png'))
    # Rectangle vertices in pixels and in nm
    #savetxt(pjoin(outputdir, id + '_vertices_nm.csv'))

    # Write information about each region
    with open(summaryfile, 'a') as f:
        for region, (slice, vert)in enumerate(zip(slices, vertices)):
            imin, imax, jmin, jmax = vert
            toposlice = topodata[slice]
            currentslice = currentdata[slice]
            flattopo = toposlice.flatten()
            flatcurrent = currentslice.flatten()
            meantopo = np.mean(flattopo)
            meancurrent = np.mean(flatcurrent)
            maxtopo = np.max(flattopo)
            maxcurrent = np.max(flatcurrent)
            mintopo = np.min(flattopo)
            mincurrent = np.min(flatcurrent)
            stdtopo = np.std(flattopo)
            stdcurrent = np.std(flatcurrent)
            paramlist = [id, region, imin, imax, jmin, jmax, meantopo, maxtopo, mintopo, stdtopo, meancurrent, mincurrent, maxcurrent, stdcurrent]
            paramstring = ','.join([format(thing) for thing in paramlist]) + '\n'
            f.write(paramstring)

            topopath = pjoin(outputdir, id + '_topo_region_{:02}.csv'.format(region))
            currentpath = pjoin(outputdir, id + '_current_region_{:02}.csv'.format(region))
            np.savetxt(topopath, toposlice, delimiter=',')
            np.savetxt(currentpath, currentslice, delimiter=',')

# for troubleshooting when onselect has as error and RectangleSelector does not show it
'''
class dummyclick(object):
    def __init__(self, x, y):
        self.xdata = x
        self.ydata = y
onselect(dummyclick(0, 0), dummyclick(1,1))
'''
