# Convert all the LCAFM data files to other formats
# This script will convert from all the folders in the directory it is contained in
# Run from python2 environment because that's where access2thematrix lives..
import access2thematrix
from scipy.io import savemat
import os
import sys
from numpy import savetxt
import numpy as np
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import pickle
import pandas as pd
import time
import re


alldata = pd.DataFrame()

def getdata(fn, folder=''):
    mtrx_data = access2thematrix.MtrxData()
    fp = os.path.join(folder, fn)
    # traces is a dict specifying what data arrays are in the file
    # they can be different types and different a different number
    traces, message = mtrx_data.open(fp)
    print(message)
    if not message.startswith('Success'):
        return message

    # There is a lot of metadata in these files
    # I am trying to pick out the stuff that I think is useful at the moment
    # There can be two different kinds of data, corresponding to xy scans and
    # parameter sweeps (iv loops). There can be more than one of each kind ...
    # Access2thematrix is not very uniform about how it returns the data.
    dataout = dict()
    if 'trace' in traces.values():
        trace, message = mtrx_data.select_curve('trace')
        datain = trace.__dict__
        xoff, yoff = datain['referenced_by']['Location (m)']
        dataout['x_offset'] = xoff
        dataout['y_offset'] = yoff
        dataout['V'] = datain['data'][0]
        dataout['I'] = datain['data'][1]
        dataout['type'] = 'iv'
        if 'retrace' in traces.values():
            # There is a retrace.  I don't think there is separate metadata
            retrace, message = mtrx_data.select_curve('retrace')
            datain2 = retrace.__dict__
            dataout['V2'] = datain2['data'][0]
            dataout['I2'] = datain2['data'][1]
    else:
        # Must be x-y data.
        im, message = mtrx_data.select_image(traces[0])
        datain = im.__dict__
        dataout['channel_unit'] = datain['channel_name_and_unit'][1]
        #dataout['type'] = os.path.splitext(fn)[1][1:].replace('_mtrx', '')
        dataout['type'] = 'xy'
        dataout['scan'] = datain['data']
        dataout['angle'] = datain['angle']
        dataout['height'] = datain['height']
        dataout['width'] = datain['width']
        dataout['y_offset'] = datain['y_offset']
        dataout['x_offset'] = datain['x_offset']
        dataout['voltage'] = mtrx_data.param['EEPA::GapVoltageControl.Voltage'][0]
        dataout['aspect'] = dataout['width'] / dataout['height']
        if len(traces) > 1:
            # Get the next trace.
            im2, message = mtrx_data.select_image(traces[1])
            dataout['scan2'] = im2.__dict__['data']
            # There might be more traces but I am ignoring them now

    # These are independent of measurement type
    dataout['ctime'] = os.path.getctime(fp)
    dataout['date'] = time.strftime(r'%d.%m.%Y', time.gmtime(dataout['ctime']))
    dataout['sample_name'] = mtrx_data.sample_name
    # Get the measurement ID from out-of-control file name
    # There is a 6 digit number, and two ? digit numbers at the end ...
    match = re.search('-(\d{6})_.*--([0-9]+)_([0-9]+).', fn)
    if match is None:
        mystery_id, run, cycle = '', '', ''
        dataout['id'] = ''
    else:
        mystery_id, run, cycle = match.groups()
        dataout['id'] = '_'.join([mystery_id, run, cycle])
    dataout['mystery_id'] = mystery_id
    dataout['run'] = run
    dataout['cycle'] = cycle
    dataout['channel_name'] = mtrx_data.channel_name
    dataout['filename'] = fn
    dataout['folder'] = folder

    return dataout

def valid_name(name):
    # replace . - and () in name
    name = name.replace('-', '_')
    name = name.replace('(', '_')
    name = name.replace(')', '_')
    name = name.replace('.', '_')
    return name

### Determine folders to go through
#folder = '01-Jun-2017'
folders = os.listdir(sys.path[0])
folders = [f for f in folders if os.path.isdir(f)]
# Don't touch folders that have no data files inside of them ...
def anydata(folder):
    dirlist = os.listdir(folder)
    for fn in dirlist:
        if fn.endswith('mtrx'): return True
    return False
folders = [f for f in folders if anydata(f)]

for folder in folders:
    # Load all the data from the folders
    files = os.listdir(folder)
    files = [f for f in files if f.endswith('mtrx')]
    data = {valid_name(f):getdata(f, folder) for f in files}

    # Remove data that isn't a dict
    # means it didn't load correctly.
    data = {k:v for k,v in data.items() if type(v) == dict}

    # Write to pandas dataframe
    pdfilename = folder + '.df'
    pdfile = os.path.join(folder, pdfilename)
    df = pd.DataFrame(data.values())

    # Write corrected topography to dataframe as well
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

    def correcttopo(series):
        if series['channel_name'] == 'Z':
            series['corrscan'] = 1e9 * (series['scan'] - fitplane(series['scan']))
            series['corrscan2'] = 1e9 * (series['scan2'] - fitplane(series['scan2']))
            return series
        else: return series
    df = df.apply(correcttopo, 1)

    df.to_pickle(pdfile)
    print('Wrote file {}'.format(pdfile))
    # It will be useful to group by the filename without extension
    #df['filename'] = df['filename'].apply(lambda fn: os.path.splitext(fn)[0])
    alldata = alldata.append(df, ignore_index=True)

    # Write to matlab format
    matfilename = folder + '.mat'
    matfile = os.path.join(folder, matfilename)
    savemat(matfile, data)
    print('Wrote file {}'.format(matfile))

    # Write to text files ...
    # Separate one for each file.
    csvfolder = os.path.join(folder, 'csv')
    if not os.path.isdir(csvfolder):
        os.makedirs(csvfolder)
    for d in data.values():
        txtname = d['filename'] + '.txt'
        txtpath = os.path.join(csvfolder, txtname)
        if d['type'] == 'xy':
            savekeys = ['filename', 'channel_name', 'channel_unit', 'angle', 'height', 'width', 'x_offset', 'y_offset', 'voltage']
            header = '\n'.join(['{}: {}'.format(k, d[k]) for k in savekeys])
            # Files have very strange extensions.  Just keep them there.
            savetxt(txtpath, d['scan'], header=header, delimiter='\t')
            # Not writing the second scan unless someone starts screaming
            print('Wrote {}'.format(txtpath))
        elif d['type'] == 'iv':
            savekeys = ['filename', 'x_offset', 'y_offset']
            header = '\n'.join(['{}: {}'.format(k, d[k]) for k in savekeys])
            if 'V2' in d.keys():
                header += '\ncolumns: V1\tI1\tV2\tI2'
                table = np.vstack((d['V'], d['I'], d['V2'], d['I2'])).T
            else:
                header += '\ncolumns: V\tI'
                table = np.vstack((d['V'], d['I'])).T
            savetxt(txtpath, table, header=header, delimiter='\t')
            print('Wrote {}'.format(txtpath))


    # Dump pngs for data files containing a matrix
    # Dump pickles of figs too
    pngfolder = os.path.join(folder, 'raw_png')
    figfolder = os.path.join(folder, 'figs')
    if not os.path.isdir(pngfolder):
        os.makedirs(pngfolder)
    if not os.path.isdir(figfolder):
        os.makedirs(figfolder)
    for d in data.values():
        pngname = d['filename'] + '.png'
        pngpath = os.path.join(pngfolder, pngname)
        if d['type'] == 'xy' and np.shape(d['scan'])[1] > 0:
            # Files have very strange extensions.  Just keep them there.
            figname = d['filename'] + '.plt'
            figpath = os.path.join(figfolder, figname)

            fig, ax = plt.subplots()
            left = 0
            right = d['width'] * 1e9
            bottom = 0
            top = d['height'] * 1e9
            im = ax.imshow(1e9 * d['scan'], cmap='viridis', extent=(left, right, bottom, top))
            ax.set_xlabel('X [nm]')
            ax.set_ylabel('Y [nm]')
            fig.colorbar(im, label='{} [n{}]'.format(d['channel_name'], d['channel_unit']))
            #print('Wrote {}'.format(pngpath))
            #with open(figpath, 'wb') as f:
            #    pickle.dump(fig, f)
            #    print('Wrote {}'.format(figpath))
        elif d['type'] == 'iv':
            # make IV plot
            fig, ax = plt.subplots()
            ax.plot(d['V'], 1e9 * d['I'])
            if 'V2' in d.keys():
                ax.plot(d['V2'], 1e9 * d['I2'])
            ax.set_xlabel('Voltage [V]')
            ax.set_ylabel('Current [nA]')
            x, y = d['x_offset'], d['y_offset']
            ax.set_title('Location: {} nm, {} nm'.format(x * 1e9, y * 1e9))
            ax.legend(['Up', 'Down'], title='Sweep direction', loc=0)
        fig.savefig(pngpath, bbox_inches=0)
        plt.close(fig)

# When done, put dataframe containing all of the data into
alldata.to_pickle('all_lcafm_data.pd')
print('Wrote all_lcafm_data.pd')
