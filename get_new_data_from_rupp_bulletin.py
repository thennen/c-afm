# Get new data from Rupp bulletin.
import sys
import os

def anydata(folder):
    dirlist = os.listdir(folder)
    for fn in dirlist:
        if fn.endswith('mtrx'): return True
    return False

# Identify Rupp folders with data inside of them
rupp_data_folder = r'X:\emrl\Pool\Bulletin\Rupp\Messungen\LCAFM'
rupp_folders = os.listdir(rupp_data_folder)
rupp_paths = [os.path.join(rupp_data_folder, f) for f in rupp_folders]
rupp_folder_paths = [f for f in rupp_paths if os.path.isdir(f)]
rupp_data_folder_paths = [f for f in rupp_folder_paths if anydata(f)]
rupp_data_folder_names = [os.path.split(f)[-1] for f in rupp_data_folder_paths]

# Identify local folders with data inside of them
pwd = sys.path[0]
local_folders = os.listdir(pwd)
local_folders = [f for f in local_folders if os.path.isdir(f)]
local_folders = [f for f in local_folders if anydata(f)]

for fp in rupp_data_folder_paths:
    folder_name = os.path.split(fp)[-1]
    if folder_name not in local_folders:
        print('Copying {}'.format(fp))
        source = fp
        dest = os.path.join(pwd, folder_name)
        robocmd = 'robocopy \"{}\" \"{}\" /S'.format(source, dest)
        #print(robocmd)
        os.system(robocmd)
    else:
        print('Already have {}'.format(folder_name))
