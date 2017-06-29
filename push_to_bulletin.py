# Push analysis to bulletin, but not all the data files, overwriting only when necessary
import sys
import os

pwd = sys.path[0]
folders = os.listdir(pwd)
folders = [f for f in folders if os.path.isdir(f)]
# Don't touch folders that have no data files inside of them ...
def anydata(folder):
    dirlist = os.listdir(folder)
    for fn in dirlist:
        if fn.endswith('mtrx'): return True
    return False
folders = [f for f in folders if anydata(f)]

for f in folders:
    source = os.path.join(pwd, f)
    dest = os.path.join('X:\emrl\Pool\Bulletin\Hennen\LCAFM', f)
    robocmd = 'robocopy \"{}\" \"{}\" /S /MIR /XF *_mtrx *.mtrx'.format(source, dest)
    #print(robocmd)
    os.system(robocmd)
