
import shutil
import os
import sys
from glob import glob

def ignore_except(src,contents):
    #print 'sys.argv:', sys.argv[0]
    #all files in 2 dirs layers
    #rootdirs=filter(os.path.isdir,os.path.listdir(sys.argv[0]))
    allowed_dirs=['causal_controller','causal_began','causal_dcgan','figure_scripts']

    files=filter(os.path.isfile,contents)
    dirs=filter(os.path.isdir,contents)

    #Copy All python files within 2 levels
    ignored_files=[f for f in files if not f.endswith('.py')]
    ignored_dirs=[d for d in dirs if not d in allowed_dirs]
    #ignored_dirs=['old','data','figures','began']

    ignored=ignored_files+ignored_dirs

#    all_files1 =glob('*')
#    py_files1  =glob('*.py')
#    all_files2=glob('*/*')
#    py_files2 =glob('*/*.py')
#
#    all_files=set(all_files1).union(all_files2)
#    py_files=set(py_files1).union(py_files2)
#
#    return all_files-py_files

    return ignored

if __name__=='__main__':
    pass


