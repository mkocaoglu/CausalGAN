import os
import sys

from subprocess import call

def file2number(fname):
    nums=[s for s in fname.split('_') if s.isdigit()]
    if len(nums)==0:
        nums=['0']
    number=int(''.join(nums))
    return number

if __name__=='__main__':
    root='./logs'

    logs=os.listdir(root)
    logs.sort(key=lambda x:file2number(x))


    logdir=os.path.join(root,logs[-1])
    print 'running tensorboard on logdir:',logdir

    call(['tensorboard', '--logdir',logdir])

