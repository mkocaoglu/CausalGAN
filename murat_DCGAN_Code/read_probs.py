#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun May 21 23:04:37 2017

@author: mkocaoglu
"""

# read already saved crosstab files during training

#name_end = '_glabel_crosstab'
#import csv
import matplotlib.pyplot as plt
import numpy as np

#path = '/Users/mkocaoglu/OneDrive/CausalGAN/checkpoint/mustache_causes_male_pretrainer/celebA_64_64_64/'
#path = '/Users/mkocaoglu/OneDrive/CausalGAN/checkpoint/male_causes_mustache_pretrainer/celebA_64_64_64/'
#path = '/Users/mkocaoglu/OneDrive/CausalGAN/checkpoint/big_causal_graph_pretrainer_faster/celebA_64_64_64/'
path = '/Users/mkocaoglu/OneDrive/CausalGAN/checkpoint/big_causal_graph_pretrainer_layerwise/celebA_64_64_64/'
my_dict = {}
file_no_range = range(1,91)
for k in file_no_range:
  if k == 1:
    f = open(path+str(1000*k) +'_glabel_crosstab.txt')
    flag = 0
    pairwise_flag = 0
    for line in f:
      line = line[0:len(line)-1]
      #print line[0:5]
      print line[0:7]
      if line[0:5] == 'Node:':
        name = line[6::]
        print name
        flag = 1
      elif line[0:7] == '  mean=' and flag == 1:
        my_dict[name]=[float(line[8::])]
        flag =0
#==============================================================================
#       elif line[0:9] == 'Pairwise:':
#         pairwise_flag =1
#       elif pairwise_flag:
#==============================================================================
        
    print my_dict
  else:
      f = open(path+str(1000*k) +'_glabel_crosstab.txt')
      flag = 0
      for line in f:
        line = line[0:len(line)-1]
        #print line[0:5]
        print line[0:7]
        if line[0:5] == 'Node:':
          name = line[6::]
          print name
          flag = 1
        elif line[0:7] == '  mean=' and flag == 1:
          my_dict[name].append(float(line[8::]))
          flag =0      
x = file_no_range
counter = 0
for i in my_dict.keys():
  y = np.array(my_dict[i])
  plt.figure(counter)
  plt.plot(x,y, label = i)
  plt.legend(loc='upper left')
  #plt.show()
  plt.savefig(path+i+'.pdf')
  counter = counter + 1
#plt.legend(loc='upper left')
#plt.ylim(0,3.0)
#plt.show()
      
#line#f.read()
#for i in range(1,32):