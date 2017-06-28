#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 12:29:16 2017

@author: mkocaoglu
"""

import os
from causal_graph import get_causal_graph
#graph = 'big_causal_graph'
graph_name = 'male_smiling_lipstick_complete'
graph = get_causal_graph(graph_name)
name_list = [i[0] for i in graph]
model_ID = 50
#path = '/Users/mkocaoglu/OneDrive/CausalGAN_model_config'
path = '.'

os.chdir(path)
for name in name_list:
  mystr =  "python visualize.py --model_type dcgan --sample_model True --cross_tab True --do_dict_name %s --dataset celebA --input_height 108 --is_train False --is_crop True --graph %s --checkpoint_dir ./checkpoint/%d --noCC True"% (name, graph_name, model_ID)
  #mystr =  "convert -delay %d $(for i in $(seq 500 500 %d); do if [ $i -eq %d ]; then (for j in $(seq 1 1 5); do echo test_arange_%s%d.png; done); else echo test_arange_%s${i}.png;fi done) -loop 0 %s.gif"% (delay,upper,upper,name,upper,name,name)
  os.system(mystr)
mystr = 'python visualize.py --model_type dcgan --sample_model True --cross_tab True --do_dict_name fixed_label_diversity --dataset celebA --input_height 108 --is_train False --is_crop True --graph %s --checkpoint_dir ./checkpoint/%d --noCC True'%(graph_name, model_ID)
os.system(mystr)
mystr = 'python visualize.py --model_type dcgan --sample_model True --cross_tab True --do_dict_name interpolation --dataset celebA --input_height 108 --is_train False --is_crop True --graph %s --checkpoint_dir ./checkpoint/%d --noCC True'%(graph_name, model_ID)
os.system(mystr)
mystr = 'python visualize.py --model_type dcgan --sample_model True --cross_tab True --do_dict_name interpolation_in_z --dataset celebA --input_height 108 --is_train False --is_crop True --graph %s --checkpoint_dir ./checkpoint/%d --noCC True'%(graph_name, model_ID)
os.system(mystr)
print "done!"



# with CC
# python visualize.py --model_type dcgan --sample_model True --cross_tab False --do_dict_name Male --dataset celebA --input_height 108 --is_train False --is_crop True --graph male_smiling_lipstick_complete --checkpoint_dir ./checkpoint/50 --noCC False
