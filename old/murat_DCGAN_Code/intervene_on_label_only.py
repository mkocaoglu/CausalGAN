#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 18:17:21 2017

@author: mkocaoglu
"""

import os
from causal_graph import get_causal_graph
def intervene_on_current_model():
  graph = 'big_causal_graph'
  graph = get_causal_graph(graph)
  name_list = [i[0] for i in graph]
  model_ID = 35
  path = '/Users/mkocaoglu/OneDrive/CausalGAN_model_config'
  
  os.chdir(path)
  for name in name_list:
    mystr =  "python visualize.py --model_type dcgan --sample_model True --cross_tab True --do_dict_name %s --dataset celebA --input_height 108 --is_train False --is_crop True --graph big_causal_graph --checkpoint_dir ./checkpoint/%d --noCC True"% (name, model_ID)
    #mystr =  "convert -delay %d $(for i in $(seq 500 500 %d); do if [ $i -eq %d ]; then (for j in $(seq 1 1 5); do echo test_arange_%s%d.png; done); else echo test_arange_%s${i}.png;fi done) -loop 0 %s.gif"% (delay,upper,upper,name,upper,name,name)
    os.system(mystr)