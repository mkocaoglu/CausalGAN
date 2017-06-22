# model_config.py
# To simplify command lines used to call models
# Provides the parameter selection for the model to be run

def get_config(FLAGS,model_ID):
  FLAGS.checkpoint_dir = "./checkpoint/" + str(model_ID)
  if model_ID == 1:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = True
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "iid_uniform"
    FLAGS.gamma_k = 0.5  #inactive
    FLAGS.gamma_m = 10.0 #inactive
    FLAGS.gamma_l = 20.0 #inactive
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    return FLAGS

  elif model_ID == 2:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = True
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "iid_uniform"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 5.0
    FLAGS.gamma_l = 20.0
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    return FLAGS

  elif model_ID == 3:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = True
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "iid_uniform"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 10.0
    FLAGS.gamma_l = 5.0
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    return FLAGS

  elif model_ID == 4:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "iid_uniform"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 5.0
    FLAGS.gamma_l = 5.0
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    return FLAGS

# ABOVE are all mode-collapsed, most likely due to pretrained LabelerR, remove pretraining and try again with following:

  elif model_ID == 5:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "iid_uniform"
    FLAGS.gamma_k = 0.5  #inactive
    FLAGS.gamma_m = 10.0 #inactive
    FLAGS.gamma_l = 20.0 #inactive
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    return FLAGS

  elif model_ID == 6:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "iid_uniform"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 2.0 # made more extreme as 3-4 wasnt different at all
    FLAGS.gamma_l = 20.0
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    return FLAGS

  elif model_ID == 7:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "iid_uniform"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 10.0
    FLAGS.gamma_l = 5.0
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    return FLAGS

  elif model_ID == 8:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "iid_uniform"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 2.0 # made more extreme as 3-4 wasnt different at all
    FLAGS.gamma_l = 5.0
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    return FLAGS

  # Next, use realistic distribution

  elif model_ID == 9:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.5  #inactive
    FLAGS.gamma_m = 8.0 #inactive
    FLAGS.gamma_l = 100.0 # made more extreme as it was becoming active at some point for 20
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    return FLAGS

  elif model_ID == 10:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 4.0
    FLAGS.gamma_l = 100.0 # made more extreme as it was becoming active at some point for 20
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    return FLAGS

  elif model_ID == 11:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 8.0
    FLAGS.gamma_l = 4.0
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    return FLAGS

  elif model_ID == 12:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 4.0 # made more extreme as 3-4 wasnt different at all
    FLAGS.gamma_l = 4.0
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    return FLAGS

# addition of label_type = 'continuous' or 'discrete' to Causal_model.py
  elif model_ID == 13:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.5  #inactive
    FLAGS.gamma_m = 8.0 #inactive
    FLAGS.gamma_l = 100.0 # made more extreme as it was becoming active at some point for 20
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

  elif model_ID == 14:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 4.0
    FLAGS.gamma_l = 100.0 # made more extreme as it was becoming active at some point for 20
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

  elif model_ID == 15:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 8.0
    FLAGS.gamma_l = 3.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

  elif model_ID == 16:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 4.0
    FLAGS.gamma_l = 3.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS
  # DO: Add independence network to discriminator. use l_t to determine the coefficient of independence loss.
  elif model_ID == 17:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.5  #inactive
    FLAGS.gamma_m = 20.0 #inactive
    FLAGS.gamma_l = 10.0 # made more extreme as it was becoming active at some point for 20
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

  elif model_ID == 18:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 4.0
    FLAGS.gamma_l = 10.0 # made more extreme as it was becoming active at some point for 20
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

  elif model_ID == 19:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 8.0
    FLAGS.gamma_l = 2.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

  elif model_ID == 20:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 4.0 
    FLAGS.gamma_l = 2.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

# Maybe remove all the margins and run the plain version once more, with independence loss
  elif model_ID == 21:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.5  #inactive
    FLAGS.gamma_m = 20.0 #inactive
    FLAGS.gamma_l = 0.0001 # made always active to assure it is 1 always: independence enforced
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS
  # also 1 disc 2 gen seems to be hurting a lot! it was working better before, which seems to be the only difference at this point except for the bugs!

  elif model_ID == 22:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 4.0 
    FLAGS.gamma_l = 0.0001 # made more extreme as it was becoming active at some point for 20
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

  elif model_ID == 23:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 20.0
    FLAGS.gamma_l = 20.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

  elif model_ID == 24:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.5
    FLAGS.gamma_m = 4.0 
    FLAGS.gamma_l = 20.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

# The previous values didn't work for multiple G updates. 
  elif model_ID == 25:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 1.0  #active
    FLAGS.gamma_m = 20.0 #inactive
    FLAGS.gamma_l = 0.0001 
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS
  # also 1 disc 2 gen seems to be hurting a lot! it was working better before, which seems to be the only difference at this point except for the bugs!

  elif model_ID == 26:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 1.0
    FLAGS.gamma_m = 1.0  # hopefully active now
    FLAGS.gamma_l = 0.0001 
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

  elif model_ID == 27:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 1.0
    FLAGS.gamma_m = 20.0
    FLAGS.gamma_l = 20.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

  elif model_ID == 28:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 1.0
    FLAGS.gamma_m = 1.0 
    FLAGS.gamma_l = 20.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

# The best, the most promising so far is model number 27: 
# No independence enforcing, label loss included, 1 disc 6 gen updates: m_t = 1, l_t = 0, k_t = 1
# I cannot decide on the effect of k_t yet

  elif model_ID == 29:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.8
    FLAGS.gamma_m = 20.0
    FLAGS.gamma_l = 20.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

  elif model_ID == 30:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 1.2
    FLAGS.gamma_m = 20.0
    FLAGS.gamma_l = 20.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

  elif model_ID == 31:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 1.5
    FLAGS.gamma_m = 20.0
    FLAGS.gamma_l = 20.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

# 29 provides an incrementing labelerG_loss, and it seems to have reflected into image quality also. gamma_k=0.8 is a nice value
# Next model, 32, is the same as 29, but makes less often labeler updates. The hope is that it will provide more time to generator to catch up with the labelers and maybe move down labelerR loss while still moving up labelerG loss.
  elif model_ID == 32:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.8
    FLAGS.gamma_m = 20.0
    FLAGS.gamma_l = 20.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

# Even for 32 labelerR is going up together with labelerG. Maybe I should control the coefficient of labelerR loss based on the difference between LabelerR and LabelerG losses, rather than the one i use currently
  elif model_ID == 33:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.8
    FLAGS.gamma_m = 20.0
    FLAGS.gamma_l = 20.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

# 33 was unsuccessful because k_t needed more room than just 0-1 to adjust.
# Try removing coefficient of labelerR loss and incrementing gamma_k slowly to pull apart the two loss terms LabelerR and LabelerG
  elif model_ID == 34:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.8
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = 20.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

# reducing gamma_k may make more sense. after a certain point, maximizing labelerG loss voids label capturing. because labelerG is close to labelerR then. reeducing gamma_k will reduce and get rid of the coefficient of LabelerG
  elif model_ID == 35:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.8
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = 20.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

# added intervention code into training, will save intervention images
# also made gamma_k more slowly changing
  elif model_ID == 36:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 1.0
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = 20.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS
# reverted back these changes, compared to 35, 37 only has interventinal code applied to it.
  elif model_ID == 37:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.8
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = 20.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS
# changed LabelerG loss to l2 loss, removed k_t update, use GANloss+LabelerR-LabelerG losses
  elif model_ID == 38:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = 0.8
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = 20.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

# following keeps k_t same at 1.0 until 3k iters, then sets it to 0
  elif model_ID == 39:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = -1.0
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = 20.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS
# following is where k_t is exponentially decaying
  elif model_ID == 40:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = -1.0
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = 20.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

# ADDING d_loss_on_z: a discriminator that estimates z and also corresponding loss is added to gen loss.
  elif model_ID == 41:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = -1.0
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = -1.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

# 41 was bugged: last layer of z estimator was sigmoid rather than tanh. 42 will have that fix. 
# 43 will include self.rec_loss_coeff=1 also
  elif model_ID == 42:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = -1.0
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = -1.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

  elif model_ID == 43:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = -1.0
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = -1.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

# following use features fed to the minibatch at the input of z estimator for both real and generated images
# 44 for self.rec_loss_coeff=0
# 45 for self.rec_loss_coeff=1
  elif model_ID == 44:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = -1.0
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = -1.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

  elif model_ID == 45:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = -1.0
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = -1.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

# use model 44 with a different causal graph, to check what happens with label conditioned mode collapse with less number of labels
  elif model_ID == 46:
    FLAGS.is_train = True
    FLAGS.graph = "male_smiling_lipstick_complete"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = -1.0
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = -1.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

# try different causal graphs

  elif model_ID == 47:
    FLAGS.is_train = True
    FLAGS.graph = "standard_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = -1.0
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = -1.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

  elif model_ID == 48:
    FLAGS.is_train = True
    FLAGS.graph = "all_nodes"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = -1.0
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = -1.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

###### THE FOLLOWING MODELS ARE WITH CC ADDED 
  elif model_ID == 49:
    FLAGS.is_train = True
    FLAGS.graph = "big_causal_graph"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = -1.0
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = -1.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

  elif model_ID == 50:
    FLAGS.is_train = True
    FLAGS.graph = "male_smiling_lipstick_complete"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = -1.0
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = -1.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS

# following weights g_on_z loss more and scaling with k_t also
  elif model_ID == 51:
    FLAGS.is_train = True
    FLAGS.graph = "male_smiling_lipstick_complete"
    FLAGS.loss_function = 1
    FLAGS.pretrain_LabelerR = False
    FLAGS.pretrain_LabelerR_no_of_epochs = 3
    FLAGS.fakeLabels_distribution = "real_joint"
    FLAGS.gamma_k = -1.0
    FLAGS.gamma_m = -1.0 # set to 1/gamma_k in the code
    FLAGS.gamma_l = -1.0 # made more extreme
    FLAGS.lambda_k = 0.05
    FLAGS.lambda_m = 0.05
    FLAGS.lambda_l = 0.001
    FLAGS.label_type = 'continuous'
    return FLAGS
