import numpy as np
import pandas as pd
means = pd.read_csv("./data/means",header = None)
means = dict(zip(means[0],means[1]))

intervention_dicts={
    'example_dict':{
                     'Male':0.1,
                     'Mustache':np.linspace(-1.38,1.38,64)
                    },
    'second_example_dict':{
                     'Male':np.linspace(-1.38,1.38,8),
                     'Mustache':np.linspace(-1.38,1.38,8)
                    },
    'Eyeglasses':{'Eyeglasses':[0.12,-1] #~2*p}
                    },
    'Narrow_Eyes':{'Narrow_Eyes':[0.22,-1.5]
                    },
    'Mustache':{'Mustache':[1,-1]},
    'repMustache':{'Mustache':np.repeat([1,-1],64)},
    #'Mustache':{'Mustache':[0.2,-2]},
    'Smiling':{'Smiling':[1,-1]},
    # 'Big_Eyes':{'Narrow_Eyes':-1.6,
    #                 'Male':np.random.uniform(-1,1,8),
    #                 'Smiling':np.random.uniform(-1,1,8)
    #                 },

    'Young':{'Young':[1.5,-0.5]},#lab spec
    'npYoung':{'Young':np.repeat([1.5,-0.5],32) },#lab spec
    #'Young':{'Young':[1.,-1.]},
    'd_Young':{'Young':'model_default'},
    'd_Smiling':{'Smiling':'model_default'},
    #'Smiling': {'Smiling':[1.,-1.1]},

    #'repYoung':{'Young':np.repeat([1.5,-0.5],64)},#lab spec
    'rep2Young':{'Young':np.repeat([1.,-1.],64)},#lab spec
    'repSmiling': {'Smiling':np.repeat([1.,-1.1],64)},
    'd_Bald':{'Bald':'model_default'},
    'repBald':{'Bald':np.repeat([1.,-4],64)},
    'rep2Bald':{'Bald':np.repeat([0.1,-4],64)},


    'mNarrowEyeglasses':{'Eyeglasses'  :np.repeat([0.6,-0.6],4),
                         'Narrow_Eyes': np.repeat([0.6,-0.6],4)},


    'mNarrowEyeglasses':{'Eyeglasses'  :np.repeat([0.6,-0.6],4),
                         'Narrow_Eyes': np.repeat([0.6,-0.6],4)},

    'repLipstick':{'Wearing_Lipstick':np.repeat([1,-1],64)},
    'dLipstick':{'Wearing_Lipstick':'model_default'},

    #'repMustache':{'Mustache':np.repeat([0.2,-2],64)},
    #'repMustache':{'Mustache':np.repeat([1,-1],64)},
    'perMustache':{'Mustache':'percentile'},

    'mMustache':{'Mustache':[0.6,-0.6]},
    'mLipstick':{'Wearing_Lipstick':[0.1,-0.1]},
    #'mLipstick':{'Wearing_Lipstick':[0.6,-0.6]},
    'mEyeglasses':{'Eyeglasses'  :[0.6,-0.6]},
    'mNarrow_Eyes':{'Narrow_Eyes':[0.6,-0.6]},
    'mBald':{'Bald':[0.6,-0.6]},
    'mSmiling':{'Smiling':[0.6,-0.6]},

    'mNarrowEyeglasses':{'Eyeglasses'  :np.repeat([0.6,-0.6],4),
                         'Narrow_Eyes': np.repeat([0.6,-0.6],4)},

    'MaleLipstick':{  'Male':np.repeat([1.,-1.],4),
                      'Wearing_Lipstick':np.repeat([1,-1],4)},

    'perMale':{'Male':'percentile'},

    'd_Young':{'Young':'model_default'},
    'd_Smiling':{'Smiling':'model_default'},
    'd_Gray':{'Gray_Hair':'model_default'},
    'd_Male':{'Male':'model_default'},
    'd_Mustache':{'Mustache':'model_default'},
    'd_MSO':{'Mouth_Slightly_Open':'model_default'},
    'd_Smiling':{'Smiling':'model_default'},
    'd_Eyeglasses':{'Eyeglasses':'model_default'},
    'd_Bald':{'Bald':'model_default'},
    'd_Narrow_Eyes':{'Narrow_Eyes':'model_default'},

    'Eyeglasses':{'Eyeglasses':[0.11,-1.7]},

    'Gray':{'Gray_Hair':[0.2,-2.]},
    'Male':{'Male':[1.,-1.]},
    'Wearing_Lipstick':{'Wearing_Lipstick':[1,-1]},
    'MSO':{'Mouth_Slightly_Open':[1,-1]},

    #Third example: (only began implemented)
    'gender_lipstick_default':{
                    'Male':'model_default',
                    'Wearing_Lipstick':'model_default',
                    },



}




def get_do_dict(do_dict_name):
    '''
    product of the dimensions of all interventions must be divisible by
    batch_size. Strings can be any model attribute
    '''

    if not do_dict_name in intervention_dicts.keys():
        raise ValueError('do_dict_name:',do_dict_name,' is not in\
                         causal_intervention.intervention_dicts')

    return intervention_dicts[do_dict_name]



