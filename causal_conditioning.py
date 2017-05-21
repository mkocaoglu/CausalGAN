import numpy as np
import pandas as pd
means = pd.read_csv("./data/means",header = None)
means = dict(zip(means[0],means[1]))




condition_dicts={
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
    'Mustache':{'Mustache':[0.2,-2]},


    'perEyeglasses':{'Eyeglasses':'percentile'},
    'perNarrow':{'Narrow_Eyes':'percentile'},

    'perLipstick':{'Wearing_Lipstick':'percentile'},
    'perMustache':{'Mustache':'percentile'},
    'perBald':{'Bald':'percentile'},
    'perSmiling':{'Smiling':'percentile'},

    'MSO':{'Mouth_Slightly_Open':[1,-1]},

    'NarrowEyeglasses':{'Eyeglasses'  :np.repeat([0.6,-0.6],4),
                         'Narrow_Eyes': np.repeat([0.6,-0.6],4)},

    'Smiling':{'Smiling':[1,-1]},
    # 'Big_Eyes':{'Narrow_Eyes':-1.6,
    #                 'Male':np.random.uniform(-1,1,8),
    #                 'Smiling':np.random.uniform(-1,1,8)
    #                 },
    #'Young':{'Young':[1.5,-0.5]},#interv

    'repYoung':{'Young':np.repeat([1.5,-0.3],64)},


    'Male':{'Male':[1,-1]},
    'd_Male':{'Male':'model_default'},
    'd_Young':{'Young':'model_default'},
    'd_Smiling':{'Smiling':'model_default'},

    'Wearing_Lipstick':{'Wearing_Lipstick':[1,-1],'Male':-1},

    'gender_lipstick_default':{
                    'Male':'model_default',
                    'Wearing_Lipstick':'model_default',
                    },
}




def get_cond_dict(cond_dict_name):
    '''
    product of the dimensions of all interventions must be divisible by
    batch_size. Strings can be any model attribute
    '''

    if not cond_dict_name in condition_dicts.keys():
        raise ValueError('cond_dict_name:',cond_dict_name,' is not in\
                         causal_intervention.condition_dicts')

    return condition_dicts[cond_dict_name]



