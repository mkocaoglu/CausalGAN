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
    'Eyeglasses':{'Eyeglasses':[0.8,-0.8] #~2*p}
                    },
    'Narrow_Eyes':{'Narrow_Eyes':[0.8,-0.8]
                    },
    'Mustache':{'Mustache':[0.8,-0.8]},
    'Smiling':{'Smiling':[1,-1]},
    # 'Big_Eyes':{'Narrow_Eyes':-1.6,
    #                 'Male':np.random.uniform(-1,1,8),
    #                 'Smiling':np.random.uniform(-1,1,8)
    #                 },
    'Young':{'Young':[0.8,-0.8]},
    'Gray':{'Gray':[0.8,-0.8]},
    'Male':{'Male':[0.8,-0.8]},
    'Wearing_Lipstick':{'Wearing_Lipstick':[0.8,-0.8]},
    'MSO':{'Mouth_Slightly_Open':[0.8,-0.8]},

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



