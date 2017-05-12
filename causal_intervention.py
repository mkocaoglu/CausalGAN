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
    'Mustache':{'Mustache':[0.2,-2]},
    'Smiling':{'Smiling':[1,-1]},
    # 'Big_Eyes':{'Narrow_Eyes':-1.6,
    #                 'Male':np.random.uniform(-1,1,8),
    #                 'Smiling':np.random.uniform(-1,1,8)
    #                 },
    'Young':{'Young':[1.5,-0.5]},
    'Male':{'Male':[1,-1]},
    'Wearing_Lipstick':{'Wearing_Lipstick':[1,-1],'Male':-1},
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



