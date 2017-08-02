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
    'Eyeglasses':{'Eyeglasses':[0.6,-0.6] #~2*p}
                    },
    'Narrow_Eyes':{'Narrow_Eyes':[0.6,-0.6]
                    },
    'Narrow_Eyes_Eyeglasses1':{'Narrow_Eyes':[0.6,-0.6],'Eyeglasses':[0.6] #~2*p}
                    },
    'Bald':{'Bald':[0.6,-0.6]},
    'No_Beard':{'No_Beard':[0.6,-0.6]},
    'Mustache':{'Mustache':[0.6,-0.6]},
    'Mustache_Male1':{'Mustache':[0.6,-0.6], 'Male':[0.6]},
    'Mustache_Male0':{'Mustache':[0.6,-0.6], 'Male':[-0.6]},
    'Mustache_Wearing_Lipstick1':{'Mustache':[0.6,-0.6], 'Wearing_Lipstick':[0.6]},
    'Smiling':{'Smiling':[0.6,-0.6]},
    'Smiling_Young0':{'Smiling':[0.6,-0.6], 'Young':[-0.6]},
    # 'Big_Eyes':{'Narrow_Eyes':-1.6,
    #                 'Male':np.random.uniform(-1,1,8),
    #                 'Smiling':np.random.uniform(-1,1,8)
    #                 },
    'Young':{'Young':[0.6,-0.6]},
    'Gray':{'Gray':[0.8,-0.8]},
    'Male':{'Male':[0.6,-0.6]},
    'Wearing_Lipstick':{'Wearing_Lipstick':[0.8,-0.8]},
    'Lipstick_Male1':{'Wearing_Lipstick':[0.6,-0.6], 'Male':[0.6]},
    'Lipstick_Male0':{'Wearing_Lipstick':[0.6,-0.6], 'Male':[-0.6]},
    'Male_Lipstick1':{'Wearing_Lipstick':[0.6], 'Male':[0.6,-0.6]},
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



