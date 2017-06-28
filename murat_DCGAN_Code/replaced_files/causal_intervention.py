import numpy as np
import pandas as pd
means = pd.read_csv("./data/means",header = None)
means = dict(zip(means[0],means[1]))

#int_val = 0.847
int_val = 4.0 #This makes sense for integer pretrained cc

intervention_dicts={
    'example_dict':{
                     'Male':0.1,
                     'Mustache':np.linspace(-1.38,1.38,64)
                    },
    'second_example_dict':{
                     'Male':np.linspace(-1.38,1.38,8),
                     'Mustache':np.linspace(-1.38,1.38,8)
                    },
    'Eyeglasses':{'Eyeglasses':[int_val,-int_val] #~2*p}
                    },
    'Narrow_Eyes':{'Narrow_Eyes':[int_val,-int_val]
                    },
    'Narrow_Eyes_Eyeglasses1':{'Narrow_Eyes':[int_val,-int_val],'Eyeglasses':[int_val] #~2*p}
                    },
    'Bald':{'Bald':[int_val,-int_val]},
    'No_Beard':{'No_Beard':[int_val,-int_val]},
    'Mustache':{'Mustache':[int_val,-int_val]},
    'Mustache_Male1':{'Mustache':[int_val,-int_val], 'Male':[int_val]},
    'Mustache_Male0':{'Mustache':[int_val,-int_val], 'Male':[-int_val]},
    'Mustache_Wearing_Lipstick1':{'Mustache':[int_val,-int_val], 'Wearing_Lipstick':[int_val]},
    'Smiling':{'Smiling':[int_val,-int_val]},
    'Smiling_Young0':{'Smiling':[int_val,-int_val], 'Young':[-int_val]},
    # 'Big_Eyes':{'Narrow_Eyes':-1.6,
    #                 'Male':np.random.uniform(-1,1,8),
    #                 'Smiling':np.random.uniform(-1,1,8)
    #                 },
    'Young':{'Young':[int_val,-int_val]},
    'Gray':{'Gray':[int_val,-int_val]},
    'Male':{'Male':[int_val,-int_val]},
    'Wearing_Lipstick':{'Wearing_Lipstick':[int_val,-int_val]},
    'Lipstick_Male1':{'Wearing_Lipstick':[int_val,-int_val], 'Male':[int_val]},
    'Lipstick_Male0':{'Wearing_Lipstick':[int_val,-int_val], 'Male':[-int_val]},
    'Male_Lipstick1':{'Wearing_Lipstick':[int_val], 'Male':[int_val,-int_val]},
    'MSO':{'Mouth_Slightly_Open':[int_val,-int_val]},
    'Mouth_Slightly_Open':{'Mouth_Slightly_Open':[int_val,-int_val]},

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



