'''
To use a particular causal graph, just specify it here


Strings specified have to match *exactly* to keys in attribute text file


A graph lists each node and it's parents in pairs

A->B, C->D, D->B:
    [['A',[]],
     ['B',['A','D']],
     ['C',[]],
     ['D',[]]]

'''

all_nodes=[
        ['5_o_Clock_Shadow',[]],
        ['Arched_Eyebrows',[]],
        ['Attractive',[]],
        ['Bags_Under_Eyes',[]],
        ['Bald',[]],
        ['Bangs',[]],
        ['Big_Lips',[]],
        ['Big_Nose',[]],
        ['Black_Hair',[]],
        ['Blond_Hair',[]],
        ['Blurry',[]],
        ['Brown_Hair',[]],
        ['Bushy_Eyebrows',[]],
        ['Chubby',[]],
        ['Double_Chin',[]],
        ['Eyeglasses',[]],
        ['Goatee',[]],
        ['Gray_Hair',[]],
        ['Heavy_Makeup',[]],
        ['High_Cheekbones',[]],
        ['Male',[]],
        ['Mouth_Slightly_Open',[]],
        ['Mustache',[]],
        ['Narrow_Eyes',[]],
        ['No_Beard',[]],
        ['Oval_Face',[]],
        ['Pale_Skin',[]],
        ['Pointy_Nose',[]],
        ['Receding_Hairline',[]],
        ['Rosy_Cheeks',[]],
        ['Sideburns',[]],
        ['Smiling',[]],
        ['Straight_Hair',[]],
        ['Wavy_Hair',[]],
        ['Wearing_Earrings',[]],
        ['Wearing_Hat',[]],
        ['Wearing_Lipstick',[]],
        ['Wearing_Necklace',[]],
        ['Wearing_Necktie',[]],
        ['Young',[]]
    ]

subset1_nodes=[
        ['Bald',[]],
#        ['Blurry',[]],
#        ['Brown_Hair',[]],
#        ['Bushy_Eyebrows',[]],
#        ['Chubby',[]],
        ['Double_Chin',[]],
#        ['Eyeglasses',[]],
#        ['Goatee',[]],
#        ['Gray_Hair',[]],
        ['Male',[]],
        ['Mustache',[]],
        ['No_Beard',[]],
        ['Smiling',[]],
#        ['Straight_Hair',[]],
#        ['Wavy_Hair',[]],
        ['Wearing_Earrings',[]],
#        ['Wearing_Hat',[]],
        ['Wearing_Lipstick',[]],
        ['Young',[]]
    ]


standard_graph=[
       ['Male'   , []              ],
       ['Young'  , []              ],
       ['Smiling', ['Male','Young']]
       ]

male_causes_beard=[
        ['Male',[]],
        ['No_Beard',['Male']],
    ]
male_causes_mustache=[
        ['Male',[]],
        ['Mustache',['Male']],
    ]

big_causal_graph=[
        ['Young',[]],
        ['Male',[]],
        ['Eyeglasses',['Young']],
        ['Bald',            ['Male','Young']],
        ['Mustache',        ['Male','Young']],
        ['Smiling',         ['Male','Young']],
        ['Wearing_Lipstick',['Male','Young']],
        ['Mouth_Slightly_Open',['Smiling']],
        ['Narrow_Eyes',        ['Smiling']],
    ]
#old_big_causal_graph=big_causal_graph#fyi

new_big_causal_graph=[
        ['Young',[]],
        ['Male',[]],
        ['Eyeglasses',['Young']],
        ['Bald',            ['Male','Young']],
        ['Mustache',        ['Male','Young']],
        ['Smiling',         ['Male','Young']],
        ['Wearing_Lipstick',['Male','Young']],
        ['Mouth_Slightly_Open',['Young','Smiling']],
        ['Narrow_Eyes',        ['Male','Young','Smiling']],
    ]

male_ind_mustache = [
        ['Male',[]],
        ['Mustache',[]]
    ]

male_smiling_lipstick=[
       ['Male'   , []],
       ['Wearing_Lipstick'  , ['Male']],
       ['Smiling', ['Male']]
       ]

def get_causal_graph(causal_model=None,*args,**kwargs):

    if causal_model is 'male.young.smiling':
        graph=standard_graph
    elif causal_model == 'subset1':
        graph=subset1_nodes
    elif causal_model == 'male_causes_beard':
        graph = male_causes_beard
    elif causal_model == 'male_causes_mustache':
        graph = male_causes_mustache
    elif causal_model == 'big_causal_graph':
        graph = big_causal_graph
    elif causal_model == 'male_ind_mustache':
        graph = male_ind_mustache
    elif causal_model == 'male_smiling_lipstick':
        graph = male_smiling_lipstick

    elif causal_model == 'new_big_causal_graph':
        graph = new_big_causal_graph


    if causal_model is 'empty':
        graph=[[],[]]


    return graph
