



class Trainer(object):


    def __init__(self,config,cc_config,dcgan_config,began_config):







        if config.model_type:
            if config.model_type=='dcgan'
                model_config,_ = get_dcgan_config()
            if config.model_type=='began'
                model_config,_ = get_began_config()



