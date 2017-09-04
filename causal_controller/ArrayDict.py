import numpy as np
class ArrayDict(object):

    '''
    This is a class for manipulating dictionaries of arrays
    or dictionaries of scalars. I find this comes up pretty often when dealing
    with tensorflow, because you can pass dictionaries to feed_dict and get
    dictionaries back. If you use a smaller batch_size, you then want to
    "concatenate" these outputs for each key.
    '''

    def __init__(self):
        self.dict={}
    def __len__(self):
        if len(self.dict)==0:
            return 0
        else:
            return len(self.dict.values()[0])
    def __repr__(self):
        return repr(self.dict)
    def keys(self):
        return self.dict.keys()
    def items(self):
        return self.dict.items()

    def validate_dict(self,a_dict):
        #Check keys
        for key,val in self.dict.items():
            if not key in a_dict.keys():
                raise ValueError('key:',key,'was not in a_dict.keys()')

        for key,val in a_dict.items():
            #Check same keys
            if not key in self.dict.keys():
                raise ValueError('argument key:',key,'was not in self.dict')

            if isinstance(val,np.ndarray):
                #print('ndarray')
                my_val=self.dict[key]
                if not np.all(val.shape[1:]==my_val.shape[1:]):
                    raise ValueError('key:',key,'value shape',val.shape,'does\
                                     not match existing shape',my_val.shape)
            else: #scalar
                a_val=np.array([[val]])#[1,1]shape array
                my_val=self.dict[key]
                if not np.all(my_val.shape[1:]==a_val.shape[1:]):
                    raise ValueError('key:',key,'value shape',val.shape,'does\
                                     not match existing shape',my_val.shape)
    def arr_dict(self,a_dict):
        if isinstance(a_dict.values()[0],np.ndarray):
            return a_dict
        else:
            return {k:np.array([[v]]) for k,v in a_dict.items()}


    def concat(self,a_dict):
        if self.dict=={}:
            self.dict=self.arr_dict(a_dict)#store interally as array
        else:
            self.validate_dict(a_dict)
            self.dict={k:np.vstack([v,a_dict[k]]) for k,v in self.items()}

    def __getitem__(self,at):
        return {k:v[at] for k,v in self.items()}

#debug, run tests
if __name__=='__main__':
    out1=ArrayDict()
    d1={'Male':np.ones((3,1)),'Young':2*np.ones((3,1))}
    d2={'Male':3,'Young':33}
    d3={'Male':4*np.ones((4,1)),'Young':4*np.ones((4,1))}

    out1.concat(d1)
    out1.concat(d2)

    out2=ArrayDict()
    out2.concat(d2)
    out2.concat(d1)
    out2.concat(d3)

