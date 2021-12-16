import tensorflow as tf

class DriftCorrectNet(tf.keras.Model):
    def __init__(self):
        #todo: intitialize some variables linear drift
        #todo: penalize huge values with l1 norm
        #todo: sparsity constraint
        #todo: define input and output? linear regrission?
        #todo: restrict drift delta from frame to frame
        pass
    def __call__(self, *args, **kwargs):
        pass