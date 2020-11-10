import tensorflow as tf
from tensorflow.keras.optimizers.schedules import LearningRateSchedule

class ExponentialDecayGANPaper(LearningRateSchedule):
    """A LearningRateSchedule that uses a custom exponential decay schedule diferente to 
    `tensorflow.keras.optimizers.schedules.ExponentialDecay`.
    

    This LearningRateSchedule is based on the one implemented in the `pylearn2` library,
    which is used by *Goodfellow et al.* in the Generative Adversarial Nets article
    available at https://arxiv.org/pdf/1406.2661.pdf to train GAN models

    Arguments:

        init_lr: float, initial learning rate to which it will be applied Exponential decay.
        decay_factor: float, factor by which the learning rate is decreased, mathematically 
            you can see it as: learning_rate(step) = init_lr / (decay_factor ** step)
        min_lr: float, lower limit of learning rate.

    Returns:
        A 1-arg callable learning rate schedule that takes the current optimizer
        step and outputs the decayed learning rate, a scalar `Tensor` of the same
        type as `initial_learning_rate`.
    """
    def __init__(self, init_lr=.1, decay_factor = 1.0003, min_lr=1e-6):
        self.init_lr = tf.cast(init_lr, tf.float32)
        self.decay_factor = tf.cast(decay_factor, tf.float32)
        self.min_lr = tf.cast(min_lr, tf.float32)
    
    def __call__(self, step):
        step = tf.cast(step, tf.float32)
        lr = self.init_lr * tf.math.pow(1./self.decay_factor, step)
        return tf.cond(lr > self.min_lr, lambda: lr, lambda: self.min_lr)
    
    def get_config(self):
        return {
            'init_lr': self.init_lr,
            'decay_factor': self.decay_factor,
            'min_lr': self.min_lr,
        } 

class MomentumAdjustor:
    """A callable object that adjusts momentum according to the epoch
    

    This MomentumAdjustor is based on the one implemented in the `pylearn2` library,
    which is used by *Goodfellow et al.* in the Generative Adversarial Nets article
    available at https://arxiv.org/pdf/1406.2661.pdf to train GAN models

    Arguments:

        init_momentum: float [0., 1.], initial momentum in training loop.
        final_momentum: float [0., 1.], momentum after satured epoch on training loop.
        start: int, initial epoch when the momentum value is diferent to 0.0.
        sturate: int, Epoch from which momentum value is always 1.0.

    Returns:
        A callable object tha takes current epoch and return Momentum value.
    """
    def __init__(self, init_momentum=.5, final_momentum=.7, start=1, saturate=250):
        self.init_momentum=init_momentum
        self.final_momentum=final_momentum
        self.start=start
        self.saturate=saturate
        
    def __call__(self, epoch):
        eta = float(epoch - self.start) / float(self.saturate - self.start)
        if eta < 0.:
            eta = 0.
        elif eta > 1.:
            eta = 1.
        return self.init_momentum*(1. - eta) + eta*self.final_momentum