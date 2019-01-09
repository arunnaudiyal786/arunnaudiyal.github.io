---
layout: post
title: "Learning Rate multipliers in Keras"
author: kunal
---

Changing learning rate multipliers for different layers might seem like a trivial task in Caffe, but in keras you would need to write your own optimizer before the keras community adds it to the upcoming releases. Here is a small example enabling the use of Learning rate multipliers in keras (tried on version 2.1.6).

First, let's begin by writing a new SGD Optimizer called LR_SGD. Save the file as LR_SGD.py .

{% highlight python %} 
from keras.legacy import interfaces
import keras.backend as K
from keras.optimizers import Optimizer

class LR_SGD(Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter updates momentum.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.
    """

    def __init__(self, lr=0.01, momentum=0., decay=0.,
                 nesterov=False,multipliers=None,**kwargs):
        super(LR_SGD, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
        self.initial_decay = decay
        self.nesterov = nesterov
        self.lr_multipliers = multipliers

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr *= (1. / (1. + self.decay * K.cast(self.iterations,
                                                  K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments
        for p, g, m in zip(params, grads, moments):
            
            matched_layer = [x for x in self.lr_multipliers.keys() if x in p.name]
            if matched_layer:
                new_lr = lr * self.lr_multipliers[matched_layer[0]]
            else:
                new_lr = lr

            v = self.momentum * m - new_lr * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                new_p = p + self.momentum * v - new_lr * g
            else:
                new_p = p + v

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates

    def get_config(self):
        config = {'lr': float(K.get_value(self.lr)),
                  'momentum': float(K.get_value(self.momentum)),
                  'decay': float(K.get_value(self.decay)),
                  'nesterov': self.nesterov}
        base_config = super(LR_SGD, self).get_config()
        return dict(list(base_config.items()) + list(config.items())) 
{% endhighlight %}

The multipliers need to be multiplied with the base learning rate before the updates are applied.

Here is a sample training script using LR_SGD as the optimizer. For simplicity, we train on MNIST data.

{% highlight python %} 

from keras.datasets import mnist
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPool2D, Dropout, Flatten, Dense
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from LR_SGD import LR_SGD

# Loading mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#reshaping the data
# assuming 'channels_last' format
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)

# Normalizing
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255.
X_test /= 255.

#creating the model
input_layer = Input(shape=(28,28,1))
x = Conv2D(32,3,activation='relu',padding='same',input_shape=(1,28,28),use_bias=False,name='c1')(input_layer)
x = Conv2D(32,3,activation='relu',padding='same',use_bias=False,name='c2')(x)
x = MaxPool2D(pool_size=(2,2))(x)
x = Dropout(0.2)(x)
x = Flatten()(x)
x = Dense(128, activation='relu',use_bias=False,name='d1')(x)
x = Dropout(0.2)(x)
x = Dense(10,activation='softmax',name='d2')(x)
model = Model(inputs=input_layer, outputs=x)

# Setting the Learning rate multipliers
LR_mult_dict = {}
LR_mult_dict['c1']=1
LR_mult_dict['c2']=1
LR_mult_dict['d1']=2
LR_mult_dict['d2']=2

# Setting up optimizer
base_lr = 0.1
momentum = 0.9
optimizer = LR_SGD(lr=base_lr, momentum=momentum, decay=0.0, nesterov=False,multipliers = LR_mult_dict)

# callbacks
checkpoint = ModelCheckpoint('weights.h5', monitor='loss', verbose=0, save_best_only=False, save_weights_only=True, mode='min')

#compiling
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# training
model.fit(x = X_train, y=Y_train,callbacks=[checkpoint], batch_size=100, epochs=2)

# testing
_,score =model.evaluate(x=X_test, y=Y_test, batch_size=100)

print "Test Score:",score

{% endhighlight %}
