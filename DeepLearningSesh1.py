import tensorflow as tf
class MyDenseLayer(tf.keras.layers.Layer):
    def __init__(self,input_dim, output_dim):
        super(MyDenseLayer, self).__init__()
        
        #Initialize weighs and bias
        self.W=self.add_weight([input_dim,output_dim])
        self.b=self.add_weight([1, output_dim])
        
        def call(self, inputs):
            #Forward propagate the inputs
            z=tf.matmul(inputs,self.W)+self.b
            
            # Feed through a non-linear activation
            output = tf.math.sigmoid(z)
            return output
        
# Deduced to short 1 line code
# import tensorflow as tf
# layer = tf.keras.layers.Dense(units=2)

# Multi Output perception
# import tensorflow as tf
# model = tf.keras.Sequential([
#    tf.keras.layers.Dense(n),
#    tf.keras.layers.Dense(2)
# ])

# Deep Neural Network
# import tensorflow as tf
# model = tf.keras.Sequential([
#     tf.keras.layers.Dense(n1),
#     tf.keras.layers.Dense(n2),
#     .
#     .
#     .
#     .
#     tf.keras.layers.Dense(2),
#  ])

# Binary Cross Entropy Loss
# loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits (y,predicted) )

# Mean Squared Error loss
# loss = tf.reduce_mean(tf.square(tf.subtract(y,predicted)))
# loss=tf.keras.losses.MSE(y,predicted) 

# Gradient Descent
# import tensorflow as tf
# weights = tf.Variable([tf/random.normal()])
# while True:             ###Loop forever
#       with tf.GradientTape() as g:
#           loss = compute(weights)
#           gradient = g.gradient(loss, weights)
#       weights = weights - lr*gradient


#Gradient Descent Algorithms
# SGD       tf.keras.optimizers.SGD
# Adam      tf.keras.optimizers.Adam
# Adadelta  tf.keras.optimizers.Adadelta
# Adagrad   tf.keras.optimizers.Adagrad
# RMSProp   tf.keras.optimizers.RMSProp

#***********PUTTING IT ALL TOGETHER***********

# import tensorflow as tf
# model=tf.keras.Sequential([.....])

# #pick your favorite optimizer
# optimizer=tf.keras.optimizer.SGD()

# while True:             #LoopForever
#     ////forward pass through the network
#     prediction=model(x)
    
#     with tf.GradientTape() as tape:
#         #compute the loss
#         loss=compute_loss(y, prediction)
    
#     ////update the weights using the gradient
#     grads=tape.gradient(loss, model.trainable_variables)
#     optimizer.apply_gradients(zip(grads, model.trainable_variables))

#Regularization: Dropout
# tf.keras.layers.Dropout(p=0.5)

