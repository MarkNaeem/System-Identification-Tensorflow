import datetime
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

class SysIden():
  def __init__(self,input_size,output_size,dt,learning_rate=5e-2):
    """The class initializer.
       Parameters:
       -----------
       input size:    int, how many inputs does the system have. 
       output size:   int, how many outputs does the system have. 
       learning rate: learnin rate for adam optimizer"""

    self.output_size = output_size
    self.input_size  = input_size
    self.dt_c = tf.constant(dt)
    self.Av = tf.Variable(tf.random_normal([output_size, output_size]))
    self.Bv = tf.Variable(tf.random_normal([output_size,  input_size]))

    self.y0 = tf.placeholder(tf.float32,[output_size,None])
    self.y = tf.placeholder(tf.float32, [output_size,None])
    self.u = tf.placeholder(tf.float32, [input_size,None])

    self.sess =  tf.Session()

    eye = tf.eye(output_size)
    temp0 = eye - self.Av * self.dt_c
    temp1 = tf.linalg.inv( temp0 )
    temp2 = tf.keras.backend.dot(self.Bv*self.dt_c,self.u) + self.y0
    y_ = tf.keras.backend.dot( temp1 , temp2) 

    self.cost = tf.reduce_mean(tf.losses.mean_squared_error(self.y, y_))
    self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
   
  def set_learning_rate(self,learning_rate):
    """sets the learning rate of the adam optimizer
       Parameters:
       -----------
       learning_rate: int, learning_rate to be set for adam optimizer 
    """
    self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.cost)
    

  def optimize(self,Y,U,iters=500,verbose=50,plot=True):
    """takes the training data to optimize the system matrices. KeyboardInterrupt can end the optimization.
       Parameters:
       -----------
       Y:       np.ndarray, system output recorded data, should be in shape (output_size, #samples).
       U:       np.ndarray, system  input recorded data, should be in shape (output_size, #samples).
       iters:   int, iterations for the optimizer.
       verbose: int, how often to print the cost value and time taken, -1 to disable printing.
       plot:    bool, plot cost values vs iterations at the end of iterations.
    """
    assert Y.shape[0] == self.output_size , "Training data output must be in size (output_size,-1)"
    assert U.shape[0] == self.input_size  , "Training data  input must be in size  (input_size,-1)"
    
    Y0 = Y[:,:-1]
    Y1 = Y[:,1:]

    costs = []

    self.sess.run(tf.global_variables_initializer())

    anow = datetime.datetime.now()
    now = datetime.datetime.now()
        
    try:    
        for it in range(iters):
          _, tk = self.sess.run([self.optimizer, self.cost], feed_dict = {self.y0:Y0 , self.y:Y1 , self.u:U})
          costs.append(tk)
          if it%verbose==0 and verbose!=-1:
              print(it,": cost =", "{:.16f}".format(tk)," took: ",datetime.datetime.now() - now)
              now = datetime.datetime.now()

    except KeyboardInterrupt: pass            
    print("\nTraining complete!"," took: ",datetime.datetime.now() - anow)
    if plot:
      plt.plot(costs)
      plt.xlabel("Iterations")
      plt.ylabel("Cost")
      plt.grid(1)
      plt.show()


  def get_matrices(self):
    """returns the values of the A, B matrices"""
    return self.sess.run(self.Av), self.sess.run(self.Bv)
