import numpy as np
import matplotlib.pyplot as plt

class StateSpace():
    def __init__(self,A,B,C,D,x0,dt):        
        #state space matrices
        assert A.shape[0]!=A.shape[1] , "A must be square matrix"
        assert x0.shape!=[A.shape[0],1], "Initial state must have shape " + str([A.shape[0],1])
        assert A.shape[0]!=B.shape[0] , "A and B must have same rows number"
        assert C.shape[0]!=D.shape[0] , "C and D must have same rows number"
        assert C.shape[1]!=A.shape[1] , "A and C must have same coulmns number"
        assert B.shape[1]!=D.shape[1] , "A and C must have same coulmns number"

        self.A = A 
        self.B = B 
        self.C = C
        self.D = D
        self.dt = dt
        #to ease calculations in every iteration
        self.__inv__ = np.linalg.inv(np.eye(A.shape[0])-A*dt)
        #loggers
        self.x=[x0]
        self.y=[]
        self.u=[]
        
    def reset(self,x0):        
        #loggers
        self.x=[x0]
        self.y=[]
        self.u=[]

    def step(self,u):
        self.u.append(u)
        self.x.append( np.dot(self.__inv__ , (np.dot( self.B*self.dt,u)+self.x[-1]) ) )
        self.y.append(self.C.dot(self.x[-1])+self.D.dot(u))
 
    def solve(self,U):
        for u in U: self.step(u)

    def plot(self, input_labels=None, labels=None, plot_state=True):
        if labels==None :       output_labels = ['Output'+str(i) for i in range(self.C.shape[0])] 
        if input_labels==None : input_labels  = ['input'+str(i) for i in range(self.B.shape[1])]  
 
        Time    = np.arange(0,len(self.y)*dt,dt)
        inputs  = np.array(self.u).T.reshape(-1,len(self.u))
        outputs = np.array(self.y).T.reshape(-1,len(self.y))

        for inp,label in zip(inputs,input_labels):          
            plt.plot(Time,inp ,'b', label =  label )
            plt.xlabel("time (s)")
            plt.ylabel(inplabel)
            plt.grid(1)
            if legend: plt.legend()
            plt.show()

        for op,label in zip(outputs,input_labels):          
            plt.plot(Time,inp ,'r', label =  label )
            plt.xlabel("time (s)")
            plt.ylabel(inplabel)
            plt.grid(1)
            if legend: plt.legend()
            plt.show()
        
        if plot_state:
           state_labels  = ['state'+str(i) for i in range(self.A.shape[0])]  
           state = np.array(self.x).T.reshape(-1,len(self.x)) 
           for st,label in zip(state,state_labels):          
               plt.plot(Time,inp ,'r', label =  label )
               plt.xlabel("time (s)")
               plt.ylabel(inplabel)
               plt.grid(1)
               if legend: plt.legend()
               plt.show()

    def get_output(self):
      return np.array(self.y)

    def get_state(self):
      return np.array(self.x)

    def get_input(self):
      return np.array(self.u)


    def get_matrix(self):
        x = np.array(self.X).reshape(len(self.X),self.X[0].shape[0],1)
        y = np.dot(self.C,x.T.reshape(-1,len(self.X))).T.reshape(len(self.X),self.C.shape[0],1)
        return x,y
