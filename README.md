# system-Identification-Tensorflow
The Python code for "Linear time invariant state space system identification using adam optimization" with some examples on how to use it.

Link to the paper: https://ieeexplore.ieee.org/document/9047808

### There is two python files in this repository:
**SysIden.py:** includes the system identificatoin class that builds the Tensorflow graph, takes the training data, and gives the optimized matrices.

**StateSpace.py:** includes a class that solves linear time invariant state space models. The class can solve the system one step at a time, which is a very good feature in case of using the model with reinforcment learning.
_Example: a reinforcment learning controller given an error signal to output a control signal to control a system -modeled by this class-._

The Ipython notebook shows examples about how to use the system identification class.
