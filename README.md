## Genetic-Programming
Genetic Programming example for Symbolic Regression

### Context
I practiced genetic programming by implementing a solver for benchmark symbolic regression problems, using a public dataset for training, and an hidden dataset for testing.
The goal is actually to find an expression that fits the given dataset with MSE as small as possible. Let’s take a look at an example : Suppose your training data set includes tuples (x,y), from which you want to learn the best model f such that y = f(x) explains the given data set, as well as unseen test data set, as precisely as possible. If the training data consist of X = 1.2, 2, 3 and Y = 3.1,4.6,6.8, and the test data set consist of X′ = {6,5} and Y ′ = {13,10.5}, one possible symbolic regression model would be y = 2x + 1.

We can actually build candidate expressions for f(x) using genetic programming, representing these solutions by trees. We can evaluate the efficiency of a solution using Mean Squared Error (MSE), which is equals to the mean squared error between expected value and real one.

### How to use scripts : 
Parameters.py : parameters for training.py  
training.py : generates the best found RPN (python training.py train.csv)  
test.py : computes MSE of a RPN. (python test.py "RPN" test.csv)  

### Abbreviations : 
RPN = Reverse Polish Notation  
GP = Genetic Programming  
GA = Genetic Algorithm  
RWS = Roulette Wheel Selection  
MSE = Mean Sqaured Error  

### Contact :
Rafael Cartenet : rafael.cartenet@insa-rouen.fr  
