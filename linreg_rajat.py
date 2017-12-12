'''
    TEMPLATE FOR MACHINE LEARNING HOMEWORK
    AUTHOR Eric Eaton, Vishnu Purushothaman Sreenivasan
'''
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, init_theta=None, alpha=0.01, n_iter=100):
        '''
        Constructor
        '''
        self.alpha = alpha
        self.n_iter = n_iter
        self.theta = init_theta
        self.JHist = None
    

    def gradientDescent(self, X, y, theta):
        '''
        Fits the model via gradient descent
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
            theta is a d-dimensional numpy vector
        Returns:
            the final theta found by gradient descent
        '''
        n,d = X.shape
    
        
        self.JHist = []
        for i in xrange(self.n_iter):
            self.JHist.append( (self.computeCost(X, y, theta), theta) )
            print "Iteration: ", i+1, " Cost: ", self.JHist[i][0], " Theta: ", theta
                        
#==============================================================================
#           This is all the old unvectorized code 
#              loop over all the j's (features)
#             for j in range (0,d):
#                 sum = 0
#                 #loop over all the instances
#                 for instance in range (0,n):
#                     predicted = 0
#                     #for each instance, find the predicted y value 
#                     for k in range (0,d):
#                         predicted = predicted + theta[k] * X[instance,k]
#                     sum = sum + (predicted - y[instance])*X[instance,j]
#                 #assign new thetas to updatedTheta
#                 updatedTheta[j] = theta[j] - self.alpha * (1.0/n) * sum  
#==============================================================================
            
            theta = theta - self.alpha*(1.0/n)*(np.transpose(X)*(X*theta-y))

        return theta 
    
    def computeCost(self, X, y, theta):
        '''
        Computes the objective function
        Arguments:
          X is a n-by-d numpy matrix
          y is an n-dimensional numpy vector
          theta is a d-dimensional numpy vector
        Returns:
          a scalar value of the cost  
              ** make certain you don't return a matrix with just one value! **
        '''
        n,d = X.shape
        
        return .5* (1.0/n)* np.transpose(X*theta-y)*(X*theta-y)
             
#==============================================================================
# #        This is the non-vectorized code I wrote to find the cost 
# #        cost = 0;
# #        for instance in range (0,n):
# #            predicted = 0
# #            for k in range (0,d):
# #                predicted = predicted + theta[k] * X[instance,k]
# #            cost = cost + (predicted - y[instance])**2
# #                    
# #        return cost * .5 * (1.0/n)
#==============================================================================
                

    def fit(self, X, y):
        '''
        Trains the model
        Arguments:
            X is a n-by-d numpy matrix
            y is an n-dimensional numpy vector
        '''
        n = len(y)
        n,d = X.shape
        #self.theta = np.zeros((d,1))

        if self.theta is None:
            self.theta = np.matrix(np.zeros((d,1)))
            
        self.theta = self.gradientDescent(X,y,self.theta)    


    def predict(self, X):
        '''
        Used the model to predict values for each instance in X
        Arguments:
            X is a n-by-d numpy matrix
        Returns:
            an n-dimensional numpy vector of the predictions
        '''
        
        n,d = X.shape
        
        return X*self.theta
        
#        predictedVector = np.zeros((d,1))
#        
#        for i in range (0,n):
#            prediction = 0 
#            for j in range (0,d):
#                prediction = prediction + self.theta[j]*X[i,j]
#            
#            predictedVector[i] = prediction
#        return predictedVector

            
