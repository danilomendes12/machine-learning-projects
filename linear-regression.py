import numpy as np 
import matplotlib.pyplot as plt 

LEARNING_RATE = 0.0001
NUMBER_OF_ITERATIONS = 10000

def linear_regression(x, y): 
    # number of observations/points 
    m = np.size(x)

    b_0 = 1
    b_1 = 1

    for _ in range(0, NUMBER_OF_ITERATIONS):
        b_0 = b_0 - (LEARNING_RATE * (1/m) * get_cost_for_b0(x, y, b_0, b_1))
        b_1 = b_1 - (LEARNING_RATE * (1/m) * get_cost_for_b1(x, y, b_0, b_1))

    return(b_0, b_1)

def get_cost_for_b0(x, y, b_0, b_1):
    sum = 0
    
    for i in range(0, np.size(x)):
        x_temp = x[i]
        y_temp = y[i]
        h = b_0 + b_1 * x_temp
        sum+= h - y_temp
    return sum

def get_cost_for_b1(x, y, b_0, b_1):
    sum = 0

    for i in range(0, np.size(x)):
        x_temp = x[i]
        y_temp = y[i]
        h = b_0 + b_1 * x_temp
        sum+=  (h - y_temp) * x_temp
    return sum

def print_result(x, y, b):
    print("Estimated coefficients:\na = {}  \nb = {}".format(b[0], b[1]))

    # plotting the actual points as scatter plot 
    plt.scatter(x, y, color = "g", 
               marker = "o", s = 30) 
  
    # predicted response vector 
    y_pred = b[0] + b[1]*x 
  
    # plotting the regression line 
    plt.plot(x, y_pred, color = "b") 
  
    # putting labels 
    plt.xlabel('x') 
    plt.ylabel('y') 
  
    # function to show plot 
    plt.show() 
  
def main(): 
    # input 
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

    #calculate
    b = linear_regression(x, y) 
  
    print_result(x, y, b) 
  
if __name__ == "__main__": 
    main() 