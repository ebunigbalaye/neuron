import numpy as np
import random as random

def training_data():
    """
    Generates training data based on a given line equation.
    
    Parameters:
    line_equation (function): A function that takes a single input (x) and returns the corresponding y value.
    
    Returns:
    x_values (numpy array): Input features.
    y_values (numpy array): Target values.
    """
    x_values = np.linspace(-10, 10, 100)
    y_values = [line_equation(x) for x in x_values] 
    return x_values, y_values

def line_equation(x):
    """
    Example line equation: y = 2x + 1
    
    Parameters:
    x (numpy array): Input features.
    
    Returns:
    numpy array: Corresponding y values
    """
    return  5*x + 2