import numpy as np 


def neuron(input, weight=0, bias=0):
    """ This is the calculation the neuron performs. It computes the output of a single artificial neuron given an input, weight, and bias."""
    output = weight * input + bias
    return output

def mean_squared_error(true_output, predicted_output):
    """This computes the loss of the neuron by calculating the mean squared error between all the true values and the predicted values."""
    return np.mean((true_output - predicted_output) ** 2)

def predict(input_values,weight,bias):
    """ This function predicts the output values for a given set of input values using the neuron with a weight and bias of 0."""
    predicted_values = np.zeros(len(input_values))
    for i, value in enumerate(input_values):
        prediction = neuron(value,weight,bias)
        predicted_values[i] = prediction
    return predicted_values

def gradient(input_values, output_values,weight,bias):
    """
    Computes the gradient of the mean squared error with respect to the weight and bias.
    
    Returns:
    weight_gradient (float): Gradient with respect to the weight.
    bias_gradient (float): Gradient with respect to the bias.
    """
  
    predicted_values = neuron(input_values,weight,bias)
    error = predicted_values - output_values
    weight_gradient = (2/len(input_values)) * np.dot(error, input_values)
    bias_gradient = (2/len(input_values)) * np.sum(error)
    return weight_gradient, bias_gradient

def update_parameters(weight, bias, weight_gradient, bias_gradient, learning_rate):
    """
    Updates the weight and bias using the computed gradients and a specified learning rate.
    
    Returns:
    updated_weight (float): Updated weight after applying gradient descent.
    updated_bias (float): Updated bias after applying gradient descent.
    """
    updated_weight = weight - learning_rate * weight_gradient
    updated_bias = bias - learning_rate * bias_gradient

    return updated_weight, updated_bias


def train(learning_rate,input_values,output_values,no_iters=100):
    weight = 0.0
    bias = 0.0
    weights = np.zeros(no_iters)
    biases = np.zeros(no_iters)
    losses = np.zeros(no_iters)
    weight_gradients = np.zeros(no_iters)
    bias_gradients = np.zeros(no_iters)



    for i in range(1,no_iters):
    
        predictions = predict(input_values, weight, bias)
        loss = mean_squared_error(output_values, predictions)
        losses[i] = loss

        weight_gradient, bias_gradient = gradient(input_values, output_values,weight,bias)
        weight_gradients[i] =  weight_gradient
        bias_gradients[i] = bias_gradient
        
        weight, bias = update_parameters(weight, bias, weight_gradient, bias_gradient, learning_rate=learning_rate)
        weights[i] = weight
        biases[i] = bias
  

    return weights, biases, losses, weight_gradients, bias_gradients