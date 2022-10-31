from tensorflow.keras.datasets import mnist
from scipy import signal

import numpy as np
import pandas as pd
from time import time

def correlation(inp,kernel):
    
    height = inp.shape[0]
    width = inp.shape[1]
    kernel_size = kernel.shape[0]
    output = np.zeros((height - kernel_size + 1, width - kernel_size + 1))

    for h in range(height - kernel_size + 1):
      for w in range(width - kernel_size + 1):
          temp_matrix = inp[h:(h+kernel_size), w:(w+kernel_size)]
          output[h,w] = np.sum(temp_matrix * kernel)
    return output

def correlationFull(inp,kernel):

    kernel = np.rot90(kernel, k=2, axes=(0, 1))
    height = inp.shape[0]
    width = inp.shape[1]
    kernel_size = kernel.shape[0]

    output_height = height + kernel_size - 1
    output_width = width + kernel_size - 1
    output = np.zeros((output_height, output_width))

    temp_output_height = (kernel_size - 1) * 2 + height
    temp_output_width = (kernel_size - 1) * 2 + width
    temp_output = np.zeros((temp_output_height, temp_output_width))

    inp = np.pad(inp, (int((temp_output_height - height)/2), int((temp_output_width - width)/2)), 'constant', constant_values=0)

    middleMaker = int( (kernel_size - 1) / 2 )
    for h in range(temp_output_height-kernel_size+1):
      for w in range(temp_output_width-kernel_size+1):
          temp_matrix = inp[h:(h+kernel_size), w:(w+kernel_size)]
          temp_output[h+middleMaker, w+middleMaker] +=  np.sum(temp_matrix * kernel)
    
    output = temp_output[middleMaker:temp_output_height-middleMaker, middleMaker:temp_output_width-middleMaker]
    
    return output

class Convolution():
  def __init__ (self, kernel_size, no_filters, input_shape):
    self.kernel_size = kernel_size
    self.no_filters = no_filters

    self.kernels_shape = (no_filters, input_shape[2], self.kernel_size, self.kernel_size)
    self.filters = np.random.randn(*self.kernels_shape) / (self.kernel_size ** 2)
    
    self.output_shape = (input_shape[0] - kernel_size + 1, input_shape[1] - kernel_size + 1,no_filters)
    self.biases = np.random.randn(*self.output_shape) / (self.kernel_size ** 2)
  
  def forward_propagation(self, input):
    self.input = input
    height, width, channels = input.shape

    output_height, output_width = height - self.kernel_size + 1, width - self.kernel_size + 1
    output = np.zeros((output_height, output_width, self.no_filters))

    output = np.copy(self.biases)
    
    for k in range(self.no_filters):
      for c in range(channels):
        output[:,:,k] += correlation(self.input[:,:,c], self.filters[k,c])

        #### Comment out below and comment upper to use SCIPY function ####
        # output[:,:,k] += signal.correlate2d(self.input[:,:,c], self.filters[k,c], "valid")

    
    return output

  def backward_propagation(self, output_gradient, learn_rate):

    input_gradient = np.zeros(self.input.shape)
    kernels_gradient = np.zeros(self.kernels_shape)

    height, width, channels = self.input.shape

    output_height, output_width = height - self.kernel_size + 1, width - self.kernel_size + 1
    output = np.zeros((output_height, output_width, self.no_filters))

    for k in range(self.no_filters):
      for c in range(channels):
        kernels_gradient[k, c] = correlation(self.input[:,:,c], output_gradient[:,:,k])
        input_gradient[:,:,c] += correlationFull(output_gradient[:,:,k], self.filters[k,c])

        #### Comment out below and comment upper to use SCIPY function ####
        # kernels_gradient[k, c] = signal.correlate2d(self.input[:,:,c], output_gradient[:,:,k], "valid")
        # input_gradient[:,:,c] += signal.convolve2d(output_gradient[:,:,k], self.filters[k,c], "full")

    # Update filters
    self.filters -= learn_rate * kernels_gradient
    self.biases -= learn_rate * output_gradient.reshape(self.output_shape)

    return input_gradient

class Pooling():
  def __init__(self, pool_size):
    self.pool_size = pool_size
  
  def forward_propagation(self, input):
    self.input = input
    height, width, no_filters = input.shape

    output_height, output_width = height // self.pool_size, width // self.pool_size
    output = np.zeros((output_height, output_width, no_filters))

    for h in range(output_height):
      for w in range(output_width):
        h_start = h * self.pool_size
        w_start = w * self.pool_size
        temp_matrix = input[h_start:(h_start + self.pool_size), w_start:(w_start + self.pool_size)]
        output[h,w] = np.amax(temp_matrix, axis=(0,1))

    return output

  def backward_propagation(self,output_gradient):
    d_L_d_input = np.zeros(self.input.shape)

    height, width, no_filters = self.input.shape
    output_height, output_width = height // self.pool_size, width // self.pool_size
    for h in range(output_height):
      for w in range(output_width):
        h_start = h * self.pool_size
        w_start = w * self.pool_size
        temp_matrix = self.input[h_start:(h_start + self.pool_size), w_start:(w_start + self.pool_size)]
        amax = np.amax(temp_matrix, axis=(0, 1))
        h1, w1, f1 = temp_matrix.shape
        
        for i2 in range(h1):
          for j2 in range(w1):
            for f2 in range(f1):
              # If this pixel was the max value, copy the gradient to it.
              if temp_matrix[i2, j2, f2] == amax[f2]:
                d_L_d_input[h_start+ i2, w_start+ j2, f2] = output_gradient[h, w, f2]

    return d_L_d_input

class Relu():
  def __init__(self):
      pass
  
  def forward_propagation(self, input):
      self.input = input
      relu_forward = np.maximum(0,input)
      return relu_forward
  
  def backward_propagation(self, grad_output):
      relu_grad = self.input > 0
      return grad_output*relu_grad
  
class FullConnected():
  '''
    Fully Connected Layer
  '''
  def __init__(self, input_len, no_neurons):
    input_units = input_len

    self.weights = np.random.normal(
        loc=0.0, 
        scale = np.sqrt(2/(input_units+no_neurons)), 
        size = (input_units,no_neurons))
    self.biases = np.zeros(no_neurons)
  
  def forward_propagation(self,input):
    self.input = input

    return np.dot(self.input, self.weights) + self.biases
  
  def backward_propagation(self,grad_output,lr):
    self.learning_rate = lr
    grad_input = np.dot(grad_output, self.weights.T)

    grad_weights = np.dot(self.input.T, grad_output)
    grad_biases = grad_output.mean(axis=0)*self.input.shape[0]
    
    assert grad_weights.shape == self.weights.shape and grad_biases.shape == self.biases.shape
    
    # Here we perform a stochastic gradient descent step.
    # print(grad_weights)
    self.weights = self.weights - self.learning_rate * grad_weights
    self.biases = self.biases - self.learning_rate * grad_biases
    
    return grad_input

class FlattenLayer:
  def __init__(self, input_shape):
      self.input_shape = input_shape

  def forward(self, input):
      return np.reshape(input, (1, -1))
  
  def backward(self, output_error):
      return np.reshape(output_error, self.input_shape)

def softmax(X):
    exps = np.exp(X - np.max(X))
    return exps / np.sum(exps)

def cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = 1
    p = softmax(X)
    log_likelihood = -np.log(p[range(m),y])
    loss = np.sum(log_likelihood) / m
    return loss

def delta_cross_entropy(X,y):
    """
    X is the output from fully connected layer (num_examples x num_classes)
    y is labels (num_examples x 1)
    	Note that y is not one-hot encoded vector. 
    	It can be computed as y.argmax(axis=1) from one-hot encoded vectors of labels if required.
    """
    m = 1
    grad = softmax(X)
    grad[range(m),y] -= 1
    grad = grad/m
    return grad

######## Load data from Keras ########
(train_X, train_y), (test_X, test_y) = mnist.load_data()
train_X = train_X / 255
test_X = test_X / 255


######## Initialize network layers ########
#Conv1
conv = Convolution(5,4,input_shape=(28,28,1))
conv1_relu = Relu()
#Pooling
pooling = Pooling(2)
#Conv2
conv2 = Convolution(5,8,input_shape=(12,12,4))
conv2_relu = Relu()
#Pooling
pooling2 = Pooling(2)
#Flatten
flatten = FlattenLayer(input_shape=(4,4,8))
#FC1
fc1 = FullConnected(4*4*8,128)
fc1_relu = Relu()
#FC2
linear = FullConnected(128,10)

######## Start Training ########
trainStart = time()
NO_EPOCHS = 3
LEARNING_RATE = 0.01
for epoch in range(NO_EPOCHS): # Epoch loop
  loss = 0
  num_correct = 0
  totalTime = 0
  accuracy = []

  for i, (im, label) in enumerate(zip(train_X, train_y)): # Iterate over every sample

    startTime = time()

    ######## Forward propagation ########
    #####################################
    conv_output = conv.forward_propagation(im.reshape(28,28,1))
    conv1_relu_output = conv1_relu.forward_propagation(conv_output)
    pool_output = pooling.forward_propagation(conv1_relu_output)

    conv2_output = conv2.forward_propagation(pool_output)
    conv2_relu_output = conv2_relu.forward_propagation(conv2_output)
    pool2_output = pooling2.forward_propagation(conv2_relu_output)

    flattenOut = flatten.forward(pool2_output)

    fc1_output = fc1.forward_propagation(flattenOut)
    fc1_output = fc1_relu.forward_propagation(fc1_output)

    output = linear.forward_propagation(fc1_output)

    acc = 1 if np.argmax(output) == label else 0

    ######## Loss Calculation ########
    l = cross_entropy(output, label)

    ######## Backward Propagation ########
    ######################################
    grad_out = delta_cross_entropy(output, label)

    grad_out = linear.backward_propagation(grad_out,lr=LEARNING_RATE)

    grad_out = fc1_relu.backward_propagation(grad_out)
    grad_out = fc1.backward_propagation(grad_out,lr=LEARNING_RATE)

    grad_out = flatten.backward(grad_out)

    grad_out = pooling2.backward_propagation(grad_out)
    grad_out = conv2_relu.backward_propagation(grad_out)
    grad_out = conv2.backward_propagation(grad_out, LEARNING_RATE)

    grad_out = pooling.backward_propagation(grad_out)
    grad_out = conv1_relu.backward_propagation(grad_out)
    grad_out = conv.backward_propagation(grad_out, LEARNING_RATE)
    

    # Update metrics
    loss += l
    num_correct += acc
    totalTime += time() - startTime

    # Print iteration after every 100 samples
    if i % 100 == 99:
      accuracy.append(num_correct/100)
      print(
        '[Epoch %d][Step %d] Past 100 steps: Duration : %.3f | Average Loss %.3f | Accuracy: %d%% | Average Accuracy: %.2f%%' %
        (epoch + 1, i + 1, totalTime, loss / 100, num_correct, np.mean(accuracy)*100)
      )
      loss = 0
      num_correct = 0
      totalTime = 0

# Training Summary
print("Total Training Time:", time() - trainStart)
print("Train Accuracy:",round(np.mean(accuracy)*100,2))


# Testing Iteration
lines = []
for i, (im, label) in enumerate(zip(test_X, test_y)):

  conv_output = conv.forward_propagation(im.reshape(28,28,1))
  conv1_relu_output = conv1_relu.forward_propagation(conv_output)
  pool_output = pooling.forward_propagation(conv1_relu_output)

  conv2_output = conv2.forward_propagation(pool_output)
  conv2_relu_output = conv2_relu.forward_propagation(conv2_output)
  pool2_output = pooling2.forward_propagation(conv2_relu_output)

  flattenOut = flatten.forward(pool2_output)

  fc1_output = fc1.forward_propagation(flattenOut)
  fc1_output = fc1_relu.forward_propagation(fc1_output)

  output = linear.forward_propagation(fc1_output)

  lines.append([np.argmax(output), label])

# Collect actual and predicted values and calculate accuracy
dfFinal = pd.DataFrame(lines, columns=["predicted","actual"])
print("Test Accuracy:",round(len(dfFinal[dfFinal.predicted == dfFinal.actual]) / len(dfFinal) * 100,2))