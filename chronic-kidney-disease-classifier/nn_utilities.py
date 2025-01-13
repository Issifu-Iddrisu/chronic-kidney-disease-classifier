#NN for classification
#2 layers and 3 layers
#Stochastic gradient descent with mini_batches, with and without momentum updates

import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def relu(Z):
    return np.maximum(0,Z)  

def relu_derivative(x):
    return (x>0).astype(int)

def tanh(Z):
    return np.tanh(Z)

def tanh_derivative(z):
    return 1 - tanh(z)**2

def initialize_parameters(layer_dimensions):
    parameters = {}
    L = len(layer_dimensions)# No. of layers

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.normal(0, 0.001,size=(layer_dimensions[l], layer_dimensions[l-1]))
        parameters['b' + str(l)] = np.zeros((layer_dimensions[l],1))

    return parameters

def velocity_initialization(parameters):
    Vds = {key: 0 for key in parameters.keys()}
    return Vds

def forward_prop2(X, parameters):

    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    Z1 = np.dot(W1,X.T)+b1
    A1 = relu(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    output = A2

    cache = {'Z1':Z1,'A1':A1, 'Z2':Z2, 'A2':A2}
    return output, cache


def forward_prop3(X, parameters):

    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    b1 = parameters['b1']
    b2 = parameters['b2']
    b3 = parameters['b3']

    Z1 = np.dot(W1,X)+b1
    A1 = relu(Z1)
    Z2 = np.dot(W2,A1)+b1
    A2 = relu(Z2)
    Z3 = np.dot(W3,A2)+b3
    A3 = sigmoid(Z3)
    output = A3

    cache = {'Z1':Z1,'A1':A1, 'Z2':Z2, 'A2':A2, 'Z3':Z3, 'A3': A3}
    return output, cache



def cost(A, Y):
    m = Y.shape[0] 
    return (-1/m)*np.sum(np.dot(np.log(A), Y) + np.dot(np.log(1-A),1-Y))


def backward_prop2(X,Y,cache, parameters):

    m= len(X)
   
    W2 = parameters['W2']

    Z1 = cache['Z1']
    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y.T
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2)
    dZ1 = np.dot(W2.T, dZ2)*relu_derivative(Z1)
    dW1 = (1/m)*np.dot(dZ1,X)
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)

    return {'dW1': dW1,'db1':db1, 'dW2':dW2, 'db2':db2}

def backward_prop3(X,Y,cache, parameters):
    
    m= len(X)
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    A1 = cache['A1']
    A2 = cache['A2']
    A3 = cache['A3']

    # return {'dW1': dW1,'db1':db1, 'dW2':dW2, 'db2':db2, 'dW3':dW3, 'db3':db3}


def update_parameters(grads,parameters, learning_rate):
    for key in parameters.keys():
        parameters[key] -= learning_rate*grads['d'+key]
    return parameters

def momentum_parameters_update(parameters, gradients, learning_rate, momentum_rate,Vds):
        for key in parameters.keys():
                Vd = Vds[key]
                gradient = gradients['d'+key]
                Vds[key] = momentum_rate*Vd +(1-momentum_rate)*gradient
                parameters[key] -= learning_rate*Vds[key]
        return parameters, Vds

def make_predictions(X,Y,parameters,model):
    if model =='2':
        A,_ = forward_prop2(X,parameters)
        predict = (A > 0.5)
        accuracy = (np.sum(predict == Y.T))/len(Y)
    
    if model =='3':
        A = forward_prop3(X,parameters)
        predict = (A > 0.5) 
        accuracy = (np.sum(predict == Y.T))/len(Y)
    
    return predict, accuracy

def train_model2(X_train,Y_train,X_val,Y_val,mini_batch_size,layers_dimension,learning_rate,max_epochs):
    
    #initialize parameters
    parameters = initialize_parameters(layers_dimension)

    train_cost= []
    val_cost  = []
    for i in range(max_epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]
        train_cost_batch = []
        val_cost_batch = []

        for i in range(0,len(X_train), mini_batch_size):
            X_batch = X_train_shuffled[i:i+mini_batch_size]
            Y_batch = Y_train_shuffled[i:i+mini_batch_size]
            
            output,cache  = forward_prop2(X_batch, parameters)
            A_val, _ = forward_prop2(X_val, parameters)
            t_loss = cost(output, Y_batch)
            v_loss = cost(A_val,Y_val)
            grads = backward_prop2(X_batch,Y_batch,cache, parameters)
            parameters = update_parameters(grads,parameters,learning_rate)

            val_cost_batch.append(v_loss)
            train_cost_batch.append(t_loss)
  
        train_cost.append(np.mean(train_cost_batch ))
        val_cost.append(np.mean(val_cost_batch))
    return parameters, train_cost, val_cost

def train_modelwithmomentum(X_train,Y_train,X_val,Y_val,mini_batch_size,layers_dimension,learning_rate,max_epochs,momentum_rate):
        #initialize parameters
    parameters = initialize_parameters(layers_dimension)
    Vds = velocity_initialization(parameters)

    train_cost= []
    val_cost  = []
    for i in range(max_epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]
        train_cost_batch = []
        val_cost_batch = []

        for i in range(0,len(X_train), mini_batch_size):
            X_batch = X_train_shuffled[i:i+mini_batch_size]
            Y_batch = Y_train_shuffled[i:i+mini_batch_size]
            
            output,cache  = forward_prop2(X_batch, parameters)
            A_val, _ = forward_prop2(X_val, parameters)
            t_loss = cost(output, Y_batch)
            v_loss = cost(A_val,Y_val)
            grads = backward_prop2(X_batch,Y_batch,cache, parameters)
            parameters, Vds = momentum_parameters_update(parameters, grads, learning_rate, momentum_rate,Vds)
            
            val_cost_batch.append(v_loss)
            train_cost_batch.append(t_loss)

        train_cost.append(np.mean(train_cost_batch ))
        val_cost.append(np.mean(val_cost_batch))
    return parameters, train_cost, val_cost



def train_model3(X_train,Y_train,X_val,Y_val,mini_batch_size,layers_dimension,learning_rate,max_epochs):
            #initialize parameters
    parameters = initialize_parameters(layers_dimension)

    train_cost= []
    val_cost  = []
    for i in range(max_epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]
        train_cost_batch = []
        val_cost_batch = []

        for i in range(0,len(X_train), mini_batch_size):
            X_batch = X_train_shuffled[i:i+mini_batch_size]
            Y_batch = Y_train_shuffled[i:i+mini_batch_size]
            
            output,cache  = forward_prop3(X_batch, parameters)
            A_val, _ = forward_prop3(X_val, parameters)
            t_loss = cost(output, Y_batch)
            v_loss = cost(A_val,Y_val)
            grads = backward_prop3(X_batch,Y_batch,cache, parameters)
            parameters = update_parameters(grads,parameters,learning_rate)

            val_cost_batch.append(v_loss)
            train_cost_batch.append(t_loss)
        train_cost.append(np.mean(train_cost_batch ))
        val_cost.append(np.mean(val_cost_batch))
    return parameters, train_cost, val_cost


#NN for classification
#2 layers and 3 layers
#Stochastic gradient descent with mini_batches, with and without momentum updates

import numpy as np


def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def relu(Z):
    return np.maximum(0,Z)  

def relu_derivative(x):
    return (x>0).astype(int)

def tanh(Z):
    return np.tanh(Z)

def tanh_derivative(z):
    return 1 - tanh(z)**2

def initialize_parameters(layer_dimensions):
    parameters = {}
    L = len(layer_dimensions)# No. of layers

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.normal(0, 0.001,size=(layer_dimensions[l], layer_dimensions[l-1]))
        parameters['b' + str(l)] = np.zeros((layer_dimensions[l],1))

    return parameters

def velocity_initialization(parameters):
    Vds = {key: 0 for key in parameters.keys()}
    return Vds

def forward_prop2(X, parameters):

    W1 = parameters['W1']
    W2 = parameters['W2']
    b1 = parameters['b1']
    b2 = parameters['b2']

    Z1 = np.dot(W1,X.T)+b1
    A1 = relu(Z1)
    Z2 = np.dot(W2,A1)+b2
    A2 = sigmoid(Z2)
    output = A2

    cache = {'Z1':Z1,'A1':A1, 'Z2':Z2, 'A2':A2}
    return output, cache


def forward_prop3(X, parameters):

    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    b1 = parameters['b1']
    b2 = parameters['b2']
    b3 = parameters['b3']

    Z1 = np.dot(W1,X)+b1
    A1 = relu(Z1)
    Z2 = np.dot(W2,A1)+b1
    A2 = relu(Z2)
    Z3 = np.dot(W3,A2)+b3
    A3 = sigmoid(Z3)
    output = A3

    cache = {'Z1':Z1,'A1':A1, 'Z2':Z2, 'A2':A2, 'Z3':Z3, 'A3': A3}
    return output, cache



def cost(A, Y):
    m = Y.shape[0] 
    return (-1/m)*np.sum(np.dot(np.log(A), Y) + np.dot(np.log(1-A),1-Y))


def backward_prop2(X,Y,cache, parameters):

    m= len(X)
   
    W2 = parameters['W2']

    Z1 = cache['Z1']
    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y.T
    dW2 = (1/m)*np.dot(dZ2,A1.T)
    db2 = (1/m)*np.sum(dZ2)
    dZ1 = np.dot(W2.T, dZ2)*relu_derivative(Z1)
    dW1 = (1/m)*np.dot(dZ1,X)
    db1 = (1/m)*np.sum(dZ1, axis=1, keepdims=True)

    return {'dW1': dW1,'db1':db1, 'dW2':dW2, 'db2':db2}

def backward_prop3(X,Y,cache, parameters):
    
    m= len(X)
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']
    A1 = cache['A1']
    A2 = cache['A2']
    A3 = cache['A3']

    # return {'dW1': dW1,'db1':db1, 'dW2':dW2, 'db2':db2, 'dW3':dW3, 'db3':db3}


def update_parameters(grads,parameters, learning_rate):
    for key in parameters.keys():
        parameters[key] -= learning_rate*grads['d'+key]
    return parameters

def momentum_parameters_update(parameters, gradients, learning_rate, momentum_rate,Vds):
        for key in parameters.keys():
                Vd = Vds[key]
                gradient = gradients['d'+key]
                Vds[key] = momentum_rate*Vd +(1-momentum_rate)*gradient
                parameters[key] -= learning_rate*Vds[key]
        return parameters, Vds

def make_predictions(X,Y,parameters,model):
    if model =='2':
        A,_ = forward_prop2(X,parameters)
        predict = (A > 0.5)
        accuracy = (np.sum(predict == Y.T))/len(Y)
    
    if model =='3':
        A = forward_prop3(X,parameters)
        predict = (A > 0.5) 
        accuracy = (np.sum(predict == Y.T))/len(Y)
    
    return predict, accuracy

def train_model2(X_train,Y_train,X_val,Y_val,mini_batch_size,layers_dimension,learning_rate,max_epochs):
    
    #initialize parameters
    parameters = initialize_parameters(layers_dimension)

    train_cost= []
    val_cost  = []
    for i in range(max_epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]
        train_cost_batch = []
        val_cost_batch = []

        for i in range(0,len(X_train), mini_batch_size):
            X_batch = X_train_shuffled[i:i+mini_batch_size]
            Y_batch = Y_train_shuffled[i:i+mini_batch_size]
            
            output,cache  = forward_prop2(X_batch, parameters)
            A_val, _ = forward_prop2(X_val, parameters)
            t_loss = cost(output, Y_batch)
            v_loss = cost(A_val,Y_val)
            grads = backward_prop2(X_batch,Y_batch,cache, parameters)
            parameters = update_parameters(grads,parameters,learning_rate)

            val_cost_batch.append(v_loss)
            train_cost_batch.append(t_loss)
  
        train_cost.append(np.mean(train_cost_batch ))
        val_cost.append(np.mean(val_cost_batch))
    return parameters, train_cost, val_cost

def train_modelwithmomentum(X_train,Y_train,X_val,Y_val,mini_batch_size,layers_dimension,learning_rate,max_epochs,momentum_rate):
        #initialize parameters
    parameters = initialize_parameters(layers_dimension)
    Vds = velocity_initialization(parameters)

    train_cost= []
    val_cost  = []
    for i in range(max_epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]
        train_cost_batch = []
        val_cost_batch = []

        for i in range(0,len(X_train), mini_batch_size):
            X_batch = X_train_shuffled[i:i+mini_batch_size]
            Y_batch = Y_train_shuffled[i:i+mini_batch_size]
            
            output,cache  = forward_prop2(X_batch, parameters)
            A_val, _ = forward_prop2(X_val, parameters)
            t_loss = cost(output, Y_batch)
            v_loss = cost(A_val,Y_val)
            grads = backward_prop2(X_batch,Y_batch,cache, parameters)
            parameters, Vds = momentum_parameters_update(parameters, grads, learning_rate, momentum_rate,Vds)
            
            val_cost_batch.append(v_loss)
            train_cost_batch.append(t_loss)

        train_cost.append(np.mean(train_cost_batch ))
        val_cost.append(np.mean(val_cost_batch))
    return parameters, train_cost, val_cost



def train_model3(X_train,Y_train,X_val,Y_val,mini_batch_size,layers_dimension,learning_rate,max_epochs):
            #initialize parameters
    parameters = initialize_parameters(layers_dimension)

    train_cost= []
    val_cost  = []
    for i in range(max_epochs):
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]
        train_cost_batch = []
        val_cost_batch = []

        for i in range(0,len(X_train), mini_batch_size):
            X_batch = X_train_shuffled[i:i+mini_batch_size]
            Y_batch = Y_train_shuffled[i:i+mini_batch_size]
            
            output,cache  = forward_prop3(X_batch, parameters)
            A_val, _ = forward_prop3(X_val, parameters)
            t_loss = cost(output, Y_batch)
            v_loss = cost(A_val,Y_val)
            grads = backward_prop3(X_batch,Y_batch,cache, parameters)
            parameters = update_parameters(grads,parameters,learning_rate)

            val_cost_batch.append(v_loss)
            train_cost_batch.append(t_loss)
        train_cost.append(np.mean(train_cost_batch ))
        val_cost.append(np.mean(val_cost_batch))
    return parameters, train_cost, val_cost


