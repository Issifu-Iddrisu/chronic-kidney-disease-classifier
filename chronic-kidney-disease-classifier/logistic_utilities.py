#Logistic utilities
#Logistic regression for classification
#Gradient descent with batch and stochastic mini batches
import numpy as np
# import matplotlib.pyplot as plt

def initialize_parameters(num_features):
    W = np.random.randn(1,num_features)*0.01
    b = 0
    parameters = {'W':W, 'b':b}
    return parameters


def sigmoid(z):
    return 1/(1+np.exp(-z))



def cost(A, Y):
    m = Y.shape[0] 
    return (-1/m)*np.sum(np.dot(np.log(A), Y) + np.dot(np.log(1-A), 1-Y))

def output(X, parameters):
    W = parameters['W']
    b = parameters['b']
    Z = np.dot(W,X.T) + b
    return sigmoid(Z)

def gradients(X,Y,A):
    m = Y.shape[0] 
    D = A-Y.T
    dW = (1/m)*np.dot(D,X)
    db = (1/m)*np.sum(D)
    grads = {'dW':dW, 'db':db}
    return grads


def update_parameters(grads, parameters, learning_rate):
    W  = parameters['W']
    b  = parameters['b']
    dW = grads['dW']
    db  = grads['db']
    W = W - learning_rate*dW
    b = b - learning_rate*db
    parameters = {'W':W, 'b':b}
    return parameters



def ridge_cost(A,Y,parameters,lambda_parameter):
    m = Y.shape[0] 
    return (-1/m)*(np.sum(np.dot(np.log(A), Y) + np.dot(np.log(1-A), 1-Y)) - (lambda_parameter*np.sum(parameters['W']**2)))

def ridge_gradients(X,Y,A,parameters, lambda_parameter):
    m = Y.shape[0] 
    D = A-Y.T
    dW = (1/m)*(np.dot(D,X) + (lambda_parameter*parameters['W']))
    db = (1/m)*np.sum(D)
    grads = {'dW':dW, 'db':db}
    return grads


def lg_batch_model(X_train,Y_train,X_val,Y_val, learning_rate, epochs):
 
    #initialize parameters
    features_num= X_train.shape[1]
    parameters = initialize_parameters(features_num)

    train_cost= list()
    val_cost  = list()
    for i in range(0, epochs):
        A = output(X_train, parameters)
        A_val = output(X_val, parameters)
        t_loss = cost(A, Y_train)
        v_loss = cost(A_val,Y_val)
        grads = gradients(X_train,Y_train,A)
        parameters = update_parameters(grads, parameters, learning_rate)

        #Append losses for each epoch
        train_cost.append(t_loss)
        val_cost.append(v_loss)
    return {'parameters':parameters, 'train_cost':train_cost, 'val_cost':val_cost}

def ridge_batch_model(X_train,Y_train,X_val,Y_val, learning_rate, epochs,lambda_parameter):

     
    #initialize parameters
    features_num= X_train.shape[1]
    parameters = initialize_parameters(features_num)

    train_cost= list()
    val_cost  = list()
    for i in range(epochs):
        A = output(X_train, parameters)
        A_val = output(X_val, parameters)
        t_loss = ridge_cost(A, Y_train,parameters,lambda_parameter)
        v_loss = ridge_cost(A_val,Y_val,parameters,lambda_parameter)
        grads = ridge_gradients(X_train,Y_train,A,parameters,lambda_parameter)
        parameters = update_parameters(grads, parameters, learning_rate)

        #Append losses for each epoch
        train_cost.append(t_loss)
        val_cost.append(v_loss)
    return {'parameters':parameters, 'train_cost':train_cost, 'val_cost':val_cost}
 
def predictions(X,Y,parameters):
    A = output(X,parameters)
    predict = (A > 0.5).astype(int) 
    accuracy = (np.sum(predict == Y.T))/len(Y)
    return predict, accuracy
 

def lg_mini_model(X_train,Y_train,X_val,Y_val,mini_batch_size,learning_rate,max_epochs):
        #initialize parameters
    features_num= X_train.shape[1]
    parameters = initialize_parameters(features_num)

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
            
            A = output(X_batch, parameters)
            A_val = output(X_val, parameters)
            t_loss = cost(A, Y_batch)
            v_loss = cost(A_val,Y_val)
            grads = gradients(X_batch,Y_batch,A)
            parameters = update_parameters(grads,parameters,learning_rate)

            val_cost_batch.append(v_loss)
            train_cost_batch.append(t_loss)
        train_cost.append(np.mean(train_cost_batch ))
        val_cost.append(np.mean(val_cost_batch))
    return parameters, train_cost, val_cost

