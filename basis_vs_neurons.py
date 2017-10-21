"""
Created on Mon Oct  9 11:27:59 2017

@author: Sean
"""

import tensorflow as tf
import numpy as np
import math
import scipy as sp
from matplotlib import pyplot as plt
from tqdm import tqdm
import json

#######################################################
""" Config """
#######################################################
dim_num = 8      ## Max number of dimensional data to consider
data_num = 500    ## Number of data points to select from space
batch_size = 50  ## Mini batch-size
epochs = 1000     ## How many epochs to train
max_mod_par = 50 ## Maximum number of model parameters
lr = 1e-1         ## Learning rate
br_thresh = 800   ## Consecutive worse threshold for breaking training

########################################################
""" Define the n-dimensional function to approximate """
########################################################
def func_gauss(x):
    mag = np.sum(x**2, axis = 1)
    sig = 0.02
    return 20*np.exp(-mag/(2 * sig))

def func_weird(x):
    mag = np.sum(x**2, axis = 1)
    return 20*(np.sin(20 * mag) * sp.special.jv(2, 10 * mag))

def myfunc(x,a,b,n):
    terms = [a**m * sp.special.jv(m,b**m * 3.14 * x) for m in range(n)]
    return sum(terms)

def sintanh(x):
    mag = np.sum(x**2, axis = 1)
    return (np.sin(20*mag) + np.tanh(20*mag))
    
#######################################################
""" Define the data for multiple dimensions """
#######################################################
## n-th element is n-dimensional
x_ = [np.random.uniform(0.,1.,size=[data_num , n + 1]) for n in range(dim_num)]
y_gauss = [func_gauss(x) for x in x_]
y_weird = [func_weird(x) for x in x_]
y_myfunc = [np.sum(myfunc(x,2.,1.5,4),axis=1) for x in x_]
y_myfunc2 = [np.sum(myfunc(x,1.,1.1,30),axis=1) for x in x_]
y_sintanh = [sintanh(x) for x in x_]


#######################################################
""" Define the Legendre expansion """
#######################################################
def leg_basis(x, order):
    terms = np.array([sp.special.legendre(m)(x) for m in range(order)])
    return terms

def tensor_product(terms):
    ## Create an initial basis set
    ret_set = [1.]
    shape = terms.shape
    ## Iteratively remake the basis set
    for i in range(shape[1]):
        foo = []
        [[foo.append(elem * ent) for ent in terms[:,i]] for elem in ret_set]
        ret_set = foo
    return np.array(ret_set)

def get_tensor_basis(x, order):
    batch = []
    for point in x:
        terms = leg_basis(point, order)
        basis = tensor_product(terms)
        batch.append(basis)
    return np.array(batch)

def leg_expansion(basis,order,dim):
    N = tf.Variable(tf.random_uniform(shape = [order**dim]))
    return tf.einsum('i,ji->j', N, basis)

def leg_loss(expansion,y):
    return tf.reduce_sum(tf.square(expansion - y))

def leg_train(loss):
    return tf.train.AdamOptimizer(lr).minimize(loss)


#######################################################
""" Define the neural net """
#######################################################
def neural_net(x,neuron_no,dim):
    W = tf.Variable(tf.random_uniform(minval = -1., maxval = 1., shape = [neuron_no, dim]))
    b = tf.Variable(tf.random_uniform(minval = -1., maxval = 1., shape = [neuron_no]))
    N = tf.Variable(tf.random_uniform(shape = [neuron_no]))
    
    return tf.einsum('i,ji->j', N, tf.tanh(tf.einsum('ij,kj->ki', W, x_place) + b))

def neural_loss(net,y):
    return tf.reduce_sum(tf.square(net - y))

def neural_train(loss):
    return tf.train.AdamOptimizer(lr).minimize(loss)


#######################################################
""" Determine max order and neuron_no"""
#######################################################
## Return the biggest integer less than max_params
def max_order(max_params, dim):
    ## Gotta be lower than the max_params, hence floor
    return math.floor(max_params**(1/dim))

## Set the max number of neurons by the max order
def max_neuron_no(max_params, dim):
    ## Let's even give neural nets the advantage - more neurons! Hence ceil
    return math.ceil(max_params/(2+dim))

#######################################################
""" Let's get to training! """
#######################################################
## Track the loss by dimension
leg_gauss_loss = []
leg_weird_loss = []
leg_myfunc_loss = []
leg_myfunc2_loss = []
leg_sintanh_loss = []
neural_gauss_loss = []
neural_weird_loss = []
neural_myfunc_loss = []
neural_myfunc2_loss = []
neural_sintanh_loss = []

## Track the predictions by dimension
leg_gauss_pred = []
leg_weird_pred = []
leg_myfunc_pred = []
leg_myfunc2_pred = []
leg_sintanh_pred = []
neural_gauss_pred = []
neural_weird_pred = []
neural_myfunc_pred = []
neural_myfunc2_pred = []
neural_sintanh_pred = []

## Let's find the prediction and loss as a function of feature dimension
for i in range(dim_num):
    dim = i + 1
    tf.reset_default_graph()
    
    ## Data size and number of batches
    data_size = len(x_[i])
    batch_tot = math.ceil(data_size/batch_size)

    ## Specify number of bases and neurons to use
    order = max_order(max_mod_par, dim)
    neuron_no = max_neuron_no(max_mod_par, dim)
    print("Order:", order)
    print("Neuron No.:", neuron_no)
    
    ## Define TensorFlow placeholders for the data
    x_place = tf.placeholder(dtype = tf.float32, shape = [None, dim])
    basis_place = tf.placeholder(dtype = tf.float32, shape = [None, order**dim])
    y_place = tf.placeholder(dtype = tf.float32, shape = [None])
    
    ## Go ahead and get the tensor basis for the Legendre expansion
    basis = get_tensor_basis(x_[i], order)
    
    ## Relevant ops for Legendre expansion
    with tf.name_scope('legendre'):
        leg_pred = leg_expansion(basis=basis_place, order=order, dim=dim)
        leg_loss_op = leg_loss(leg_pred, y_place)
        leg_train_op = leg_train(leg_loss_op)
    
    ## Relevant ops for neural net
    with tf.name_scope('neural_net'):
        neural_pred = neural_net(x=x_place, neuron_no=neuron_no, dim=dim)
        neural_loss_op = neural_loss(neural_pred, y_place)
        neural_train_op = neural_train(neural_loss_op)
    
    
    #########################################################
    ## Gauss
    #########################################################
    print("\nTraining Gauss function in %d dimensions." % (dim))
    ## Keep track of the losses
    leg_best_loss = float('inf')
    leg_old_loss = leg_best_loss
    neural_best_loss = float('inf')
    neural_old_loss = neural_best_loss
    
    ## Create an early stopping counter
    leg_consec_worse = 0
    neural_consec_worse = 0
    
    ## Create things in the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ## Keep track of the best variables
        leg_best_var = [sess.run(elem) for elem in
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'legendre')]
        neural_best_var = [sess.run(elem) for elem in
                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'neural_net')]
        ##Prepare to save the variables
        saver = tf.train.Saver()
        
        ## Let's begin
        for epoch in tqdm(range(epochs)):
            leg_new_loss = 0
            neural_new_loss = 0
            
            for batch in range(batch_tot):
                ## Get the batch data
                batch_basis = basis[batch*batch_size:(batch+1)*batch_size]
                batch_x = x_[i][batch*batch_size:(batch+1)*batch_size]
                batch_y = y_gauss[i][batch*batch_size:(batch+1)*batch_size]
                
                ## Train stuff
                sess.run([leg_train_op, neural_train_op],
                         feed_dict={basis_place : batch_basis,
                                    x_place : batch_x,
                                    y_place : batch_y})
                ## Get the losses
                temp_leg_new_loss, temp_neural_new_loss = sess.run([leg_loss_op, neural_loss_op],
                                                                     feed_dict={basis_place : batch_basis,
                                                                                x_place : batch_x,
                                                                                y_place : batch_y})
                leg_new_loss += temp_leg_new_loss
                neural_new_loss += temp_neural_new_loss
            
            leg_new_loss /= data_size
            neural_new_loss /= data_size
            
            ## Check for loss improvement
            if leg_new_loss < leg_best_loss:
                leg_best_var = [sess.run(elem) for elem in
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'legendre')]
                leg_best_loss = leg_new_loss
                leg_consec_worse = 0
            
            if neural_new_loss < neural_best_loss:
                neural_best_var = [sess.run(elem) for elem in
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'neural_net')]
                neural_best_loss = neural_new_loss
                neural_consec_worse = 0
                
            if leg_new_loss > leg_old_loss:
                leg_consec_worse += 1
            
            if neural_new_loss > neural_old_loss:
                neural_consec_worse += 1
                
            if (leg_consec_worse >= br_thresh) or (neural_consec_worse >= br_thresh):
                break
            
            leg_old_loss = leg_new_loss
            neural_old_loss = leg_new_loss
        
        ## Ensure the graph is using the best model parameters
        [sess.run(tf.assign(elem,leg_best_var[i])) for i,elem in
                            enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'legendre'))]
        [sess.run(tf.assign(elem,neural_best_var[i])) for i,elem in
                            enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'neural_net'))]
        ## Save the parameters
        saver.save(sess, './variables/gauss' + str(i+1) + 'd.ckpt')
        
        ## Add the best loss to the trackers
        leg_gauss_loss.append(leg_best_loss)
        neural_gauss_loss.append(neural_best_loss)
        print("Best Legendre loss:", leg_best_loss)
        print("Best neural network loss:", neural_best_loss)
        
        ## Add the best prediction to the trackers
        leg_gauss_pred.append(sess.run(leg_pred,
                                       feed_dict = {basis_place : basis}))
        neural_gauss_pred.append(sess.run(neural_pred,
                                          feed_dict = {x_place : x_[i]}))
        
        
    #########################################################
    ## Weird
    #########################################################
    print("\nTraining weird function in %d dimensions." % (dim))
    ## Keep track of the losses
    leg_best_loss = float('inf')
    leg_old_loss = leg_best_loss
    neural_best_loss = float('inf')
    neural_old_loss = neural_best_loss
    
    ## Create an early stopping counter
    leg_consec_worse = 0
    neural_consec_worse = 0
    
    ## Create things in the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ## Keep track of the best variables
        leg_best_var = [sess.run(elem) for elem in
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'legendre')]
        neural_best_var = [sess.run(elem) for elem in
                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'neural_net')]
        ##Prepare to save the variables
        saver = tf.train.Saver()
        
        ## Let's begin
        for epoch in tqdm(range(epochs)):
            leg_new_loss = 0
            neural_new_loss = 0
            
            for batch in range(batch_tot):
                ## Get the batch data
                batch_basis = basis[batch*batch_size:(batch+1)*batch_size]
                batch_x = x_[i][batch*batch_size:(batch+1)*batch_size]
                batch_y = y_weird[i][batch*batch_size:(batch+1)*batch_size]
                
                ## Train stuff
                sess.run([leg_train_op, neural_train_op],
                         feed_dict={basis_place : batch_basis,
                                    x_place : batch_x,
                                    y_place : batch_y})
                ## Get the losses
                temp_leg_new_loss, temp_neural_new_loss = sess.run([leg_loss_op, neural_loss_op],
                                                                     feed_dict={basis_place : batch_basis,
                                                                                x_place : batch_x,
                                                                                y_place : batch_y})
                leg_new_loss += temp_leg_new_loss
                neural_new_loss += temp_neural_new_loss
            
            leg_new_loss /= data_size
            neural_new_loss /= data_size
            
            ## Check for loss improvement
            if leg_new_loss < leg_best_loss:
                leg_best_var = [sess.run(elem) for elem in
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'legendre')]
                leg_best_loss = leg_new_loss
                leg_consec_worse = 0
            
            if neural_new_loss < neural_best_loss:
                neural_best_var = [sess.run(elem) for elem in
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'neural_net')]
                neural_best_loss = neural_new_loss
                neural_consec_worse = 0
                
            if leg_new_loss > leg_old_loss:
                leg_consec_worse += 1
            
            if neural_new_loss > neural_old_loss:
                neural_consec_worse += 1
                
            if (leg_consec_worse >= br_thresh) or (neural_consec_worse >= br_thresh):
                break
            
            leg_old_loss = leg_new_loss
            neural_old_loss = leg_new_loss
        
        ## Ensure the graph is using the best model parameters
        [sess.run(tf.assign(elem,leg_best_var[i])) for i,elem in
                            enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'legendre'))]
        [sess.run(tf.assign(elem,neural_best_var[i])) for i,elem in
                            enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'neural_net'))]
        ## Save the parameters
        saver.save(sess, './variables/weird' + str(i+1) + 'd.ckpt')
        
        ## Add the best loss to the trackers
        leg_weird_loss.append(leg_best_loss)
        neural_weird_loss.append(neural_best_loss)
        print("Best Legendre loss:", leg_best_loss)
        print("Best neural network loss:", neural_best_loss)
        
        ## Add the best prediction to the trackers
        leg_weird_pred.append(sess.run(leg_pred,
                                       feed_dict = {basis_place : basis}))
        neural_weird_pred.append(sess.run(neural_pred,
                                          feed_dict = {x_place : x_[i]}))
        
        
    #########################################################
    ## MyFunc
    #########################################################
    print("\nTraining MyFunc function in %d dimensions." % (dim))
    ## Keep track of the losses
    leg_best_loss = float('inf')
    leg_old_loss = leg_best_loss
    neural_best_loss = float('inf')
    neural_old_loss = neural_best_loss
    
    ## Create an early stopping counter
    leg_consec_worse = 0
    neural_consec_worse = 0
    
    ## Create things in the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ## Keep track of the best variables
        leg_best_var = [sess.run(elem) for elem in
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'legendre')]
        neural_best_var = [sess.run(elem) for elem in
                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'neural_net')]
        ##Prepare to save the variables
        saver = tf.train.Saver()
        
        ## Let's begin
        for epoch in tqdm(range(epochs)):
            leg_new_loss = 0
            neural_new_loss = 0
            
            for batch in range(batch_tot):
                ## Get the batch data
                batch_basis = basis[batch*batch_size:(batch+1)*batch_size]
                batch_x = x_[i][batch*batch_size:(batch+1)*batch_size]
                batch_y = y_myfunc[i][batch*batch_size:(batch+1)*batch_size]
                
                ## Train stuff
                sess.run([leg_train_op, neural_train_op],
                         feed_dict={basis_place : batch_basis,
                                    x_place : batch_x,
                                    y_place : batch_y})
                ## Get the losses
                temp_leg_new_loss, temp_neural_new_loss = sess.run([leg_loss_op, neural_loss_op],
                                                                     feed_dict={basis_place : batch_basis,
                                                                                x_place : batch_x,
                                                                                y_place : batch_y})
                leg_new_loss += temp_leg_new_loss
                neural_new_loss += temp_neural_new_loss
            
            leg_new_loss /= data_size
            neural_new_loss /= data_size
            
            ## Check for loss improvement
            if leg_new_loss < leg_best_loss:
                leg_best_var = [sess.run(elem) for elem in
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'legendre')]
                leg_best_loss = leg_new_loss
                leg_consec_worse = 0
            
            if neural_new_loss < neural_best_loss:
                neural_best_var = [sess.run(elem) for elem in
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'neural_net')]
                neural_best_loss = neural_new_loss
                neural_consec_worse = 0
                
            if leg_new_loss > leg_old_loss:
                leg_consec_worse += 1
            
            if neural_new_loss > neural_old_loss:
                neural_consec_worse += 1
                
            if (leg_consec_worse >= br_thresh) or (neural_consec_worse >= br_thresh):
                break
            
            leg_old_loss = leg_new_loss
            neural_old_loss = leg_new_loss
        
        ## Ensure the graph is using the best model parameters
        [sess.run(tf.assign(elem,leg_best_var[i])) for i,elem in
                            enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'legendre'))]
        [sess.run(tf.assign(elem,neural_best_var[i])) for i,elem in
                            enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'neural_net'))]
        ## Save the parameters
        saver.save(sess, './variables/myfunc' + str(i+1) + 'd.ckpt')
        
        ## Add the best loss to the trackers
        leg_myfunc_loss.append(leg_best_loss)
        neural_myfunc_loss.append(neural_best_loss)
        print("Best Legendre loss:", leg_best_loss)
        print("Best neural network loss:", neural_best_loss)
        
        ## Add the best prediction to the trackers
        leg_myfunc_pred.append(sess.run(leg_pred,
                                       feed_dict = {basis_place : basis}))
        neural_myfunc_pred.append(sess.run(neural_pred,
                                          feed_dict = {x_place : x_[i]}))
    
    #########################################################
    ## MyFunc2
    #########################################################
    print("\nTraining MyFunc2 function in %d dimensions." % (dim))
    ## Keep track of the losses
    leg_best_loss = float('inf')
    leg_old_loss = leg_best_loss
    neural_best_loss = float('inf')
    neural_old_loss = neural_best_loss
    
    ## Create an early stopping counter
    leg_consec_worse = 0
    neural_consec_worse = 0
    
    ## Create things in the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ## Keep track of the best variables
        leg_best_var = [sess.run(elem) for elem in
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'legendre')]
        neural_best_var = [sess.run(elem) for elem in
                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'neural_net')]
        ##Prepare to save the variables
        saver = tf.train.Saver()
        
        ## Let's begin
        for epoch in tqdm(range(epochs)):
            leg_new_loss = 0
            neural_new_loss = 0
            
            for batch in range(batch_tot):
                ## Get the batch data
                batch_basis = basis[batch*batch_size:(batch+1)*batch_size]
                batch_x = x_[i][batch*batch_size:(batch+1)*batch_size]
                batch_y = y_myfunc2[i][batch*batch_size:(batch+1)*batch_size]
                
                ## Train stuff
                sess.run([leg_train_op, neural_train_op],
                         feed_dict={basis_place : batch_basis,
                                    x_place : batch_x,
                                    y_place : batch_y})
                ## Get the losses
                temp_leg_new_loss, temp_neural_new_loss = sess.run([leg_loss_op, neural_loss_op],
                                                                     feed_dict={basis_place : batch_basis,
                                                                                x_place : batch_x,
                                                                                y_place : batch_y})
                leg_new_loss += temp_leg_new_loss
                neural_new_loss += temp_neural_new_loss
            
            leg_new_loss /= data_size
            neural_new_loss /= data_size
            
            ## Check for loss improvement
            if leg_new_loss < leg_best_loss:
                leg_best_var = [sess.run(elem) for elem in
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'legendre')]
                leg_best_loss = leg_new_loss
                leg_consec_worse = 0
            
            if neural_new_loss < neural_best_loss:
                neural_best_var = [sess.run(elem) for elem in
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'neural_net')]
                neural_best_loss = neural_new_loss
                neural_consec_worse = 0
                
            if leg_new_loss > leg_old_loss:
                leg_consec_worse += 1
            
            if neural_new_loss > neural_old_loss:
                neural_consec_worse += 1
                
            if (leg_consec_worse >= br_thresh) or (neural_consec_worse >= br_thresh):
                break
            
            leg_old_loss = leg_new_loss
            neural_old_loss = leg_new_loss
        
        ## Ensure the graph is using the best model parameters
        [sess.run(tf.assign(elem,leg_best_var[i])) for i,elem in
                            enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'legendre'))]
        [sess.run(tf.assign(elem,neural_best_var[i])) for i,elem in
                            enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'neural_net'))]
        ## Save the parameters
        saver.save(sess, './variables/myfunc2' + str(i+1) + 'd.ckpt')
        
        ## Add the best loss to the trackers
        leg_myfunc2_loss.append(leg_best_loss)
        neural_myfunc2_loss.append(neural_best_loss)
        print("Best Legendre loss:", leg_best_loss)
        print("Best neural network loss:", neural_best_loss)
        
        ## Add the best prediction to the trackers
        leg_myfunc2_pred.append(sess.run(leg_pred,
                                       feed_dict = {basis_place : basis}))
        neural_myfunc2_pred.append(sess.run(neural_pred,
                                          feed_dict = {x_place : x_[i]}))
        
    #########################################################
    ## Sin + Tanh
    #########################################################
    print("\nTraining Sin+Tanh function in %d dimensions." % (dim))
    ## Keep track of the losses
    leg_best_loss = float('inf')
    leg_old_loss = leg_best_loss
    neural_best_loss = float('inf')
    neural_old_loss = neural_best_loss
    
    ## Create an early stopping counter
    leg_consec_worse = 0
    neural_consec_worse = 0
    
    ## Create things in the session
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ## Keep track of the best variables
        leg_best_var = [sess.run(elem) for elem in
                        tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'legendre')]
        neural_best_var = [sess.run(elem) for elem in
                           tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'neural_net')]
        ##Prepare to save the variables
        saver = tf.train.Saver()
        
        ## Let's begin
        for epoch in tqdm(range(epochs)):
            leg_new_loss = 0
            neural_new_loss = 0
            
            for batch in range(batch_tot):
                ## Get the batch data
                batch_basis = basis[batch*batch_size:(batch+1)*batch_size]
                batch_x = x_[i][batch*batch_size:(batch+1)*batch_size]
                batch_y = y_sintanh[i][batch*batch_size:(batch+1)*batch_size]
                
                ## Train stuff
                sess.run([leg_train_op, neural_train_op],
                         feed_dict={basis_place : batch_basis,
                                    x_place : batch_x,
                                    y_place : batch_y})
                ## Get the losses
                temp_leg_new_loss, temp_neural_new_loss = sess.run([leg_loss_op, neural_loss_op],
                                                                     feed_dict={basis_place : batch_basis,
                                                                                x_place : batch_x,
                                                                                y_place : batch_y})
                leg_new_loss += temp_leg_new_loss
                neural_new_loss += temp_neural_new_loss
            
            leg_new_loss /= data_size
            neural_new_loss /= data_size
            
            ## Check for loss improvement
            if leg_new_loss < leg_best_loss:
                leg_best_var = [sess.run(elem) for elem in
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'legendre')]
                leg_best_loss = leg_new_loss
                leg_consec_worse = 0
            
            if neural_new_loss < neural_best_loss:
                neural_best_var = [sess.run(elem) for elem in
                                tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'neural_net')]
                neural_best_loss = neural_new_loss
                neural_consec_worse = 0
                
            if leg_new_loss > leg_old_loss:
                leg_consec_worse += 1
            
            if neural_new_loss > neural_old_loss:
                neural_consec_worse += 1
                
            if (leg_consec_worse >= br_thresh) or (neural_consec_worse >= br_thresh):
                break
            
            leg_old_loss = leg_new_loss
            neural_old_loss = leg_new_loss
        
        ## Ensure the graph is using the best model parameters
        [sess.run(tf.assign(elem,leg_best_var[i])) for i,elem in
                            enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'legendre'))]
        [sess.run(tf.assign(elem,neural_best_var[i])) for i,elem in
                            enumerate(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope = 'neural_net'))]
        ## Save the parameters
        saver.save(sess, './variables/sintanh' + str(i+1) + 'd.ckpt')
        
        ## Add the best loss to the trackers
        leg_sintanh_loss.append(leg_best_loss)
        neural_sintanh_loss.append(neural_best_loss)
        print("Best Legendre loss:", leg_best_loss)
        print("Best neural network loss:", neural_best_loss)
        
        ## Add the best prediction to the trackers
        leg_sintanh_pred.append(sess.run(leg_pred,
                                       feed_dict = {basis_place : basis}))
        neural_sintanh_pred.append(sess.run(neural_pred,
                                          feed_dict = {x_place : x_[i]}))
        
#######################################################
""" Let's plot some stuff! """
#######################################################
dim_set = [i+1 for i in range(dim_num)]
print("Plotting gaussian loss.")
plt.plot(dim_set,leg_gauss_loss,'b',
         dim_set,neural_gauss_loss,'r')
print("Plotting gaussian prediction")
for i in range(dim_num):
    print("Dimension:", i+1)
    x_set = np.sqrt(np.sum(x_[i]**2,axis=1))
    order = np.argsort(x_set)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    ax1.plot(x_set[order], y_gauss[i][order], c='k', label='Real')
    ax1.plot(x_set[order], leg_gauss_pred[i][order], c='b', label='Legendre')
    ax1.plot(x_set[order], neural_gauss_pred[i][order], c='r', label='Neural Network')
    plt.legend(loc='upper left');
    plt.show()
    
print("Plotting weird function loss.")
plt.plot(dim_set,leg_weird_loss,'b',
         dim_set,neural_weird_loss,'r')
print("Plotting weird prediction")
for i in range(dim_num):
    print("Dimension:", i+1)
    x_set = np.sqrt(np.sum(x_[i]**2,axis=1))
    order = np.argsort(x_set)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    ax1.plot(x_set[order], y_weird[i][order], c='k', label='Real')
    ax1.plot(x_set[order], leg_weird_pred[i][order], c='b', label='Legendre')
    ax1.plot(x_set[order], neural_weird_pred[i][order], c='r', label='Neural Network')
    plt.legend(loc='upper left');
    plt.show()
    
print("Plotting myfunc function loss.")
plt.plot(dim_set,leg_myfunc_loss,'b',
         dim_set,neural_myfunc_loss,'r')
print("Plotting myfunc prediction")
for i in range(dim_num):
    print("Dimension:", i+1)
    x_set = np.sqrt(np.sum(x_[i]**2,axis=1))
    order = np.argsort(x_set)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    ax1.plot(x_set[order], y_myfunc[i][order], c='k', label='Real')
    ax1.plot(x_set[order], leg_myfunc_pred[i][order], c='b', label='Legendre')
    ax1.plot(x_set[order], neural_myfunc_pred[i][order], c='r', label='Neural Network')
    plt.legend(loc='upper left');
    plt.show()

print("Plotting myfunc2 function loss.")
plt.plot(dim_set,leg_myfunc2_loss,'b',
         dim_set,neural_myfunc2_loss,'r')
plt.plot(dim_set[0:4],leg_myfunc2_loss[0:4],'b',
         dim_set[0:4],neural_myfunc2_loss[0:4],'r')
print("Plotting myfunc2 prediction")
for i in range(dim_num):
    print("Dimension:", i+1)
    x_set = np.sqrt(np.sum(x_[i]**2,axis=1))
    order = np.argsort(x_set)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    ax1.plot(x_set[order], y_myfunc2[i][order], c='k', label='Real')
    ax1.plot(x_set[order], leg_myfunc2_pred[i][order], c='b', label='Legendre')
    ax1.plot(x_set[order], neural_myfunc2_pred[i][order], c='r', label='Neural Network')
    plt.legend(loc='upper left');
    plt.show()

print("Plotting sintanh function loss.")
plt.plot(dim_set,leg_sintanh_loss,'b',
         dim_set,neural_sintanh_loss,'r')
print("Plotting sintanh prediction")
for i in range(dim_num):
    print("Dimension:", i+1)
    x_set = np.sqrt(np.sum(x_[i]**2,axis=1))
    order = np.argsort(x_set)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    
    ax1.plot(x_set[order], y_sintanh[i][order], c='k', label='Real')
    ax1.plot(x_set[order], leg_sintanh_pred[i][order], c='b', label='Legendre')
    ax1.plot(x_set[order], neural_sintanh_pred[i][order], c='r', label='Neural Network')
    plt.legend(loc='upper left');
    plt.show()