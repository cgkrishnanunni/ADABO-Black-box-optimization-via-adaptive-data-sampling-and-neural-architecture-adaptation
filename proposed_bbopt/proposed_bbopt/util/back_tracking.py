"""
Backtracking line search when accepting a newly inserted residual layer.

Increases ``hyperp.epsilon`` until the training loss does not worsen
unacceptably, then returns the accepted ``weights`` / ``biases``.
"""

import numpy as np
from proposed_bbopt.util.back_prop import forward_prop
from proposed_bbopt.util.activation_prop import activation
from proposed_bbopt.util.activation_prop import activation_input
from proposed_bbopt.util.activation_prop import activation_input_derivative
from proposed_bbopt.util.activation_prop import activation_output
from proposed_bbopt.util.activation_prop import activation_output_derivative
from proposed_bbopt.util.activation_prop import activation_derivative
from proposed_bbopt.util.random_initialization import get_weights, get_new_weights


def back_tracking(hyperp, Nt, X, weights, biases, index, eigen_vectors, O, Y, weights_new, biases_new):
    """Tune insertion step size ``epsilon`` and return updated network params."""
    hyperp.epsilon = 0
    
    X_state, output_old=forward_prop(hyperp, Nt, O, X, activation_input, activation, activation_output,weights_new, biases_new)
    ep=hyperp.epsilon
    q_initial=0
    for samples in range(0,Nt):

        q_initial=q_initial+np.square(np.linalg.norm((Y[samples]-output_old[:,samples])))/(Nt*O)

    q_tu=q_initial
    for back_track in range(0,hyperp.back_track):
        
        q=0
        hyperp.epsilon=hyperp.epsilon+hyperp.back_track_rate

        hyperp.L=hyperp.L-1

        weights_new,biases_new=get_new_weights(hyperp,weights,biases,index,eigen_vectors)
    
        hyperp.L=hyperp.L+1


        X_state, output=forward_prop(hyperp, Nt, O, X, activation_input, activation, activation_output,weights_new, biases_new)

        for samples in range(0,Nt):

            q=q+np.square(np.linalg.norm((Y[samples]-output[:,samples])))/(Nt*O)

        if back_track==0:
            qw=2*(q_tu-q)/(hyperp.back_track_rate*hyperp.back_track_rate)
        if q>q_initial:
            break

        q_initial=q
        
    hyperp.epsilon=ep
    return weights_new,biases_new, q_tu-q, back_track,qw