"""
Simple implementation of AuGMEnT as described in:

Rombouts JO, Bohte SM, Roelfsema PR (2015)
How Attention Can Create Synaptic Tags for the Learning of Working Memories in Sequential Tasks.
PLoS Comput Biol 11: e1004060.

@author: J.O. Rombouts
"""

import numpy as np
from numpy.random import rand
import cPickle as pickle
import logging


class Sigmoid(object):
    """
    Simple container class for sigmoid transformation.
    """

    def __init__(self, theta=2.5):
        self.theta = theta

    def transform(self, np_arr):
        return 1. / (1. + np.exp(self.theta - np_arr))

    @staticmethod
    def derivative(np_arr):
        return np_arr * (1. - np_arr)


class AugmentNetwork(object):
    """
    Implementation of AuGMEnT network
    """
    
    def __init__(self, **kwargs):
        """
        Constructor. See comments below for explanation of kwargs.
        """

        # Network meta-parameters
        self.beta = kwargs.get('beta',  0.25)  # Learning rate
        self.gamma = kwargs.get('gamma', 0.90)  # Future discounting
        
        # Parameters Tags
        self.L = kwargs.get('L', 0.30)  # lambda decay
        self.mem_decays = 1.0
        
        # Parameters controller:
        self.epsilon = kwargs.get('epsilon', 0.025)  # Exploration rate
        self.explore = self.select_boltzmann  # Exploration fn. to use
        self.explore_nm = 'max-boltzmann'
        self.prec_q_acts = kwargs.get('prec_q_acts', 5)  # Cutoff of q-values

        # Architecture Parameters 
        self.nx_inst = kwargs.get('n_inputs', 4)  # Number input neurons (Instantaneous)
        self.nx_trans = self.nx_inst * 2  # Number input neurons (On, Off)
    
        self.ny_regular = kwargs.get('ny_regular', 3)  # Number of regular hidden units
        self.ny_memory = kwargs.get('ny_memory', 4)  # Number of memory hidden units
        self.nz = kwargs.get('nz', 3)  # Number of output units
        
        self.weight_range = kwargs.get('weight_range', 0.25)  # Centered range of weights

        self.theta = kwargs.get('theta', 2.5)  # Sigmoid shift
        self.mem_transform = Sigmoid(theta=self.theta)
        self.reg_transform = Sigmoid(theta=self.theta)
        
        # Synaptic weights:
        # Bias units (set to 0 to switch off)
        self.bias_input = kwargs.get('bias_input', 1)
        self.bias_hidden = kwargs.get('bias_hidden', 1)
        self.bias_mem_hidden = kwargs.get('bias_mem_hidden', 0)

        # Input to hidden weights (regular)
        self.weights_xy_reg = self.generate_weights(self.nx_inst + self.bias_input, self.ny_regular)
        # Input to hidden weights (memory)
        self.weights_xy_mem = self.generate_weights(self.nx_trans, self.ny_memory)
        
        # NOTE: Both hidden layers can have their own bias unit, but it is 
        # most straightforward to use only one (default).
        # hidden to output weights (regular)
        self.weights_yz_reg = self.generate_weights(self.ny_regular + self.bias_hidden, self.nz)
        # hidden to output weights (memory)
        self.weights_yz_mem = self.generate_weights(self.ny_memory + self.bias_mem_hidden, self.nz)
        
        # Dynamic Network Parameters
        self._initialize_dynamic_variables()

        logging.getLogger(__name__).debug("Initialized Network.")
    
    def _initialize_dynamic_variables(self):
        """
        Sets up all dynamic variables
        """
        # Layer activations:
        self.x_reg = np.zeros(self.nx_inst)
        self.x_trans = np.zeros(self.nx_trans)
        self.y_reg = np.zeros(self.ny_regular)
        self.y_mem = np.zeros(self.ny_memory)
        
        self.z = np.zeros(self.nz)
        
        # Memory unit activations
        self.y_mem_tot = np.zeros(self.ny_memory)
        
        # Input to hidden traces:
        self.xy_reg_traces = np.zeros_like(self.weights_xy_reg)
        self.xy_mem_traces = np.zeros_like(self.weights_xy_mem)
        self.yz_reg_traces = np.zeros_like(self.weights_yz_reg)
        self.yz_mem_traces = np.zeros_like(self.weights_yz_mem)
        
        # Tags (~ eligibility traces)
        self.xy_reg_tags = np.zeros_like(self.weights_xy_reg)
        self.xy_mem_tags = np.zeros_like(self.weights_xy_mem)
        self.yz_reg_tags = np.zeros_like(self.weights_yz_reg)
        self.yz_mem_tags = np.zeros_like(self.weights_yz_mem)        

        self.prev_obs = np.zeros(self.nx_inst)
        self.z_prime = np.zeros(self.nz)
        self.prev_action = -1
        self.prev_qa = None
        self.delta = 0

    def do_step(self, observation, reward, reset):
        """
        This function handles the interaction with Tasks, by taking
        (observation, reward) and selecting an action.
        'reset' communicates to the network that the episode has finished.

        :param: observation: current output from Task
        :param: reward: scalar reward obtained from Task
        :param: reset: boolean value that indicates Task end.

        :return: selected action (one-hot encoded)
        """
        logging.getLogger(__name__).debug("reward: {}, reset: {}, observation: {}".format(reward, reset, observation))

        self.compute_input(observation)
        self.compute_hiddens()
        self.compute_output()

        # NOTE: select_action sets the selected action to prev_action
        self.select_action()

        # Set predicted expected value for action
        self.exp_val = self.z[self.prev_action]

        # Compute TD-error
        if self.prev_qa is None:
            # No previous expectation. Set to current expectation.
            self.prev_qa = self.exp_val

        # Transition to terminal state: expected value is 0
        if reset:
            self.exp_val = 0.

        # Calculate TD-error:
        self.delta = self.compute_delta(reward)

        # Update weights:
        self.update_weights()

        # Check whether episode end was encountered
        if reset:
            logging.getLogger(__name__).debug("Episode end - resetting dynamic variables.")
            self.reset_all()
        else:
            # Update traces and tags:
            self.update_traces()
            self.update_tags()

        # Set previous observation to current
        self.prev_obs = self.x_reg
        self.prev_qa = self.exp_val

        # Return action (1-hot encoded vector)
        return self.z_prime

    def compute_input(self, observation):
        """
        Compute input to the network (takes care of transient units)
        :param: observation: current output from Task
        """

        diff = observation - self.prev_obs

        # where(cond, a, b) returns a vector with a if cond is True and b if cond is False
        d_pos = np.where((diff > 0), 1, 0)  # indicator array with diffs > 0
        d_neg = np.where((diff < 0), 1, 0)  # indicator array with diffs < 0

        self.x_trans = np.hstack((np.abs(d_pos * diff), np.abs(d_neg * diff)))
        self.x_reg = observation

    def compute_hiddens(self):
        """
        Compute activations of hidden units.
        """
        self.compute_regular_hiddens()
        self.compute_memory_hiddens()

    def compute_regular_hiddens(self):
        """
        Compute regular hidden unit activations
        """
        # Linear sum through weights:
        yacts = np.dot(np.hstack((np.ones(self.bias_input), self.x_reg)), self.weights_xy_reg)
        # Sigmoid transform
        self.y_reg = self.reg_transform.transform(yacts)

    def compute_memory_hiddens(self):
        """
        Compute memory hidden unit activations
        """
        # Linear sum through weights:
        yacts = np.dot(self.x_trans, self.weights_xy_mem)
        self.y_mem_tot += yacts

        # Sigmoid transform
        self.y_mem = self.mem_transform.transform(self.y_mem_tot)

    def compute_output(self):
        """
        Compute Q-unit activations
        """
        zacts_reg = np.dot(np.hstack((np.ones(self.bias_hidden), self.y_reg)), self.weights_yz_reg)
        zacts_mem = np.dot(np.hstack((np.ones(self.bias_mem_hidden), self.y_mem)), self.weights_yz_mem)

        self.z = zacts_reg + zacts_mem

        # Round the Q-values to desired precision
        self.z = np.round(self.z, self.prec_q_acts)

    def select_action(self):
        """
        Select actions based on Q-unit activations and selected controller.
        """
        max_q = np.max(self.z)

        # Determine whether to make exploratory move:
        if np.random.sample() <= self.epsilon:
            action = self.explore(self.z)
        else:
            # Take greedy action (break ties randomly):
            idces = np.where(self.z == max_q)[0]
            if np.size(idces) > 1:
                action = idces[np.random.randint(0, np.size(idces))]
            else:
                action = idces[0]

        # Set output-action:
        self.z_prime = np.zeros(self.nz)
        self.z_prime[action] = 1
        self.prev_action = action

    @staticmethod
    def compute_softmax(values):
        """
        Compute softmax transformation
        """
        # Trick for numerical stability
        values = values - np.max(values)

        # Pull result through softmax operator:
        exps = np.exp(values)
        values = exps / np.sum(exps)

        return values

    @staticmethod
    def select_boltzmann(values):
        """
        Select random action from Boltzmann distribution
        by Roulette wheel selection
        """
        boltz = AugmentNetwork.compute_softmax(values)

        # Create wheel:
        probs = [sum(boltz[:i + 1]) for i in range(len(boltz))]

        # Select from wheel
        rnd = np.random.sample()
        for (i, prob) in enumerate(probs):
            if rnd <= prob:
                return i

    def select_uniform_random(self, values):
        """
        select uniform random action
        """
        return np.random.randint(0, self.nz)

    def compute_delta(self, reward):
        """
        Compute SARSA TD error.
        """
        return reward + (self.gamma * self.exp_val) - self.prev_qa

    def update_weights(self):
        """
        Do weight updates
        """
        # Update weights:
        # hidden to output:
        self.weights_yz_reg += self.beta * self.delta * self.yz_reg_tags
        self.weights_yz_mem += self.beta * self.delta * self.yz_mem_tags

        # input to hidden:
        self.weights_xy_reg += self.beta * self.delta * self.xy_reg_tags
        self.weights_xy_mem += self.beta * self.delta * self.xy_mem_tags

    def update_traces(self):
        """
        Update the traces. Note that these are only located between the input and hidden layer
        """
        # Decay traces:
        # Input to hidden:
        self.xy_reg_traces *= 0.  # Regular traces decay in one time step
        self.xy_mem_traces *= self.mem_decays  # Memory input traces do not decay in vanilla AuGMEnT

        # Update traces:
        # Input to hidden
        self.xy_reg_traces += (np.hstack((np.ones(self.bias_input), self.x_reg))).reshape(self.nx_inst + self.bias_input, 1)
        self.xy_mem_traces += (self.x_trans).reshape(self.nx_trans, 1)

    def update_tags(self):
        """
        Compute the updates for the tags (~eligibity traces). Traces are the
        records of feedforward activation that went over the synapses to
        postsynaptic hidden neurons.
        """
        # 1. Decay old tags:

        # Input to hidden:
        self.xy_reg_tags = self.xy_reg_tags * self.L * self.gamma
        self.xy_mem_tags = self.xy_mem_tags * self.L * self.gamma

        # Hidden to output:
        self.yz_reg_tags = self.yz_reg_tags * self.L * self.gamma
        self.yz_mem_tags = self.yz_mem_tags * self.L * self.gamma

        # 2. Update tags:

        # Output to hidden:
        self.yz_reg_tags[:, self.prev_action] += np.hstack((np.ones(self.bias_hidden), self.y_reg))
        self.yz_mem_tags[:, self.prev_action] += np.hstack((np.ones(self.bias_mem_hidden), self.y_mem))

        # Input to hidden:
        # Here feedback and traces interact to form tag update:

        # Regular units:

        # Compute derivatives for regular units
        d_hr = self.reg_transform.derivative(self.y_reg)

        # Feedback from output layer to regular hidden units:
        fb_reg = self.weights_yz_reg[self.bias_hidden:, self.prev_action]

        # Actual update:
        fbxderiv_reg = d_hr * fb_reg
        self.xy_reg_tags += self.xy_reg_traces * fbxderiv_reg

        # Memory units:

        # Compute derivatives for memory units
        d_hm = self.mem_transform.derivative(self.y_mem)

        # Feedback from output layer to memory hidden units:
        fb_mem = self.weights_yz_mem[self.bias_mem_hidden:, self.prev_action]

        # Actual update:
        fbxderiv_mem = d_hm * fb_mem
        self.xy_mem_tags += self.xy_mem_traces * fbxderiv_mem

    def reset_traces(self):
        """
        Reset all Traces to zero
        """
        self.xy_reg_traces = np.zeros_like(self.xy_reg_traces)
        self.xy_mem_traces = np.zeros_like(self.xy_mem_traces)
        self.yz_reg_traces = np.zeros_like(self.yz_reg_traces)
        self.yz_mem_traces = np.zeros_like(self.yz_mem_traces)

    def reset_tags(self):
        """
        Reset all Tags to zero
        """
        self.xy_reg_tags = np.zeros_like(self.weights_xy_reg)
        self.xy_mem_tags = np.zeros_like(self.weights_xy_mem)
        self.yz_reg_tags = np.zeros_like(self.weights_yz_reg)
        self.yz_mem_tags = np.zeros_like(self.weights_yz_mem)

    def reset_memory(self):
        """
        Reset all memory unit activations to zero
        """
        self.y_mem_tot = np.zeros(self.ny_memory)
        self.y_mem = np.zeros(self.ny_memory)

    def reset_all(self):
        """
        Resets memory units and all traces at end of episode.
        """
        self.reset_memory()
        self.reset_traces()
        self.reset_tags()

        self.prev_obs = np.zeros(self.nx_inst)
        self.prev_qa = 0
        self.prev_max = 0.

    def set_learning(self, state='off'):
        """
        Turn learning off / on
        """
        if state == 'off':
            self.__beta = self.beta
            self.beta = 0.
        else:
            self.beta = self.__beta
            
    def set_exploration(self, state='off'):
        """
        Turn exploration off / on
        """
        if state == 'off':
            self.__epsilon = self.epsilon
            self.epsilon = 0.
        else:
            self.epsilon = self.__epsilon

    def generate_weights(self, n_in, n_out):
        """
        Generate random (n_in, n_out) weight matrix in range self.weight_range.
        """
        weights = (2. * self.weight_range) * rand(n_in, n_out) - self.weight_range

        return weights

    def __getstate__(self):
        """
        Determines class attributes that get pickled to file
        """
        odict = self.__dict__.copy()  # copy the dict since we change it

        # Remove methods, as they can not be saved to a file
        del odict['explore']  # Note: this is restored by explore_nm - brittle
        del odict['mem_transform']
        del odict['reg_transform']

        return odict

    def __setstate__(self, dict):
        """
        Reconstruct AuGMEnT object from pickled dict
        """
        if dict['explore_nm'] == 'max-boltzmann':
            dict['explore'] = self.select_boltzmann
        elif dict['explore_nm'] == 'e-greedy':
            dict['explore'] = self.select_uniform_random

        dict['mem_transform'] = Sigmoid(theta=dict['theta'])
        dict['reg_transform'] = Sigmoid(theta=dict['theta'])

        # Update dictionary:
        self.__dict__.update(dict)

    def save_network(self, path_to_file):
        """
        Pickle the network object to a file. Note that this is brittle, it just pickles the objects' self.__dict__
        """
        try:
            fl = open(path_to_file, 'wb')
            pickle.dump(self, fl)
            fl.close()
        except IOError, v:
            print "Something went wrong trying to access: {}".format(path_to_file)
            print v

