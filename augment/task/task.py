"""
Implementation of Task base class and some example Tasks

Main entry point in Tasks is do_step, which takes latest network action
and returns new Task state.

@author: J.O. Rombouts
"""

import numpy as np
import logging


class Task(object):
    """
    Base class for Task.
    """

    def __init__(self, **kwargs):
        """
        Constructor. See comments below for parameters.
        """
        # Possible Task states
        self.states = {'intertrial', 'waitfix', 'fix', 'mem', 'go'}
        
        self.state = 'intertrial'
        self.trial_end = False
        self.output = None
        self.display = None
        self.observation = None
        
        # Task parameters 
        self.n_actions = 3  # Number of actions agent can select
        self.n_outputs = 4  # Number of outputs (i.e. input for agents)
        self.intertrial_dur = 0  # Number of time steps between trials
        self.fix_dur = 0  # Required duration of fixation
        self.mem_dur = 2  # Number of delay steps
       
        # Set these to allow for random length memory delays
        self.min_mem_dur = 2.
        self.max_mem_dur = 2.

        self.max_dur = 8  # Maximal number of steps to wait in GO state
        self.waitfix_timeout = 10  # Maximal number of steps to wait for fixation
        self.cur_reward = 0  # Reward accrued in current time step
        self.fix_reward = 0.2  # Reward for fixating
        self.fin_reward = 1.5  # Reward for correct completion of task
        
        self.total_trials = 0  # Trial counter
        self.correct_trials = 0  # Number of correct trials
        self.timer = Ticker()  # Timer
        
        # Dict mapping state to functions
        self.flowcontrol = {'intertrial': self.do_intertrial,
                            'waitfix': self.do_waitfix,
                            'fix': self.do_fix,
                            'mem': self.do_mem,
                            'go': self.do_go}

        self.trial_type = None
        self.trial_set_externally = False
    
    def do_step(self, observation):
        raise NotImplementedError
    
    def get_performance(self):
        if self.total_trials > 0:
            return float(self.correct_trials) / float(self.total_trials)
        else:
            return 0.
        
    def reset_trial_stats(self):
        self.correct_trials = 0
        self.total_trials = 0
    
    def state_reset(self):
        """
        End episode, reset Task
        """
        self.state = 'intertrial'
        self.total_trials += 1
        self.set_default_output()
        self.timer.reset()
        self.display.reset()
        self.trial_end = True
        self.last_trial_type = self.trial_type
        self.trial_type = -1
    
    def check_input(self, observation):
        """
        Make sure received action is a valid action
        """
        assert(len(observation) == self.n_actions)
        assert(sum(observation) == 1)

    def set_trial_type(self, type_int):
        """
        Set and fix trial type for next trials
        :param: type_int : index into trial_types dictionary
        """
        assert(type_int <= len(self.trial_types))
        self.trial_type = type_int
        self.trial_set_externally = True  
            
    def set_default_output(self):
        self.output = np.zeros(self.n_outputs)

    @staticmethod
    def interpret_observation(observation):
        raise NotImplementedError


class Ticker(object):

    def __init__(self):
        self.ticks = 0

    def reset(self):
        self.ticks = 0

    def tick(self):
        self.ticks += 1
        

class GGSADisplay(object):
    """
    Emulates a "display" for a GGSA Task
    """
    def __init__(self):
        """
        Constructor
        """
        self.cue_left = False
        self.cue_right = False
        self.fp_pro = False
        self.fp_anti = False
    
    def reset(self):
        self.cue_left = False
        self.cue_right = False
        self.fp_pro = False
        self.fp_anti = False
            
    def to_output(self):
        return np.array([self.fp_pro, self.fp_anti, self.cue_left, self.cue_right], dtype=int)
    
    def set_fp(self, opt):
        if opt == 'anti':
            self.fp_anti = True
        else:
            self.fp_pro = True
    
    def clear_fp(self):
        self.fp_pro = False
        self.fp_anti = False
            
    def set_cue(self, opt):
        if opt == 'right':
            self.cue_right = True
        else:
            self.cue_left = True
            
    def clear_cue(self):
        self.cue_left = False
        self.cue_right = False  
    

class GGSA(Task):
    """
    Models delayed anti-saccade tasks as in e.g.:

    Gnadt JW, Andersen RA (1988) Memory related motor planning activity in posterior parietal cortex of macaque.
    Exp Brain Res 70: 216-220.

    Gottlieb J, Goldberg ME (1999) Activity of neurons in the lateral intraparietal area of the monkey
    during an antisaccade task. Nat Neurosci 2: 906-912.
    """

    def __init__(self, **kwargs):
        """
        Constructor
        """
        super(GGSA, self).__init__(**kwargs)
        
        self.display = GGSADisplay()
        
        self.trial_type = None
        self.last_trial_type = None
        self.trial_set_externally = False
        
        # Cue, FP, Target
        self.trial_types = {0: ["left", "pro", "left"],
                            1: ["right", "pro", "right"],
                            2: ["left", "anti", "right"],
                            3: ["right", "anti", "left"]}
    
    def state_reset(self):
        logging.getLogger(__name__).debug("Restarting trial.")
        super(GGSA, self).state_reset()
   
    def do_step(self, observation):
        logging.getLogger(__name__).debug("Agent action: {}".format(self.interpret_observation(observation)))

        self.check_input(observation)
        self.trial_end = False
        self.observation = self.interpret_observation(observation)

        self.flowcontrol[self.state]()
        
        reward = self.cur_reward
        self.cur_reward = 0
        return self.display.to_output(), reward, self.trial_end
    
    def do_intertrial(self):
        logging.getLogger(__name__).debug("In inter-trial.")
        if self.timer.ticks == self.intertrial_dur:
            self.generate_trial() 
            self.display.set_fp(self.fp)
            self.state = 'waitfix'
            self.timer.reset()
        else:
            self.timer.tick()
        
    def do_waitfix(self):
        logging.getLogger(__name__).debug("In wait-fix state.")
        if self.timer.ticks <= self.waitfix_timeout:
            if self.observation == 'fixate':
                logging.getLogger(__name__).debug("Fixation acquired.")
                self.state = 'fix'
                self.timer.reset()
            else:
                self.timer.tick()
        else:
            self.state_reset()
        
    def do_fix(self):
        logging.getLogger(__name__).debug("In fixation state.")
         
        if self.observation != 'fixate':
            logging.getLogger(__name__).debug("Broke fixation.")
            self.state_reset()
        else:
            if self.timer.ticks == self.fix_dur:
                
                self.timer.reset()
                self.cur_reward += self.fix_reward

                # Enable cue
                self.display.set_cue(self.cue)
                
                if self.mem_dur > 0:
                    self.state = 'mem'
                else:
                    self.display.clear_fp()
                    self.state = 'go'
            else:
                self.timer.tick()
                            
    def do_mem(self):
        logging.getLogger(__name__).debug("In memory delay state")
        self.display.clear_cue()

        if self.observation != 'fixate':
            logging.getLogger(__name__).debug("Broke fixation.")
            self.state_reset()
        else:
            if self.timer.ticks == self.mem_dur:
                self.display.clear_fp()
                
                self.state = 'go'
                self.timer.reset()
            else:
                self.timer.tick()
                
    def do_go(self):
        logging.getLogger(__name__).debug("In go state.")
        if self.timer.ticks <= self.max_dur:
            if self.observation != 'fixate':
                if self.observation == self.target:
                    logging.getLogger(__name__).debug("Success!")
                    self.cur_reward += self.fin_reward
                    self.correct_trials += 1
                else:
                    logging.getLogger(__name__).debug("Failure due to wrong action!")
                    pass
                self.state_reset()
            else:
                self.timer.tick()
        else:  # Time-out
            logging.getLogger(__name__).debug("Time-out failure.")
            self.state_reset()
        
    def generate_trial(self):
        if not self.trial_set_externally:
            # Generate a cue location
            self.trial_type = np.random.randint(0, len(self.trial_types))
            
        self.cue = self.trial_types[self.trial_type][0]
        self.fp = self.trial_types[self.trial_type][1]
        self.target = self.trial_types[self.trial_type][2]

        # generate a memory delay in [self.min_mem_dur, self.max_mem_dur + 1)
        self.mem_dur = np.random.randint(self.min_mem_dur, self.max_mem_dur + 1)

    @staticmethod
    def interpret_observation(observation):
        if observation[0]:
            return 'fixate'
        if observation[1]:
            return 'left'
        if observation[2]:
            return 'right'

        
class GGSAProOnly(GGSA):
    """
    Gottlieb and Goldberg task with only pro-saccades
    """

    def __init__(self):
        """
        Constructor
        """
        super(GGSAProOnly, self).__init__()
        
        # Cue, FP, Target
        self.trial_types = {0: ["left", "pro", "left"],
                            1: ["right", "pro", "right"],
                            2: ["left", "anti", "left"],
                            3: ["right", "anti", "right"]}

 
class GGSAGnadt(GGSA):
    """
    Gnadt and Andersen task with only pro saccade markers
    """
    def __init__(self):
        super(GGSAGnadt, self).__init__()
        # Cue, FP, Target
        self.trial_types = {0: ["left", "pro", "left"],
                            1: ["right", "pro", "right"]}
