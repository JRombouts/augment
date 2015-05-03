"""
Simple example to show how to configure and run a task and an AuGMEnT network.

Trains AuGMEnT network on a saccade-antisaccade task.

If you have matplotlib installed, performance graphs will be shown.

@author: J.O. Rombouts
"""
from augment import AugmentNetwork
from task import GGSA
import numpy as np
import logging

# Disable plotting if matplotlib is not installed
global PLOTTING
PLOTTING = False
try:
    import matplotlib.pyplot as plt
except ImportError as e:
    PLOTTING = False
    print "matplotlib is not installed - disabling plotting"


def run_experiment(seed):

    # Set random seed:
    np.random.seed(seed)

    # Maximal number of trials to run
    trials = 100000

    # Target proportion correct over all trial types:
    stopping_crit = 0.85

    # Measurement window for stopping criterion
    stopping_crit_window = 100

    task = GGSA()

    # Buffers for storing performance for all trial_types
    n_trial_types = len(task.trial_types)
    result_buffer = np.zeros((n_trial_types, stopping_crit_window))
    result_buffer_idces = np.zeros(n_trial_types)
    result_buffer_avgs = np.zeros((n_trial_types, trials))

    # Trial results (0: failure, 1: success)
    trial_res = np.zeros(trials)

    # Get size of task output (to determine input layer size)
    new_input = task.display.to_output()

    # Build network (see AugmentNetwork source for all options)
    network = AugmentNetwork(beta=0.15, L=0.2, n_inputs=len(new_input), nz=3, ny_memory=4, ny_regular=3,
                             weightrange=.25, gamma=0.90)

    reward = 0.
    reset = False
    converged = False
    c_epoch = -1
    unstable = False

    if PLOTTING:
        # For live update of plots
        plt.ion()
        g = Graphs(max_trials=trials, n_subtasks=n_trial_types)
        plot_interval = 500
    else:
        logging.getLogger(__name__).warning("Not generating plots - matplotlib is ")

    for trial in range(0, trials):

        # Plotting of network performance
        if PLOTTING and trial % plot_interval == 0:
            logging.getLogger(__name__).info("trial = {:d}".format(trial))

            # Plot performance
            g.update_network_performance(trial_res)

            # Plot performance on all trial types separately
            for i in range(n_trial_types):
                y = result_buffer_avgs[i, :]
                g.update_subtask_performance(i, y)
            g.draw()

        if converged:
            break

        trial_running = True
        while trial_running and not converged and not unstable:

            # Get action from network based on latest output from Task
            action = network.do_step(new_input, reward, reset)

            # Networks can become unstable with high learning rates and
            # long trace decay times. If this happens, try tuning the parameters
            # Figure 7 in the paper indicates good parameter combinations
            if not check_stable(network.z):
                unstable = True
                break

            if reset:  # End of trial detected:
                logging.getLogger(__name__).info("Trial {:d} end".format(trial))

                trial_running = False
                tmp_tp = task.last_trial_type

                if reward == task.fin_reward:  # Mark trial as successful
                    logging.getLogger(__name__).info("Obtained reward!")

                    trial_res[trial] = 1

                    # Add result to results for specific input-pattern:
                    result_buffer[tmp_tp, result_buffer_idces[tmp_tp]] = 1

                    # Compute convergence for all buffers:
                    # Note that all last stopping_crit_window trials of all types have to be
                    # at convergence criterion.
                    if np.all(np.average(result_buffer, axis=1) > stopping_crit):
                        # Achieved criterion performance on all trial types
                        # Now, check that all patterns can be classified correctly
                        # without exploration (and fixed network-weights)
                        if check_performance(network):
                            converged = True
                            c_epoch = trial
                else:  # Mark trial type as failed
                    result_buffer[tmp_tp, result_buffer_idces[tmp_tp]] = 0

                # Increase (circular) buffer index
                result_buffer_idces[tmp_tp] += 1
                result_buffer_idces[tmp_tp] %= stopping_crit_window

                # Compute average performance on each trial type
                for i in range(n_trial_types):
                    result_buffer_avgs[i, trial] = np.mean(result_buffer[i, :])

            # Obtain new task state, based on last network action
            new_input, reward, reset = task.do_step(action)

    # Done running experiment.
    # Compute performance averaged over all trials
    mean_performance = np.mean(trial_res[:c_epoch])

    summary = {'unstable': int(unstable), 'convergence': int(converged), 'c_epoch': c_epoch,
               'mean_performance': mean_performance, 'trial_res': trial_res}

    # Save trained network
    network.save_network('ggsa_%i.cpickle' % seed)

    if PLOTTING:
        # Show the graph, blocking execution
        print "Close graph to terminate."

        # Update graphs:
        g.update_network_performance(trial_res)

        # Plot performance on all trial types separately
        for i in range(n_trial_types):
            y = result_buffer_avgs[i, :]
            g.update_subtask_performance(i, y)

        g.plot_convergence_trial(c_epoch, rescale=True)
        g.draw()
        plt.ioff()
        plt.show(block=True)

    return summary


def check_performance(network):
    """
    Run all trial types without learning and exploration.
    Returns True if all valid input patterns are correctly dealt with.
    """

    # Toggle learning and exploration off
    network.set_learning()
    network.set_exploration()

    success = True
    # Iterate over trial types:
    task = GGSA()
    for i in range(0, len(task.trial_types)):

        tmp_task = GGSA()

        new_input = tmp_task.display.to_output()
        tmp_task.set_trial_type(i)

        reward = 0
        reset = False

        # Run trial type until completion
        while True:
            action = network.do_step(new_input, reward, reset)

            if reset:  # End of trial detected:
                # Check for failure
                if reward != tmp_task.fin_reward:
                    success = False
                break

            (new_input, reward, reset) = tmp_task.do_step(action)

        if not success:
            break

    # Reactivate Network
    network.set_learning('on')
    network.set_exploration('on')

    return success

def check_stable(vals):
    """
    Check whether any value in vals is NaN or extremely large
    """
    if np.max(np.abs(vals)) > 1e+3:
        return False
    if np.any(np.isnan(vals)):
        return False
    if np.any(np.isinf(vals)):
        return False
    return True


class Graphs(object):
    """
    Simple container class for plots
    """

    def __init__(self, max_trials, n_subtasks):
        """
        Create empty graphs with subplots for global network performance
        and detailed subtask performance
        """
        self.figure, self.axarr = plt.subplots(1, 2, sharey=True)
        
        self.axarr[0].set_title('Network performance')
        self.axarr[0].set_xlabel('Trials')
        self.axarr[0].set_ylabel('Performance')
        self.axarr[0].set_ylim([0, 1.05])
        self.filter_size = 100
        
        self.max_trials = max_trials
        self.perf_line, = self.axarr[0].plot(range(max_trials), np.zeros(self.max_trials,), lw=2)
        
        self.axarr[1].set_title('Trial type performance')
        self.axarr[1].set_xlabel('Trials')
        self.perf_lines = self.axarr[1].plot(np.zeros((self.max_trials, n_subtasks)), lw=2)
        
        # callback for getting coordinate data from plots
        self.figure.canvas.mpl_connect('button_press_event', self.on_click)
        
        # Rotate x-labels
        for ax in self.axarr:
            labels = ax.get_xticklabels() 
            plt.setp(labels, rotation=45)
        
        self.c_epoch = -1

    def on_click(self, event):
        """
        Callback for graphs
        """
        if event.button == 1:
            print "x={:2f}, y={:2f}".format(event.xdata, event.ydata)
    
    def update_network_performance(self, trial_res):
        """
        Plot smoothed network performance
        """
        y = np.convolve(np.ones(self.filter_size)/float(self.filter_size), trial_res, mode='same')
        self.perf_line.set_ydata(y)
    
    def update_subtask_performance(self, task_id, ydata):
        self.perf_lines[task_id].set_ydata(ydata)
        
    def plot_convergence_trial(self, c_epoch, rescale=False):
        self.c_epoch = c_epoch
        [ax.axvline(x=self.c_epoch, color='r', linestyle='--') for ax in self.axarr]
        
        # Rescale plots to learning time?
        if rescale and self.c_epoch != -1:
            [ax.set_xlim([0, self.c_epoch * 1.01]) for ax in self.axarr]
        
    def draw(self):
        self.figure.canvas.draw()


def main():
    # Change the logging level for more/less detailed information
    # e.g. use level=logging.ERROR to see only messages logged to ERROR level (and higher)
    logging.basicConfig(format=' [%(levelname)s] %(name)s %(message)s', level=logging.DEBUG)

    # Run a single experiment
    example = run_experiment(1)

    print example


if __name__ == '__main__':
    main()
