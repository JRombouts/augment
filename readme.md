# AuGMEnT

This is a simple implementation of the AuGMEnT network and learning rule as described in:

Rombouts JO, Bohte SM, Roelfsema PR (2015)
How Attention Can Create Synaptic Tags for the Learning of Working Memories in Sequential Tasks.
[PLoS Comput Biol 11: e1004060.](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004060)

As an example, we provide an implementation of several variations of (delayed) saccade/anti-saccade tasks.

# Dependencies

We have tried to keep dependencies as minimal as possible. To run the example you will need:

- [python](https://www.python.org) (code was tested in Python 2.7.6)
- [numpy](http://www.numpy.org) (tested on 1.8.0)
- [matplotlib](http://matplotlib.org) (tested on 1.3.1, required if you want to plot performance results)

On MacOS, we recommend to use [homebrew](http://brew.sh) to install python and the required dependencies. See e.g. 
[here](https://joernhees.de/blog/2014/02/25/scientific-python-on-mac-os-x-10-9-with-homebrew/) for a step by step guide that includes installing our dependencies.

# Running a simple example task

To run, type:

```python runner.py```

This creates an AuGMEnT Network that learns the Gottlieb and Goldberg Delayed Saccade/Antisaccade Task (the first task in the paper). 

The code consists of an agent (```augment/augment.py```) that interacts with a task environment (```task/task.py```). 

The main function in augment is ```do_step()```, which takes care of interacting with the task. 
The most tricky part in the network code is in ```update_traces()``` and ```update_tags()``` - these compute the variables required for the weight updates based on feed-forward and feedback activations (in this code the feedback is computed by assuming symmetric weights, but it would also work to start from random forward and backward weights).

If you have matplotlib installed, ```runner.py``` generates a plot that shows the performance of the network on the complete task and on each sub-task individually. 


# Citing

If you use this code for any publications, please consider citing our paper:

Rombouts JO, Bohte SM, Roelfsema PR (2015)
How Attention Can Create Synaptic Tags for the Learning of Working Memories in Sequential Tasks.
[PLoS Comput Biol 11: e1004060.](http://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1004060)

# Licence
 
This software is distributed under the MIT licence - see LICENSE for details.

