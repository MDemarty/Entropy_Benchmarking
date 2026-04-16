"""
PAPER

Circuit size bound for quantum advantage with VQA using a hardware-efficient quantum circuit
"""
import matplotlib.pyplot as plt
import numpy as np
import os

#Fixed parameters
p_DP2 = 0.001 #0.01 #0.054 #0.097
alpha = p_DP2
c  = 0.7

#Parameters for plotting
num_qubits_max = 100
depth_max = 400



fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, gridspec_kw={'width_ratios': [6, 1]})
fig.subplots_adjust(hspace=0.1)  # adjust space between axes

# what I want to plot
width = np.linspace(1, num_qubits_max, 1000)
dep_ours = [(1/(2*alpha))*(1/(n-1))*np.log((2**n - 1)/(2**(n*c) - 1)) for n in width] #non ceiling version for a continuous plot
dep_ours_0 = [(-1/((2*alpha)*(n-1)))*np.log(c) for n in width]
dep_ours_infty = [(1/(2*alpha))*(1 - c)*np.log(2) for _ in width]
dep_DR = [ (1/(2*alpha))*np.log(c**(-1)) for _ in width] #SFGP21 paper

# plot the same data on both axes
ax1.plot(width, dep_ours, label = 'this work') #Our condition
ax1.plot(width, dep_ours_infty, color = 'blue', linestyle = 'dotted', label = 'this work $n \\rightarrow +\\infty$') #Our condition n -> infty limit
ax1.plot(width, dep_DR, color = 'cyan', label = 'SFGP21') #Daniel and Raul's condition

ax2.plot(width, dep_ours, label = 'this work') #Our condition
ax2.plot(width, dep_ours_infty, color = 'blue', linestyle = 'dotted', label = 'this work $n \\rightarrow +\\infty$') #Our condition n -> infty limit
ax2.plot(width, dep_DR, color = 'cyan', label = 'SFGP21') #Daniel and Raul's condition

# zoom-in / limit the view to different portions of the data
ax1.set_xlim(0, 6)  # most of the data
ax2.set_xlim(99, 100)  # limit for large systems

ax1.set_ylim(0, depth_max)
ax2.set_ylim(0, depth_max) 

# hide the spines between ax and ax2
ax1.spines.right.set_visible(False) #hide right edge of left box
ax2.spines.left.set_visible(False) #hide left edge of right box
# remove useless ticks
ax2.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    left=False,      # ticks along the bottom edge are off
    right=False)#,         # ticks along the top edge are off
    #labelbottom=False) # labels along the bottom edge are off
# tick every two integer values for ax1
ax1.set_xticks(np.arange(0, 6, 1))
# tick every two integer values for ax2
ax2.set_xticks(np.arange(99, 101, 1))
# x and y axis labels
fig.supxlabel('Number of qubits')
fig.supylabel('Circuit depth')
# reduce space between two subplots
fig.subplots_adjust(wspace=0.1)

# legend
ax1.legend(loc=1)

# adding small diagonal lines for the cut-out
d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass plot, just so we don't keep repeating them
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False, markersize=12)
ax1.plot((1-d, 1+d), (-d, +d), **kwargs)
ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)

# Prepare directory - check if directory exists, if not create it
resultdir = 'Paper/Circuit_size_bound/'
if not os.path.exists(resultdir):
    os.makedirs(resultdir)
plt.savefig(resultdir + 'circuit_size_qadv_boundary-c_'+str(c)+'_p_DP2_'+str(p_DP2)+'.pdf')

plt.show()