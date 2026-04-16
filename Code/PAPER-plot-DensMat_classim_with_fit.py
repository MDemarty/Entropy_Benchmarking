"""
PAPER

A script where the purity, second-order Renyi entropy and von Neumann entropy
 of the output of a VQA circuit is computed numerically by direct simulation of 
 the evolution of the density matrix of the quantum register of the circuit
 -- noise model: local depolarising noise --
"""

import argparse

from libIO import load_from_json
from libExperiment import ExperimentParams
from libUtils import renyi_entropy_from_purity
from libPurityModel import purity_model_globalDP
from libPlot import compute_xticks, compute_depth_tab, compute_depth_tab_more_points

import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit


def build_parser (description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enables verbose mode")
    parser.add_argument("-i", "--input", type=str, required=False, default='results/Aer_sim/DensMat_HEA_RIGETTI/experiment_2025-03-17_n2-10_D15.json', help="Json file with parameters from experiment")
    parser.add_argument("-o", "--output", type=str, required=False, default='Paper/DensMat_Fit_plot', help="Folder where to store the results")

    return parser

parser = build_parser('Plotting metrics for DensMat with fitting (FOR PAPER)')
args = parser.parse_args()
verbose = args.verbose
jsonfilename = args.input
resultdir = args.output

# ========================= Read parameters of experiment from json file =============================
experiment_params = ExperimentParams.from_dict(load_from_json(jsonfilename))
if experiment_params == None:
    print ("ERROR: reading json file, no experiment can be loaded")
    exit()
experiment_params.results_dir = resultdir
circuit_params = experiment_params.circuit_params
noise_params = experiment_params.noise_params
#np.random.seed(experiment_params.seed)
num_qubits_min = circuit_params.num_qubits_min
num_qubits_max = circuit_params.num_qubits_max
num_qubits_step = circuit_params.num_qubits_step

save_fig = True
show = True # to show figures

# ==================================================================
# Load experiment results (metrics)
metrics = load_from_json(experiment_params.metrics_file)
all_R2d_results = metrics['all_R2d_diff_n']

# ==================================================================
# Fit
all_R2d_results_fit = []
all_alpha_1_optim_classim = []
all_alpha_2_optim_classim = []

depth_tab = compute_depth_tab(circuit_params.depth_min, circuit_params.depth_max, circuit_params.depth_step)
depth_tab_more_points = compute_depth_tab_more_points(circuit_params.depth_min, circuit_params.depth_max)
c = noise_params.p_DP1 / noise_params.p_DP2

count = 0
for num_qubits in range(num_qubits_min, num_qubits_max+1, num_qubits_step):
    # get the fit
    def R2d_model_globalDP_part_eval (depth, alpha_2):
        return renyi_entropy_from_purity(purity_model_globalDP (num_qubits, depth, alpha_2 * c, alpha_2))/num_qubits
    popt_classim, _ = curve_fit(R2d_model_globalDP_part_eval, depth_tab, all_R2d_results[count], bounds=(0,1))
    alpha_1_optim_classim = popt_classim[0] * c
    alpha_2_optim_classim = popt_classim[0]
    R2d_results_fit = [renyi_entropy_from_purity(purity_model_globalDP(num_qubits, depth, alpha_1_optim_classim, alpha_2_optim_classim))/num_qubits for depth in depth_tab_more_points]
    all_R2d_results_fit.append(R2d_results_fit)
    all_alpha_1_optim_classim.append(alpha_1_optim_classim)
    all_alpha_2_optim_classim.append(alpha_2_optim_classim)
    count += 1

# ==================================================================
# Plot

if save_fig:
    # Prepare directory - check if directory exists, if not create it
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

plt.figure(figsize=(15, 6))
xticks = compute_xticks(depth_tab, circuit_params.depth_step)

count = 0
for num_qubits in range(num_qubits_min, num_qubits_max+1, num_qubits_step):
    # the density matrix simulation data
    plt.scatter(x=depth_tab, y=all_R2d_results[count], marker='.', label='$n = $'+str(num_qubits)+' density matrix sim')

    alpha_1 = all_alpha_1_optim_classim[count]
    alpha_2 = all_alpha_2_optim_classim[count]
    label = '$n = $'+str(num_qubits)+' model - $\\alpha_1, \\alpha_2$ = {:.4f}, {:.4f}'.format(alpha_1, alpha_2)

    plt.plot(depth_tab_more_points, all_R2d_results_fit[count], label=label)
    count += 1
plt.gca().set_prop_cycle(None) #reset the color cycle


plt.axhline(y=1, color='black', linestyle='-.', label='maximally mixed state')
plt.legend(loc='lower right', fontsize='12')
plt.xlabel('Depth', fontsize='12')
plt.xticks(xticks)
plt.ylabel('Renyi-2 entropy density', fontsize='12')
plt.ylim(bottom=0, top=1.1)
if save_fig:
    filename = 'R2d-fit_density_matrix-n'+str(num_qubits_min)+'-'+str(num_qubits_max)+'_D'+str(circuit_params.depth_max)+'_DP'+str(noise_params.p_DP1)+'-'+str(noise_params.p_DP2) + '.pdf'
    plt.savefig(resultdir+'/'+filename)
plt.show()