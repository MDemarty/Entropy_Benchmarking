"""
PAPER
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

from libUtils import get_metrics_specific_width, renyi_entropy_from_purity
from libIO import load_from_json

from libPurityModel import purity_model_globalDP_CS_circuit_measerr
from libPlot import compute_xticks, compute_depth_tab, compute_depth_tab_more_points, compute_filename
from libExperiment import ExperimentParams

def build_parser (description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enables verbose mode")
    parser.add_argument("-i1", "--input1", type=str, required=False, default='results/Aer_sim/CS_HEA_RIGETTI/experiment_2025-04-06_n3-3_D15_M320_K1000_grp5_meas0-0rand_wo.json', help="Json file for HEA_RIGETTI - CS - without measurement error (detector+circ)")
    parser.add_argument("-i2", "--input2", type=str, required=False, default='results/Aer_sim/CS_HEA_RIGETTI/experiment_2025-07-15_n3-3_D15_M320_K1000_grps5_meas0.03rand_w_ADcirc_correct.json', help="Json file for HEA_RIGETTI - CS - with measurement error (detector+circ)")
    parser.add_argument("-i3", "--input3", type=str, required=False, default='results/Rigetti_QPU/CS_HEA_RIGETTI/experiment_2023-12-09_n3-3_D15_M320_K1000_artif1_grps5.json', help="Json file for Rigetti_QPU - HEA_RIGETTI - CS artif1")
    parser.add_argument("-i4", "--input4", type=str, required=False, default='results/Rigetti_QPU/CS_HEA_RIGETTI/experiment_2023-12-09_n3-3_D15_M320_K1000_artif2_grps5.json', help="Json file for Rigetti_QPU - HEA_RIGETTI - CS artif2")
    parser.add_argument("-i5", "--input5", type=str, required=False, default='results/Aer_sim/DensMat_HEA_RIGETTI/experiment_2025-03-18_n3-3_D15.json', help="Json file for DensMat")
    parser.add_argument("-o", "--output", type=str, required=False, default='Paper/Experimental_plot', help="Folder where to store the results")

    return parser

parser = build_parser('Plotting comparison between two backends (for PAPER)')
args = parser.parse_args()
verbose = args.verbose
jsonfilename1 = args.input1
jsonfilename2 = args.input2
jsonfilename3 = args.input3
jsonfilename4 = args.input4
jsonfilename5 = args.input5
resultdir = args.output

compute_exact = True #Exact purity/entropy #NOTE keep as True
show_fully_mixed_state = True
save_fig = True

# ========================= Experiment1 (HEA_RIGETTI - Classical Shadows - without measurement error) =============================
experiment1 = ExperimentParams.from_dict(load_from_json(jsonfilename1))
if experiment1 == None:
    print ("ERROR: reading json file, no experiment #1 can be loaded")
    exit()
circuit_params1 = experiment1.circuit_params
noise_params1 = experiment1.noise_params
num_qubits = circuit_params1.num_qubits_min

#CS_params = CSParams(num_samples=3, num_groups=3, M=450, K=1000, protocol_choice='randomized')

# Get metrics
metrics_classim = load_from_json(experiment1.metrics_file)

# Extract purity, renyi entropy density (average, std and exact)
short_metrics_classim = get_metrics_specific_width(metrics_classim, num_qubits, num_qubits)

# ========================= Experiment2 (HEA_RIGETTI - Classical Shadows - with measurement error) =============================
experiment2 = ExperimentParams.from_dict(load_from_json(jsonfilename2))
if experiment2 == None:
    print ("ERROR: reading json file, no experiment #2 can be loaded")
    exit()

# Get metrics
metrics_classim_measerr = load_from_json(experiment2.metrics_file)

# Extract purity, renyi entropy density (average, std and exact)
short_metrics_classim_measerr = get_metrics_specific_width(metrics_classim_measerr, num_qubits, num_qubits)

# ========================= Experiment3 (Rigetti_QPU - artif rand method 1 - derand) =============================
experiment3 = ExperimentParams.from_dict(load_from_json(jsonfilename3))
if experiment3 == None:
    print ("ERROR: reading json file, no experiment #3 can be loaded")
    exit()

# Get metrics
metrics = load_from_json(experiment3.metrics_file)

# Extract purity, renyi entropy density (average, std and exact)
short_metrics = get_metrics_specific_width(metrics, num_qubits, num_qubits)

# ========================= Experiment4 (Rigetti_QPU - artif rand method 2 - derand) =============================
experiment4 = ExperimentParams.from_dict(load_from_json(jsonfilename4))
if experiment4 == None:
    print ("ERROR: reading json file, no experiment #4 can be loaded")
    exit()
#CS_params.num_samples = 1
CS_params = experiment4.protocol_params

# Get metrics
metrics_2 = load_from_json(experiment4.metrics_file)

# Extract purity, renyi entropy density (average, std and exact)
short_metrics_2 = get_metrics_specific_width(metrics_2, num_qubits, num_qubits)

# ========================= Experiment5 (HEA_RIGETTI - DensMat =============================
experiment5 = ExperimentParams.from_dict(load_from_json(jsonfilename5))
if experiment5 == None:
    print ("ERROR: reading json file, no experiment #5 can be loaded")
    exit()

# Get metrics
metrics_exact = load_from_json(experiment5.metrics_file)

# Extract purity, renyi entropy density (average, std and exact)
short_metrics_exact = get_metrics_specific_width(metrics_exact, num_qubits, num_qubits)

# ========================= Fitted model for the classical simulation & for the QPU =============================
np.random.seed(experiment1.seed) #reference
c = noise_params1.p_DP1/noise_params1.p_DP2

# Model for the QPU and for the simulation

def purity_model_globalDP_CS_circuit_measerr_part_eval (depth, alpha_2, beta): #with CS circuit+measurement error
    return purity_model_globalDP_CS_circuit_measerr (num_qubits, depth, alpha_2 * c, alpha_2, beta)

depth_tab = compute_depth_tab(circuit_params1.depth_min, circuit_params1.depth_max, circuit_params1.depth_step)
depth_tab_more_points = compute_depth_tab_more_points(circuit_params1.depth_min, circuit_params1.depth_max)
xticks = compute_xticks(depth_tab, circuit_params1.depth_step)

# Fitting the model for the classical simulation (WTHOUT MEASUREMENT ERROR)
popt_classim, _ = curve_fit(purity_model_globalDP_CS_circuit_measerr_part_eval, depth_tab, short_metrics_classim['all_pur_mean_diff_n'], bounds=(0,1))
alpha_1_optim_classim = popt_classim[0] * c
alpha_2_optim_classim = popt_classim[0]
beta_optim_classim = popt_classim[1]

# Fitting the model for the classical simulation (WTH MEASUREMENT ERROR)
popt_classim_measerr, _ = curve_fit(purity_model_globalDP_CS_circuit_measerr_part_eval, depth_tab, short_metrics_classim_measerr['all_pur_mean_diff_n'], bounds=(0,1))
alpha_1_optim_classim_measerr = popt_classim_measerr[0] * c
alpha_2_optim_classim_measerr = popt_classim_measerr[0]
beta_optim_classim_measerr = popt_classim_measerr[1]


# Fitting the model for the QPU (method 1)
popt_QPU, _ = curve_fit(purity_model_globalDP_CS_circuit_measerr_part_eval, depth_tab, short_metrics['all_pur_mean_diff_n'], bounds=(0,1))
alpha_1_optim_QPU = popt_QPU[0] * c
alpha_2_optim_QPU = popt_QPU[0]
beta_optim_QPU = popt_QPU[1]

# Fitting the model for the QPU (method 2)
popt_QPU_2, _ = curve_fit(purity_model_globalDP_CS_circuit_measerr_part_eval, depth_tab, short_metrics_2['all_pur_mean_diff_n'], bounds=(0,1))
alpha_1_optim_QPU_2 = popt_QPU_2[0] * c
alpha_2_optim_QPU_2 = popt_QPU_2[0]
beta_optim_QPU_2 = popt_QPU_2[1]

# ========================= Plots =============================

if save_fig:
    # Prepare directory - check if directory exists, if not create it
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

num_samples = 3
filename = 'n%d_M%d_K%d_grps%d_spls%d-rand-T1.pdf' % (num_qubits, CS_params.M, CS_params.K, CS_params.num_groups, num_samples)
full_filename_puri = compute_filename(resultdir, 'puri', filename)
full_filename_R2d = compute_filename(resultdir, 'R2d', filename)

# Purity
plt.figure(figsize=(15, 6))
# heuristic model
pur_model_QPU = [purity_model_globalDP_CS_circuit_measerr(num_qubits, depth, alpha_1_optim_QPU, alpha_2_optim_QPU, beta_optim_QPU) for depth in depth_tab_more_points]
pur_model_QPU_2 = [purity_model_globalDP_CS_circuit_measerr(num_qubits, depth, alpha_1_optim_QPU_2, alpha_2_optim_QPU_2, beta_optim_QPU_2) for depth in depth_tab_more_points] #NEW
pur_model_classim = [purity_model_globalDP_CS_circuit_measerr(num_qubits, depth, alpha_1_optim_classim, alpha_2_optim_classim, beta_optim_classim) for depth in depth_tab_more_points]
pur_model_classim_measerr = [purity_model_globalDP_CS_circuit_measerr(num_qubits, depth, alpha_1_optim_classim_measerr, alpha_2_optim_classim_measerr, beta_optim_classim_measerr) for depth in depth_tab_more_points] #NEW

plt.plot(depth_tab_more_points, pur_model_QPU, color="goldenrod", label="QPU model 1 - $\\alpha_1, \\alpha_2, \\beta$ = {:.4f}, {:.4f}, {:.4f}".format(alpha_1_optim_QPU, alpha_2_optim_QPU, beta_optim_QPU))
plt.plot(depth_tab_more_points, pur_model_QPU_2, color="darkgoldenrod", label="QPU model 2 - $\\alpha_1, \\alpha_2, \\beta$ = {:.4f}, {:.4f}, {:.4f}".format(alpha_1_optim_QPU_2, alpha_2_optim_QPU_2, beta_optim_QPU_2)) #NEW
plt.plot(depth_tab_more_points, pur_model_classim, color="steelblue", label="sim model - $\\alpha_1, \\alpha_2, \\beta$ = {:.4f}, {:.4f}, {:.4f}".format(alpha_1_optim_classim, alpha_2_optim_classim, beta_optim_classim))
plt.plot(depth_tab_more_points, pur_model_classim_measerr, color="darkblue", label="sim model meas+$T_1$ err - $\\alpha_1, \\alpha_2, \\beta$ = {:.4f}, {:.4f}, {:.4f}".format(alpha_1_optim_classim_measerr, alpha_2_optim_classim_measerr, beta_optim_classim_measerr)) #NEW
# classical shadows
plt.errorbar(x=depth_tab, y=short_metrics['all_pur_mean_diff_n'], yerr=short_metrics['all_pur_std_diff_n'], ecolor="goldenrod", capsize=3, ls='none', label="QPU shadows 1")
plt.errorbar(x=depth_tab, y=short_metrics_2['all_pur_mean_diff_n'], yerr=short_metrics_2['all_pur_std_diff_n'], ecolor="darkgoldenrod", capsize=3, ls='none', label="QPU shadows 2") #NEW
plt.errorbar(x=depth_tab, y=short_metrics_classim['all_pur_mean_diff_n'], yerr=short_metrics_classim['all_pur_std_diff_n'], ecolor="steelblue", capsize=3, ls='none', label="sim shadows")
plt.errorbar(x=depth_tab, y=short_metrics_classim_measerr['all_pur_mean_diff_n'], yerr=short_metrics_classim_measerr['all_pur_std_diff_n'], ecolor="darkblue", capsize=3, ls='none', label="sim shadows meas+$T_1$ err") #NEW
if compute_exact:
    # n=3 density matrix simulation
    plt.plot(depth_tab, short_metrics_exact['all_pur_diff_n'], color="black", label="density matrix sim")
if show_fully_mixed_state:
    plt.axhline(y=1/(2**num_qubits), color='black', linestyle='-.', label="maximally mixed state") # n=3
plt.xlabel('Depth', fontsize='12')
plt.xticks(xticks)
plt.ylabel('Purity', fontsize='12')
plt.ylim(bottom=0, top=1.1)

plt.legend(fontsize='12')
if save_fig:
    plt.savefig(full_filename_puri)
plt.show()

# Renyi-2 entropy density
plt.figure(figsize=(15, 6))
# heuristic model
R2d_model_QPU = [renyi_entropy_from_purity(puri)/num_qubits for puri in pur_model_QPU]
R2d_model_QPU_2 = [renyi_entropy_from_purity(puri)/num_qubits for puri in pur_model_QPU_2] #NEW
R2d_model_classim = [renyi_entropy_from_purity(puri)/num_qubits for puri in pur_model_classim]
R2d_model_classim_measerr = [renyi_entropy_from_purity(puri)/num_qubits for puri in pur_model_classim_measerr] #NEW
plt.plot(depth_tab_more_points, R2d_model_QPU, color="goldenrod", label="QPU model 1 - $\\alpha_1, \\alpha_2, \\beta$ = {:.4f}, {:.4f}, {:.4f}".format(alpha_1_optim_QPU, alpha_2_optim_QPU, beta_optim_QPU))
plt.plot(depth_tab_more_points, R2d_model_QPU_2, color="darkgoldenrod", label="QPU model 2 - $\\alpha_1, \\alpha_2, \\beta$ = {:.4f}, {:.4f}, {:.4f}".format(alpha_1_optim_QPU_2, alpha_2_optim_QPU_2, beta_optim_QPU_2)) #NEW
plt.plot(depth_tab_more_points, R2d_model_classim, color="steelblue", label="sim model - $\\alpha_1, \\alpha_2, \\beta$ = {:.4f}, {:.4f}, {:.4f}".format(alpha_1_optim_classim, alpha_2_optim_classim, beta_optim_classim))
plt.plot(depth_tab_more_points, R2d_model_classim_measerr, color="darkblue", label="sim model meas+$T_1$ err - $\\alpha_1, \\alpha_2, \\beta$ = {:.4f}, {:.4f}, {:.4f}".format(alpha_1_optim_classim_measerr, alpha_2_optim_classim_measerr, beta_optim_classim_measerr)) #NEW
# classical shadows
plt.errorbar(x=depth_tab, y=short_metrics['all_R2d_mean_diff_n'], yerr=short_metrics['all_R2d_std_diff_n'], ecolor="goldenrod", capsize=3, ls='none', label="QPU shadows 1") #NOTE I have checked the definition of stdv for that case
plt.errorbar(x=depth_tab, y=short_metrics_2['all_R2d_mean_diff_n'], yerr=short_metrics_2['all_R2d_std_diff_n'], ecolor="darkgoldenrod", capsize=3, ls='none', label="QPU shadows 2") #NOTE I have checked the definition of stdv for that case
plt.errorbar(x=depth_tab, y=short_metrics_classim['all_R2d_mean_diff_n'], yerr=short_metrics_classim['all_R2d_std_diff_n'], ecolor="steelblue", capsize=3, ls='none', label="sim shadows") #NOTE I have checked the definition of stdv for that case
plt.errorbar(x=depth_tab, y=short_metrics_classim_measerr['all_R2d_mean_diff_n'], yerr=short_metrics_classim_measerr['all_R2d_std_diff_n'], ecolor="darkblue", capsize=3, ls='none', label="sim shadows meas+$T_1$ err") #NOTE I have checked the definition of stdv for that case
if compute_exact:
    # n=3 density matrix simulation
    plt.plot(depth_tab, short_metrics_exact['all_R2d_diff_n'], color="black", label="density matrix sim")
if show_fully_mixed_state:
    plt.axhline(y=renyi_entropy_from_purity(1/2**num_qubits)/num_qubits, color='black', linestyle='-.', label="maximally mixed state") # n=3
plt.xlabel('Depth', fontsize='12')
plt.xticks(xticks)
plt.ylabel('Renyi-2 entropy density', fontsize='12')
plt.ylim(bottom=0, top=1.1)

plt.legend(fontsize='12')

if save_fig:
    plt.savefig(full_filename_R2d)
plt.show()

# Useful fitting data
print("\n alpha_1_optim_classim = ", alpha_1_optim_classim)
print("\n alpha_2_optim_classim = ", alpha_2_optim_classim)
print("\n beta_optim_classim = ", beta_optim_classim)

print("\n alpha_1_optim_classim_measerr = ", alpha_1_optim_classim_measerr)
print("\n alpha_2_optim_classim_measerr = ", alpha_2_optim_classim_measerr)
print("\n beta_optim_classim_measerr = ", beta_optim_classim_measerr)

print("\n alpha_1_optim_QPU = ", alpha_1_optim_QPU)
print("\n alpha_2_optim_QPU = ", alpha_2_optim_QPU)
print("\n beta_optim_QPU = ", beta_optim_QPU)

print("\n alpha_1_optim_QPU_2 = ", alpha_1_optim_QPU_2)
print("\n alpha_2_optim_QPU_2 = ", alpha_2_optim_QPU_2)
print("\n beta_optim_QPU_2 = ", beta_optim_QPU_2)