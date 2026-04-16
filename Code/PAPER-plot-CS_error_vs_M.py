"""
PAPER

Plotting the CS estimate over exact for different number of measurement settings M values
"""

import argparse
import matplotlib.pyplot as plt
import os

from libUtils import get_metrics_specific_width, renyi_entropy_from_purity
from libPlot import compute_xticks, compute_depth_tab, compute_filename, plot_params
from libExperiment import ExperimentParams
from libIO import load_from_json

def build_parser (description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("-v", "--verbose", action="store_true", help="Enables verbose mode")
    parser.add_argument("-i1", "--input1", type=str, required=False, default='results/Aer_sim/CS_HEA_RIGETTI/experiment_2025-06-21_n3-3_D15_M50.json', help="Json file for HEA_RIGETTI - CS - without measurement error and meas circuit errors - M=50")
    parser.add_argument("-i2", "--input2", type=str, required=False, default='results/Aer_sim/CS_HEA_RIGETTI/experiment_2025-06-21_n3-3_D15_M350.json', help="Json file for HEA_RIGETTI - CS - without measurement error and meas circuit errors - M=350")
    parser.add_argument("-i3", "--input3", type=str, required=False, default='results/Aer_sim/CS_HEA_RIGETTI/experiment_2025-06-22_n3-3_D15_M800.json', help="Json file for HEA_RIGETTI - CS - without measurement error and meas circuit errors - M=800")
    parser.add_argument("-i4", "--input4", type=str, required=False, default='results/Aer_sim/CS_HEA_RIGETTI/experiment_2025-06-22_n3-3_D15_M1000.json', help="Json file for HEA_RIGETTI - CS - without measurement error and meas circuit errors - M=1000")
    
    parser.add_argument("-i5", "--input5", type=str, required=False, default='results/Aer_sim/DensMat_HEA_RIGETTI/experiment_2025-03-18_n3-3_D15.json', help="Json file for DensMat")
    parser.add_argument("-o", "--output", type=str, required=False, default='Paper/CS_params', help="Folder where to store the results")

    return parser

parser = build_parser('Plotting CS estimate for different M values (for PAPER)')
args = parser.parse_args()
verbose = args.verbose
jsonfilename1 = args.input1
jsonfilename2 = args.input2
jsonfilename3 = args.input3
jsonfilename4 = args.input4
jsonfilename5 = args.input5
resultdir = args.output

compute_exact = True #Exact purity/entropy #NOTE keep as True
save_fig = True
show_fully_mixed_state = True

# ========================= Experiment1 (HEA_RIGETTI - Classical Shadows - without measurement error and meas circuit errors)) =============================
experiment1 = ExperimentParams.from_dict(load_from_json(jsonfilename1))
if experiment1 == None:
    print ("ERROR: reading json file, no experiment #1 can be loaded")
    exit()

circuit_params = experiment1.circuit_params
noise_params = experiment1.noise_params
CS_params = experiment1.protocol_params
M1 = CS_params.M
# CS_params = CSParams(num_samples=3, num_groups=3, M=729, K=100, protocol_choice='randomized')

num_qubits = circuit_params.num_qubits_min

# Get metrics
metrics_CS = load_from_json(experiment1.metrics_file)
short_metrics_CS = get_metrics_specific_width(metrics_CS, num_qubits, num_qubits)

# ========================= Experiment2 (HEA_RIGETTI - SWAP - without measurement error and meas circuit errors)) =============================
experiment2 = ExperimentParams.from_dict(load_from_json(jsonfilename2))
if experiment2 == None:
    print ("ERROR: reading json file, no experiment #2 can be loaded")
    exit()

M2 = experiment2.protocol_params.M
# Get metrics
metrics_CS_2 = load_from_json(experiment2.metrics_file)
short_metrics_CS_2 = get_metrics_specific_width(metrics_CS_2, num_qubits, num_qubits)

# ========================= Experiment3 (HEA_RIGETTI - SWAP - without measurement error and meas circuit errors)) =============================
experiment3 = ExperimentParams.from_dict(load_from_json(jsonfilename3))
if experiment3 == None:
    print ("ERROR: reading json file, no experiment #3 can be loaded")
    exit()

M3 = experiment3.protocol_params.M
# Get metrics
metrics_CS_3 = load_from_json(experiment3.metrics_file)
short_metrics_CS_3 = get_metrics_specific_width(metrics_CS_3, num_qubits, num_qubits)

# ========================= Experiment4 (HEA_RIGETTI - SWAP - without measurement error and meas circuit errors)) =============================
experiment4 = ExperimentParams.from_dict(load_from_json(jsonfilename4))
if experiment4 == None:
    print ("ERROR: reading json file, no experiment #4 can be loaded")
    exit()

M4 = experiment4.protocol_params.M
# Get metrics
metrics_CS_4 = load_from_json(experiment4.metrics_file)
short_metrics_CS_4 = get_metrics_specific_width(metrics_CS_4, num_qubits, num_qubits)

# ========================= Experiment5 (HEA_RIGETTI - DensMat) =============================
if compute_exact:
    experiment5 = ExperimentParams.from_dict(load_from_json(jsonfilename5))
    if experiment5 == None:
        print ("ERROR: reading json file, no experiment #5 can be loaded")
        exit()

    # Get metrics
    metrics_exact = load_from_json(experiment5.metrics_file)
    short_metrics_exact = get_metrics_specific_width(metrics_exact, num_qubits, num_qubits)

"""
Plots
"""
if save_fig:
    # Prepare directory - check if directory exists, if not create it
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

filename = 'n%d_M%d_K%d_grps%d_spls%d.pdf' % (num_qubits, CS_params.M, CS_params.K, CS_params.num_groups, CS_params.num_samples)

depth_tab = compute_depth_tab(circuit_params.depth_min, circuit_params.depth_max, circuit_params.depth_step)
xticks = compute_xticks(depth_tab, circuit_params.depth_step)

full_filename_puri = compute_filename (resultdir, 'pur', filename)
full_filename_R2d = compute_filename (resultdir, 'R2d', filename)

# Purity
plt.figure()
plt.errorbar(x=depth_tab, y=short_metrics_CS['all_pur_mean_diff_n'], yerr=short_metrics_CS['all_pur_std_diff_n'], label="M=%d" %(M1), fmt='.')#capsize=3, ls='none') #Aer Sim CS
plt.errorbar(x=depth_tab, y=short_metrics_CS_2['all_pur_mean_diff_n'], yerr=short_metrics_CS_2['all_pur_std_diff_n'], label="M=%d" %(M2), fmt='.')#capsize=3, ls='none') #Aer Sim CS 2
plt.errorbar(x=depth_tab, y=short_metrics_CS_3['all_pur_mean_diff_n'], yerr=short_metrics_CS_3['all_pur_std_diff_n'], label="M=%d" %(M3), fmt='.') #capsize=3, ls='none') #Aer Sim CS 3
plt.errorbar(x=depth_tab, y=short_metrics_CS_4['all_pur_mean_diff_n'], yerr=short_metrics_CS_4['all_pur_std_diff_n'], label="M=%d" %(M4), fmt='.') #capsize=3, ls='none') #Aer Sim CS 4

if compute_exact:
    plt.plot(depth_tab, short_metrics_exact['all_pur_diff_n'], label="density matrix sim", color="black")
    #plt.plot(depth_tab, pur_exact_SWAP, color="salmon")
if show_fully_mixed_state:
    plt.axhline(y=1/2**num_qubits, label="maximally mixed state", color='black', linestyle='-.')
plot_params('Depth', 'Purity', xticks, 1.1, save_fig, full_filename_puri)
plt.show()

# Renyi-2 entropy density
plt.figure()
plt.errorbar(x=depth_tab, y=short_metrics_CS['all_R2d_mean_diff_n'], yerr=short_metrics_CS['all_R2d_std_diff_n'], label="M=%d" %(M1), fmt='.') #capsize=3, ls='none') #Aer Sim CS
plt.errorbar(x=depth_tab, y=short_metrics_CS_2['all_R2d_mean_diff_n'], yerr=short_metrics_CS_2['all_R2d_std_diff_n'], label="M=%d" %(M2), fmt='.') #capsize=3, ls='none') #Aer Sim CS 2
plt.errorbar(x=depth_tab, y=short_metrics_CS_3['all_R2d_mean_diff_n'], yerr=short_metrics_CS_3['all_R2d_std_diff_n'], label="M=%d" %(M3), fmt='.') #capsize=3, ls='none') #Aer Sim CS 2
plt.errorbar(x=depth_tab, y=short_metrics_CS_4['all_R2d_mean_diff_n'], yerr=short_metrics_CS_4['all_R2d_std_diff_n'], label="M=%d" %(M4), fmt='.') #capsize=3, ls='none') #Aer Sim CS 2

if compute_exact:
    plt.plot(depth_tab, short_metrics_exact['all_R2d_diff_n'], label="density matrix sim", color="black")
    #plt.plot(depth_tab, short_metrics_exact['all_R2d_diff_n'], color="salmon")
if show_fully_mixed_state:
    plt.axhline(y=renyi_entropy_from_purity(1/2**num_qubits)/num_qubits, label="maximally mixed state", color='black', linestyle='-.')
plot_params('Depth', 'Renyi-2 entropy density', xticks, 1.1, save_fig, full_filename_R2d)
plt.show()