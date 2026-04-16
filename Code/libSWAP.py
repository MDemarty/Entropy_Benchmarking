"""
This library contains the functions used to estimate the purity of a state using the SWAP test
"""

import pandas as pd
import numpy as np

from qiskit import QuantumRegister, QuantumCircuit

from libUtils import renyi_entropy_from_purity, bitwise_AND, parity_bit, get_dataframe_specific_depth
from libQC import ProtocolParams, measure_Zbasis, add_native_ionq_bell_meas

# Classes ==========================================================================
class SWAPParams(ProtocolParams):
    def __init__(self, num_samples, num_measurements, num_groups=1, noisy_gates=False):
        self.name = 'SWAP'
        self.num_samples = num_samples # Number of samples for the purity/entropy estimate
        self.num_measurements = num_measurements #10**4 (resp. 10**6) for one (resp. two) correct digits in the fractional part of the purity estimate
        self.num_groups = num_groups
        self.num_measurements_per_group = int(num_measurements/num_groups)
        self.noisy_gates = noisy_gates
    @staticmethod
    def from_dict(temp):
        return SWAPParams(temp['num_samples'], temp['num_measurements'], temp['num_groups'], temp['noisy_gates'])

# Functions To Get Measurements ====================================================

def concatenate_swap_circuit(qc, circuit_choice, num_qubits):
    # Creating the SWAP test circuit used to estimate the purity of the above circuit output
    swap_test_qr1 = QuantumRegister(num_qubits, 'q1')
    swap_test_qr2 = QuantumRegister(num_qubits, 'q2')
    swap_test_circuit = QuantumCircuit(swap_test_qr1, swap_test_qr2, name="circuit + SWAP Circuit")

    qubit_index_list = [i for i in range(0, 2*num_qubits)]
    swap_test_circuit.compose(qc, qubits=qubit_index_list[0:num_qubits], inplace=True)
    swap_test_circuit.compose(qc, qubits=qubit_index_list[num_qubits:2*num_qubits], inplace=True)

    # SWAP test part of the circuit
    for i in range(num_qubits):
        if circuit_choice == 'HEA_IONQ':
            swap_test_circuit = add_native_ionq_bell_meas(swap_test_circuit, i, i+num_qubits)
        else:    
            swap_test_circuit.cx(i, i+num_qubits)
            swap_test_circuit.h(i)

    # print("full circuit : \n", swap_test_circuit)
    return swap_test_circuit

def get_circuit_measurements_per_depth_SWAP(depth, num_qubits, qc, circuit_params, protocol_params, backend, backend_params, fullfilename, verbose=False):
    """
    Gets and saves to a csv file the measurement outcomes of a SWAP test measurement applied to a noisy circuit
    with num_measurements shots. This set of measurements is run num_samples times.
    """
    swap_test_circuit = concatenate_swap_circuit(qc, circuit_params.choice, num_qubits)

    for _ in range(protocol_params.num_samples):
        IONQ =  backend_params.type == 'IonQ_sim' or backend_params.type == 'IonQ_QPU'
        df_line = [depth]
        for _ in range(protocol_params.num_groups):
            counts = measure_Zbasis(swap_test_circuit, protocol_params.num_measurements_per_group, backend, backend_params.initial_layout, False, IONQ)
            df_line.append(counts)
        # print("\n full circuit which should be unchanged after execution: \n", swap_test_circuit)

        # Save results to csv file
        df = pd.DataFrame(df_line).T
        df.to_csv(fullfilename, mode='a', index=False, header=False)
        print("Data saved successfully.")

# Functions To Get Metrics ====================================================

def extract_swap_counts_from_df(original_df, wanted_depth_index, sample_index, i):
    df_fixed_depth = get_dataframe_specific_depth(original_df, wanted_depth_index)
    df_fixed_depth_sample = df_fixed_depth.iloc[sample_index]
    swap_counts = df_fixed_depth_sample.T[1+i] #the first column is not counts (depth_index)
    return(eval(swap_counts))

def swap_test_outcome (outcome):
    """
    returns the parity bit of the bitwise AND of the two halves of the string outcome
    """
    num_qubits = int(len(outcome)/2)
    result = parity_bit(bitwise_AND(outcome[0:num_qubits], outcome[num_qubits:2*num_qubits]))
    return (result)

def estimate_purity_from_swap_test (counts):
    """
    returns the purity estimate from the SWAP test circuit measurements outcomes counts
    """
    nb_outcome_0 = 0
    nb_outcome_1 = 0
    for outcome, nb_times_outcome in counts.items():
        if swap_test_outcome(outcome) == 0:
            nb_outcome_0 += nb_times_outcome
        else:
            nb_outcome_1 += nb_times_outcome
    nb_measurements = nb_outcome_0 + nb_outcome_1
    pur_estimate = 2*nb_outcome_0/nb_measurements - 1
    return (pur_estimate)

def compute_metrics_per_depth_SWAP (df, depth, num_qubits, protocol_params, verbose=False):
    pur_samples, R2d_samples = [], []

    for sample_index in range(protocol_params.num_samples):
        pur_means = []
        for i in range(protocol_params.num_groups):
            
            counts = extract_swap_counts_from_df(df, depth, sample_index, i)
            # Purity & Renyi density estimates
            pur_estimate = estimate_purity_from_swap_test(counts)
            pur_means.append(pur_estimate)

        pur_mom = np.median(pur_means)
        R2d_mom = renyi_entropy_from_purity(pur_mom)/num_qubits
        
        #print("\n Purity estimate from SWAP test circuit : ", pur_estimate)
        pur_samples.append(pur_mom)
        R2d_samples.append(R2d_mom)

    return pur_samples, R2d_samples

