"""
Useful functions to implement the classical shadows protocol
"""

# Usedul packages ====================================================================
import numpy as np
import pandas as pd
import time

from qiskit.quantum_info import DensityMatrix
#from qiskit_rigetti import RigettiQCSProvider

from libUtils import ternary_list_to_decimal, renyi_entropy_from_purity, get_dataframe_specific_depth
from libQC import ProtocolParams, apply_Pauli_meas_unit, measure_Zbasis

# Global variables ===================================================================

# --- Build beta values with all possible results for (9 times) norm squared of complex number (minus 4) in expression 
# for purity estimate where keys are the different pairs possible for the two unitaries in the expression and
# elements are the value beta of (9 times) norm squared (minus 4) when the two output states are equal (1st element) or different
# (2nd element)
# N.B. beta_values {'00': [5, -4], '01': [0.5, 0.5], '02': [0.5, 0.5], '10': [0.5, 0.5], '11': [5, -4], '12': [0.5, 0.5], '20': [0.5, 0.5], '21': [0.5, 0.5], '22': [5, -4]}
#NOTE beta in this code is 9*beta - 4 in our article
beta_values = dict()
for i in range(3): #i=0/1/2 corresponds to H/HSdag/Id
    for j in range(3): #j=0/1/2 corresponds to H/HSdag/Id
        pair = str(i)+str(j)
        if i!=j:
            beta_values[pair] = [1/2, 1/2] #the first (resp. second) element corresponds to two identical (resp. different) outcomes
        else:
            beta_values[pair] = [5, -4]#[1, 0] #the first (resp. second) element corresponds to two identical (resp. different) outcomes

# Classes ==========================================================================

class CSParams(ProtocolParams):
    def __init__(self, num_samples, num_groups, M, K, protocol_choice='randomized', artif_randomized=None, noisy_gates=False):
        self.name = 'CS'
        self.num_samples = num_samples # Number of samples for the purity/entropy estimate
        self.num_groups = num_groups #for the median of means
        self.M = M # number of random unitaries or measurement settings
        self.K = K # number of shots per random unitary
        self.protocol_choice = protocol_choice # 'derandomized' or 'randomized' 
        self.artif_randomized = artif_randomized # 'artif1' or 'artif2' or None
        self.noisy_gates = noisy_gates
    @staticmethod
    def from_dict(temp):
        return CSParams(temp['num_samples'], temp['num_groups'], temp['M'], temp['K'], temp['protocol_choice'], temp['artif_randomized'], temp['noisy_gates'])
    

# Functions To Get Measurements ====================================================

def extract_Pauli_shadows(rho_out:DensityMatrix, num_qubits:int, M:int, K:int):
    """ returns a list of classical shadows obtained from applying Pauli measurements on the density matrix rho_out
    Args:
    * rho_out: output state of the quantum circuit of interest (Qiskit DensityMatrix object)
    * num_qubits: width/number of qubits of the quantum circuit of interest
    * M: total number of random unitaries to create randomized measurements on
     the output of the quantum circuit of interest
    * K: number of shots per random unitary m
    Output:
    shadow_list: a list of the form [[array([0,2]), {'01' : 3, '11' : 1}]]

    NOTE function not in use (theoretical use only not practical)
    """
    random_unitaries = [(1/np.sqrt(2)) * np.matrix([[1, 1], [1, -1]]), (1/np.sqrt(2)) * np.matrix([[1, -1j], [1, +1j]]), np.matrix([[1, 0], [0, 1]])]
    shadow_list = []
    for _ in range(M):
        random_unitary_index = np.random.randint(0, 3, size=num_qubits) # draw n single-qubit random unitaries
        random_unitary_size_n = random_unitaries [random_unitary_index[0]]
        for n in range(1, num_qubits):
            random_unitary_size_n = np.kron(random_unitary_size_n, random_unitaries[random_unitary_index[n]])
        rho_out_random = DensityMatrix(random_unitary_size_n @ rho_out.data @ random_unitary_size_n.getH()) # apply random unitary to example quantum state
        counts = rho_out_random.sample_counts(shots=K) # - draw K samples
        shadow_list.append([random_unitary_index, counts])
    print("Shadows extracted successfully")
    return(shadow_list)

def extract_Pauli_shadows_circuit(qc, M, K, num_qubits, backend, initial_layout, CS_protocol_choice, verbose=False):
    """
    This function fills in and returns a list shadow_list with
    1) the description of a random unitary (for the randomized measurement performed)
    2) the output string obtained for each of those repeated measurements
    N.B.: derand_protocol is a boolean that is True if the de-randomized version of the classical shadows protocol is used
    else it is the standard version of the protocol that is used
    """
    if CS_protocol_choice == 'derandomized':
        M_max = 3**num_qubits
    qc_copy = qc.copy("qc_copy") #to reuse the quantum circuit without the random unitary later
    shadow_list = []
    for meas_setting_number in range(M):
        if CS_protocol_choice == 'derandomized':
            meas_setting_number = meas_setting_number % M_max
            # Convert measurement setting number to corresponding unit index (ternary)
            ternary_description = np.base_repr(meas_setting_number, base=3, padding=0)#now a string
            unit_index = [0 for _ in range(num_qubits - len(ternary_description))] + [int(i) for i in ternary_description] #now a list of ints
        elif CS_protocol_choice == 'randomized': # Apply random unitary to parameterised quantum circuit
            unit_index = np.random.randint(3, size=num_qubits) # description of random unitary to be applied to output state before measurement (0 = X meas, 1 = Y meas, 2 = Z meas)
        
        qc = apply_Pauli_meas_unit(qc, num_qubits, unit_index, backend)
        if verbose: print("\n 1. Unit index: ", unit_index)

        #NOTE layer of identities to model T1 noise from the measurement
        for qubit in range(num_qubits):
            qc.id(qubit)
        # Perform K measurements (and store outcomes/counts in a dictionary)
        counts = measure_Zbasis (qc, K, backend, initial_layout, verbose=verbose)
        shadow_list.append([unit_index, counts])
        if verbose: print('\n 2. Quantum circuit qc before removing measurement and random unitary : \n \n', qc)
        qc = qc_copy.copy("qc") #quantum circuit before the random unitary was applied
        if verbose: print('Quantum circuit qc after removing measurement and random unitary : \n \n', qc)
    return (shadow_list, qc)

def get_circuit_measurements_per_depth_CS(depth, num_qubits, qc, protocol_params, backend, backend_params, fullfilename, verbose=False):
    """
    Gets and saves to a csv file the measurement outcomes of a set of random Pauli basis measurements applied to a noisy circuit
    with K shots per random measurement and M random measurements. This set of measurements is run num_samples times.
    """

    for _ in range(protocol_params.num_samples):
        # Extract Pauli shadows
        shadow_full_list, qc = extract_Pauli_shadows_circuit(qc, protocol_params.M, protocol_params.K, num_qubits, backend, backend_params.initial_layout, protocol_params.protocol_choice, verbose=verbose)
        print("Pauli shadows extracted successfully.")
        # Save results to csv file
        df = pd.DataFrame([depth] + shadow_full_list).T
        df.to_csv(fullfilename, mode='a', index=False, header=False)
        print("Data saved to csv successfully.")


# Functions To Get Metrics ====================================================

def extract_shadow_element_from_df(original_df, wanted_depth_index, sample_index, measurement_setting_index, verbose=False):
    df_fixed_depth = get_dataframe_specific_depth(original_df, wanted_depth_index)  
    df_fixed_depth_sample = df_fixed_depth.iloc[sample_index]
    if verbose:
        print("\n WANTED DEPTH INDEX: ", wanted_depth_index)
        print("\n WANTED MEASUREMENT SETTING INDEX: ", measurement_setting_index)
        # print("\n df_fixed_depth_sample: \n", df_fixed_depth_sample)
        
    shadow_element = df_fixed_depth_sample.T[1+measurement_setting_index] #the first column is not shadows (depth_index)
    if verbose:
        print("checking shadow element to see if as expected \n", shadow_element)
    shadow_element = eval(shadow_element.replace("array", "np.array"))
    if verbose:
        print("final shadow element \n", shadow_element)
    return(shadow_element)

def extract_shadows_from_df(df, depth_index, sample_index, M, verbose=False):
    shadows = np.zeros(M, dtype=object)
    for m in range(M):
        shadows[m] = extract_shadow_element_from_df(df, depth_index, sample_index, m, verbose=verbose)
    return (shadows)

def get_artif1_randomized_shadow_full_list(derand_shadow_full_list, M, num_qubits):
    """
    Returns a shadow_full_list for any number M of random unitaries (or measurement settings) obtained from the derandomized shadow_full_list (with 3**num_qubits measurement settings) 
    by copy pasting each given derand shadow the number of times the measurement setting is repeated in the new shadow_full_list.
    """
    shadow_full_list = np.zeros(M, dtype=object)
    for i in range(M):
        unit_index = np.random.randint(3, size=num_qubits) # description of random unitary to be applied to output state before measurement (0 = X meas, 1 = Y meas, 2 = Z meas)
        index = ternary_list_to_decimal(unit_index) # convert unit_index to index
        corresponding_shadow = derand_shadow_full_list[index]
        counts = corresponding_shadow[1]
        shadow_full_list[i]=[unit_index, counts]
    return shadow_full_list

#print(get_artif1_randomized_shadow_full_list([[[0],{'0':1, '1':6}],[[1],{'0':5, '1':2}],[[2],{'0':0, '1':7}]], 5, 1))

def get_artif2_randomized_shadow_full_list(derand_shadow_full_list_sample_1, derand_shadow_full_list_sample_2, derand_shadow_full_list_sample_3, M, num_qubits, verbose=False):
    """
    Returns a shadow_full_list for any number M of random unitaries (or measurement settings) obtained from the derandomized shadow_full_list (with 3**num_qubits measurement settings) 
    by taking each given derand shadow the number of times the measurement setting is repeated in the new shadow_full_list (exploiting the 3 different samples for that).
    """
    shadow_full_list = np.zeros(M, dtype=object)
    num_occurences = np.zeros(3**num_qubits) # number of occurences of each measurement setting
    for i in range(M):
        unit_index = np.random.randint(3, size=num_qubits) # description of random unitary to be applied to output state before measurement (0 = X meas, 1 = Y meas, 2 = Z meas)
        index = ternary_list_to_decimal(unit_index) # convert unit_index to index
        nb_occurences = num_occurences[index] #so far
        if nb_occurences%3 == 0:
            corresponding_shadow = derand_shadow_full_list_sample_1[index]
        elif nb_occurences%3 == 1:
            corresponding_shadow = derand_shadow_full_list_sample_2[index]
        elif nb_occurences%3 == 2:
            corresponding_shadow = derand_shadow_full_list_sample_3[index]
        counts = corresponding_shadow[1]
        shadow_full_list[i]=[unit_index, counts] #np.array([unit_index, counts])
        num_occurences[index] = nb_occurences + 1
    if verbose: print("\n CHECK num_occurences: ", num_occurences)
    return shadow_full_list

# print(get_artif2_randomized_shadow_full_list([[[0],{'0':1, '1':6}],[[1],{'0':5, '1':2}],[[2],{'0':0, '1':7}]],
#                                             [[[0],{'0':2, '1':5}],[[1],{'0':6, '1':1}],[[2],{'0':1, '1':6}]], 
#                                             [[[0],{'0':0, '1':7}],[[1],{'0':4, '1':3}],[[2],{'0':2, '1':5}]], 5, 1, verbose=True))

def estimate_purity_from_Pauli_shadows(shadow_full_list, num_qubits:int, CS_params:CSParams, circuit_bool:bool, verbose=False):
    """with median of means (MOM)
    \n Args:\n
    * shadow_full_list: a list with elements of the form [array([0,2]), {'01' : 3, '11' : 1}]
    * num_qubits: width/number of qubits of the quantum circuit of interest
    * M: number of randomized measurement settings
    * K: number of shots per randomized measurement setting
    * num_groups: shadows are grouped into num_groups groups before computing an average over all
     measurement settings and shots (purity estimate) for each of those groups then taking the
     median of those averages; /!\ num_groups does not have to divide M eventhough if satisfied
     this makes the code more optimised. However, num_groups has to be in the interval [1:M/2 + 1]
    * circuit_bool: True if the shadow list was obtained from a quantum circuit, False if it was obtained from a density matrix

    \n Output:\n
    * purity: a purity estimate built from classical shadows stored in shadow_full_list
    For one means (over M different measurement settings) of the median of means:
    This purity estimate, called purity here, is given by the formula:

    purity = ( 2/ (M*(M-1)) ) * sum_{m1} sum_{m2 such that m2 < m1} ( 1/(K^2) ) sum_{m_counts_item_1, m_counts_item_2} (nb_times_outcome_1) * (nb_times_outcome_2) * prod_{n} (9 * beta - 4)

    where beta = beta(m_description_1, m_description_2, outcome_1[n], outcome_2[n]) is an elememt of the dictionary beta_values (global variable) 

    """
    global beta_values

    K_factor = 1/(CS_params.K**2)
    M_subgroup = int(CS_params.M/CS_params.num_groups) #for the median of means (number of measurement settings per subgroup of the M shadows)
    M_factor = (2/(M_subgroup * (M_subgroup - 1)))
    if verbose: print("\n M_subgroup : ", M_subgroup)
    means = []
    for group_number in range (CS_params.num_groups):
        if verbose: print("\n ~~~~~~~~~ \n Subgroup considered is #", group_number)
        shadow_list = shadow_full_list[group_number*M_subgroup:(group_number+1)*M_subgroup] #an array of the form array([array([0,2]), {'01' : 3, '11' : 1}])
        if verbose: print("\n Sublist of shadows considered is : \n", shadow_list)
        purity = 0
        start_one_group = time.time()
        for m1 in range(M_subgroup):
            if verbose: print("========================================================")
            if verbose: print("Random unitary index m1 = ", m1)
            m_description_1 = shadow_list[m1][0] # array([0,2])
            m_counts_1 = shadow_list[m1][1] # {'01' : 3, '11' : 1}
            for m2 in range(m1):
                m_description_2 = shadow_list[m2][0] #an array
                m_counts_2 = shadow_list[m2][1]
                trace_prod_shadows = 0 # this is sum_{m1,m2} Tr[\rho^{(m1)}\rho^{(m2)}]
                beta_items = np.zeros(num_qubits, dtype=object)
                for n in range(num_qubits):
                    beta_items[n] = beta_values[str(m_description_1[n])+str(m_description_2[n])]
                for m_counts_item_1 in m_counts_1.items(): #loop over measurement outcomes
                    outcome_1 = m_counts_item_1[0]
                    nb_times_outcome_1 = m_counts_item_1[1]
                    for m_counts_item_2 in m_counts_2.items(): #loop over measurement outcomes
                        outcome_2 = m_counts_item_2[0]
                        nb_times_outcome_2 = m_counts_item_2[1]
                        prod_over_num_qubits = 1
                        for n in range(num_qubits):
                            t = num_qubits-1-n
                            #first deduce which value of beta to use:
                            if circuit_bool: # case where shadows were obtained from a quantum circuit
                                nth_bit_from_outcome_1 = outcome_1[t] #small change here for circuits
                                nth_bit_from_outcome_2 = outcome_2[t] #small change here for circuits
                            else: # case where shadows were obtained from a density matrix
                                nth_bit_from_outcome_1 = outcome_1[n]
                                nth_bit_from_outcome_2 = outcome_2[n]
                            # beta_item = beta_items[n]#beta_values[str(m_description_1[n])+str(m_description_2[n])] #based on single-qubit random unitaries
                            if nth_bit_from_outcome_1 == nth_bit_from_outcome_2:
                                beta = beta_items[n][0] #based on measurement outcome
                            else:
                                beta = beta_items[n][1] #based on measurement outcome
                            #then compute the value of the product element:
                            prod_over_num_qubits *= beta
                        if verbose: print("\n Product of interest :", prod_over_num_qubits)
                        trace_prod_shadows += nb_times_outcome_1 * nb_times_outcome_2 * prod_over_num_qubits
                purity += (K_factor)*trace_prod_shadows
        if verbose: print("TIME ===== one group loop iteration : ", time.time() - start_one_group)
        purity *= M_factor
        if verbose: print("\n The purity of this subgroup is : ", purity)
        means.append(purity)
    mom = np.median(means)
    return(mom)

def compute_metrics_per_depth_CS (df, depth, num_qubits, CS_params, verbose=False):
    """
    Returns purity metric.

    The purity is obtained using so-called classical shadows protocol, with K shots per random measurement and M random measurements.
    Shadows are grouped into num_groups to compute the median of means; and the CS protocol is run num_samples times to get error bars.
    
    NOTE if CS_params.artif_randomized == 'artif1' or CS_params.artif_randomized == 'artif2', then the csv file considered has
    to be obtained from the derandomized protocol over M=3**num_qubits meas settings and with num_samples=3; moreover num_samples of the
      resulting metrics is fixed to 3 in the first case and 1 in the second case
    """
    pur_samples, R2d_samples = [], []

    if CS_params.artif_randomized == 'artif2':
        temp_shadow_full_list_sample_1 = extract_shadows_from_df(df, depth, 0, 27, verbose=verbose)
        temp_shadow_full_list_sample_2 = extract_shadows_from_df(df, depth, 1, 27, verbose=verbose)
        temp_shadow_full_list_sample_3 = extract_shadows_from_df(df, depth, 2, 27, verbose=verbose)
        
        shadow_full_list = get_artif2_randomized_shadow_full_list(temp_shadow_full_list_sample_1, temp_shadow_full_list_sample_2, temp_shadow_full_list_sample_3, CS_params.M, num_qubits, verbose=verbose)

        # Estimate metrics from the classical shadows
        start_pur = time.time()
        pur = estimate_purity_from_Pauli_shadows(shadow_full_list, num_qubits, CS_params, True)
        if verbose:
            print("this is the purity : ", pur)
            print("TIME ===== Purity calculation: ", time.time() - start_pur)

        R2d = renyi_entropy_from_purity(pur)/num_qubits
        if verbose: print("Estimate computed successfully.")
        
        # Add estimates to lists of samples
        pur_samples.append(pur)
        R2d_samples.append(R2d)
    else:
        for sample_index in range(CS_params.num_samples):
            # Extract Pauli shadows
            # start_extract_shadows = time.time()
            if CS_params.artif_randomized == 'artif1':
                temp_shadow_full_list = extract_shadows_from_df(df, depth, sample_index, 27, verbose=verbose)

                shadow_full_list = get_artif1_randomized_shadow_full_list(temp_shadow_full_list, CS_params.M, num_qubits)

            else:
                shadow_full_list = extract_shadows_from_df(df, depth, sample_index, CS_params.M, verbose=verbose)

            # print("\n TIME ===== Extracting shadows: ", time.time() - start_extract_shadows)
            print("Pauli shadows read successfully.")            
            
            # Estimate metrics from the classical shadows
            start_pur = time.time()
            pur = estimate_purity_from_Pauli_shadows(shadow_full_list, num_qubits, CS_params, True)
            if verbose: 
                print("this is the purity : ", pur)
                print("TIME ===== Purity calculation: ", time.time() - start_pur)

            R2d = renyi_entropy_from_purity(pur)/num_qubits
            print("Estimate computed successfully.")
            
            pur_samples.append(pur)
            R2d_samples.append(R2d)

    return pur_samples, R2d_samples
