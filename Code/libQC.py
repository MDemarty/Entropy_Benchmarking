"""
Useful functions for general quantum computing tasks
"""

import numpy as np
import time
import os

from qiskit import QuantumCircuit
import qiskit_aer.noise as noise
from qiskit_aer import AerSimulator
from qiskit import execute

from qiskit_ionq import IonQProvider
from qiskit_ionq import ErrorMitigation
from qiskit_ionq import GPIGate, GPI2Gate, MSGate #IonQ native gates
from qiskit_rigetti import RigettiQCSProvider

from libUtils import create_cycle_graph

# Specific classes ===================================================================

class BackendParams:
    def __init__(self, type, initial_layout=None):
        self.type = type # 'Aer_sim' or 'Rigetti_QPU' or 'IonQ_sim' or 'IonQ_QPU' 
        self.initial_layout = initial_layout #a list of qubit indices (e.g., [100, 101, 102])
    @staticmethod
    def from_dict(temp):
        return BackendParams(temp['type'], temp['initial_layout'])

class AerSimParams(BackendParams):
    def __init__(self, type, noise_params):
        super().__init__(type, initial_layout=None)
        self.noise_params = noise_params

class CircuitParams:
    def __init__(self, choice, num_qubits_min, num_qubits_max, num_qubits_step=1, depth_min=0, depth_max=15, depth_step=1, rx_only=False, angles=None):
        self.choice = choice 
        self.num_qubits_min = num_qubits_min
        self.num_qubits_max = num_qubits_max
        self.num_qubits_step = num_qubits_step
        self.num_circuits = self.num_qubits_max - self.num_qubits_min + 1
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_step = depth_step
        self.num_layers = self.depth_max
        self.rx_only = rx_only # VQC with only RX and CZ gates (no RY gates)
        if angles:
            self.circuits_angles = angles
        else: 
            self.init_angles()
    def init_angles(self):
        self.circuits_angles = np.zeros(self.num_circuits, dtype=object)
        for i in range(self.num_circuits):
            num_qubits = self.num_qubits_min+self.num_qubits_step*i
            if self.rx_only:
                num_angles_per_layer = num_qubits
            else:
                num_angles_per_layer = 2*num_qubits
            angles = 2*np.pi*np.random.rand(num_angles_per_layer*self.num_layers) #dim=(1, angles_of_circuit)
            self.circuits_angles[i] = angles.reshape((self.num_layers, num_angles_per_layer)) #dim=(layers, angles_of_layer)

    def to_dict(self):
        temp = dict()
        temp['choice'] = self.choice
        temp['num_qubits_min'] = self.num_qubits_min
        temp['num_qubits_max'] = self.num_qubits_max
        temp['num_qubits_step'] = self.num_qubits_step
        temp['num_circuits'] = self.num_circuits
        temp['depth_min'] = self.depth_min
        temp['depth_max'] = self.depth_max
        temp['depth_step'] = self.depth_step
        temp['num_layers'] = self.num_layers
        temp['rx_only'] = self.rx_only
        temp['angles'] = dict()
        for i in range(self.num_circuits):
            temp['angles'][str(i)] = self.circuits_angles[i].tolist()

        return temp
    
    @staticmethod
    def from_dict(params_dict):
        circuit_params = CircuitParams(params_dict['choice'], params_dict['num_qubits_min'], params_dict['num_qubits_max'], params_dict['num_qubits_step'],
                                       params_dict['depth_min'], params_dict['depth_max'], params_dict['depth_step'], params_dict['rx_only'])
        circuit_params.circuits_angles = np.zeros(circuit_params.num_circuits, dtype=object)
        for i in range(circuit_params.num_circuits):
           circuit_params.circuits_angles[i] = np.array(params_dict['angles'][str(i)])
        return circuit_params

# a_circuit = CircuitParams("VQA", num_qubits_min=1, num_qubits_max=2, depth_min=0, depth_max=2)
# print("angles : ", a_circuit.circuits_angles)

class CircuitParams5Qbits (CircuitParams):
    def __init__(self, choice, depth_min=0, depth_max=15, depth_step=1, rx_only=False, angles=None):
        self.choice = choice 
        self.num_qubits_min = 5
        self.num_qubits_max = 5
        self.num_qubits_step = 1
        self.num_circuits = self.num_qubits_max - self.num_qubits_min + 1
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_step = depth_step
        self.num_layers = self.depth_max
        self.rx_only = rx_only
        if angles:
            self.circuits_angles = angles
        else: 
            self.init_angles()

class CircuitParams3Qbits (CircuitParams):
    def __init__(self, choice, depth_min=0, depth_max=15, depth_step=1, rx_only=False, angles=None):
        self.choice = choice 
        self.num_qubits_min = 3
        self.num_qubits_max = 3
        self.num_qubits_step = 1
        self.num_circuits = self.num_qubits_max - self.num_qubits_min + 1
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.depth_step = depth_step
        self.num_layers = self.depth_max
        self.rx_only = rx_only
        if angles:
            self.circuits_angles = angles
        else: 
            self.init_angles()

class NoiseParams:
    """
    Noise model, 
    for single-qubit depolarising probability p_DP1
    and two-qubit depolarising probability p_DP2,
    and single-qubit amplitude damping probability p_AD1,
    and two-qubit amplitude damping probability p_AD2,
    and single-qubit amplitude damping probability p_ADmeas (T1 noise from measurement),
    and readout error probability p_meas in this form [[0.9, 0.1],[0.25,0.75]] i.e. [[P(0|0), P(1|0)],[P(0|1), P(1|1)]] 
    where P(n|m) is the probability of measuring n given that the ideal (without readout noise) outcome is m
    """
    def __init__(self, p_DP1 = 0.008, p_DP2=0.054, p_AD1=0, p_AD2=0, p_ADmeas=0, p_meas=[[1, 0], [0, 1]]):
        self.p_DP1 = p_DP1 # 1-qubit gate
        self.p_DP2 = p_DP2 # 2-qubit gate
        self.p_AD1 = p_AD1
        self.p_AD2 = p_AD2
        self.p_ADmeas = p_ADmeas # T1 noise from measurement #0.05
        self.p_meas = p_meas #readout error probabilities
    @staticmethod
    def from_dict(temp):
        return NoiseParams(temp['p_DP1'], temp['p_DP2'], temp['p_AD1'], temp['p_AD2'], temp['p_ADmeas'], temp['p_meas'])

class ProtocolParams:
    def __init__(self, name, num_samples=3, noisy_gates=False):
        self.name = name
        self.num_samples = num_samples # Number of samples for the purity/entropy estimate
        self.noisy_gates = noisy_gates #True if the protocol includes noise in the measurement circuit

# Functions ===================================================================

# Native gates decomposition and definition
def add_native_ionq_bell_meas(qc, qubit1, qubit2):
    """
    IonQ native gates decomposition for a Bell measurement
    """
    qc.append(GPI2Gate(0.25), [qubit1])
    qc.append(MSGate(0,0), [qubit1, qubit2])
    qc.append(GPI2Gate(0), [qubit1])
    qc.append(GPI2Gate(0.5), [qubit2])
    return qc

def add_native_ionq_ybasis_meas_unit(qc, qubit):
    """
    IonQ native gates decomposition for a Y-basis measurement
    """
    qc.append(GPI2Gate(0.5), [qubit])
    return qc 

def add_native_ionq_ybasis_meas(qc, qubit):
    """
    IonQ native gates decomposition for a Y-basis measurement
    """
    qc.append(GPI2Gate(0.5), [qubit])

    # qc.append(GPIGate(0), [qubit])
    # qc.append(GPIGate(-0.125), [qubit])
    # qc.append(GPI2Gate(0.25), [qubit])
    # qc.append(GPIGate(0), [qubit])
    #print("qc \n", qc)
    return qc 

def add_native_ionq_xbasis_meas_unit(qc, qubit):
    """
    IonQ native gates decomposition for an X-basis measurement
    """
    qc.append(GPI2Gate(0.75), [qubit])
    return qc

def add_native_ionq_xbasis_meas(qc, qubit):
    """
    IonQ native gates decomposition for an X-basis measurement
    """
    qc.append(GPI2Gate(0.75), [qubit])

    # qc.append(GPI2Gate(0.25), [qubit])
    # qc.append(GPIGate(0), [qubit])
    #print("qc \n", qc)
    return qc


def apply_native_rigetti_h(qc, q0):
    """
    applies hadamard gate decomposed into Rigetti's native gates RX and RZ
    """
    qc.rx(np.pi/2, q0)
    qc.rz(np.pi/2, q0)
    qc.rx(np.pi/2, q0)
    return

def apply_native_rigetti_rzz(qc, theta, q0, q1):
    """
    applies rzz gate decomposed into Rigetti's native gates (RX, RZ, CZ)
    # TODO compile
    """
    apply_native_rigetti_h(qc, q1)
    qc.cz(q0, q1)
    apply_native_rigetti_h(qc, q1)
    qc.rz(theta, q1)
    apply_native_rigetti_h(qc, q1)
    qc.cz(q0, q1)
    apply_native_rigetti_h(qc, q1)
    return

def define_gates(circuit_choice, protocol_name, with_protocol_gates=False):
    # circuit gates
    if circuit_choice == 'HEA_RIGETTI' or circuit_choice == 'QAOA_RIGETTI':
        gates1Q_circuit = ['rx', 'ry']
        gates2Q_circuit = ['cz']
    elif circuit_choice == 'HEA_IONQ':
        gates1Q_circuit, gates2Q_circuit = [], [] #TODO replace with IONQ native gates
        
    # protocol measurement circuit gates
    if with_protocol_gates:
        if protocol_name == 'CS':
            gates1Q_meas = ['h', 'sdg']
            gates2Q_meas = []
        elif protocol_name == 'SWAP':
            gates1Q_meas = ['h']
            gates2Q_meas = ['cx']
        elif protocol_name == 'DensMat':
            gates1Q_meas, gates2Q_meas = [], []
    else:
        gates1Q_meas, gates2Q_meas = [], []
    return(list(set(gates1Q_circuit+gates1Q_meas)), list(set(gates2Q_circuit+gates2Q_meas)))

# Backend definition
def define_backend(backend_params, noise_params, basis_gates=None):
    backend_type = backend_params.type
    if backend_type == 'Aer_sim':
        #NUMERICAL (CIRCUIT) SIMULATIONS
        gates_1Q, gates_2Q = basis_gates
        noise_model = build_noise_model(noise_params, gates_1Q, gates_2Q)
        backend = AerSimulator(noise_model=noise_model)
    elif backend_type == 'Rigetti_QPU':
        #RIGETTI QPU
        provider = RigettiQCSProvider()
        provider.backends()
        backend = provider.get_backend('Ankaa-2', execution_timeout=200000, compiler_timeout=200000)
        # backend = provider.get_backend('Aspen-M-3', execution_timeout=200, compiler_timeout=200)
    elif backend_type == 'IonQ_QPU':
        #IONQ QPU
        ionq_key = os.getenv('IONQ_KEY')
        provider = IonQProvider(ionq_key)
        backend = provider.get_backend('ionq_qpu.aria-1', gateset="native")
    elif backend_type == 'IonQ_sim':
        #IONQ simulator
        ionq_key = os.getenv('IONQ_KEY')
        provider = IonQProvider(ionq_key)
        backend = provider.get_backend('ionq_simulator', gateset="native")
        backend.options.noise_model = 'aria-1' #'harmony' or 'ideal'
    return(backend)

# Noise models
def build_noise_model(noise_params, gates_1Q:list, gates_2Q:list):
    """
    returns a local depolarising + amplitude damping + readout error noise model noise_model  
    for single-qubit depolarising probability p_DP1
    and two-qubit depolarising probability p_DP2,
    and single-qubit amplitude damping probability p_AD1,
    and two-qubit amplitude damping probability p_AD2,
    and single-qubit amplitude damping probability p_ADmeas (T1 noise from measurement),
    and readout error probability p_meas in this form [[0.9, 0.1],[0.25,0.75]] i.e. [[P(0|0), P(1|0)],[P(0|1), P(1|1)]] 
    where P(n|m) is the probability of measuring n given that the ideal (without readout noise) outcome is m
    """
    # Errors
    error_DP1 = noise.depolarizing_error(noise_params.p_DP1, 1)
    error_DP2 = noise.depolarizing_error(noise_params.p_DP2, 2)
    error_AD1 = noise.amplitude_damping_error(noise_params.p_AD1)
    error_AD2 = noise.amplitude_damping_error(noise_params.p_AD2)
    error_ADmeas = noise.amplitude_damping_error(noise_params.p_ADmeas)
    error_meas = noise.errors.readout_error.ReadoutError(noise_params.p_meas)

    # Add errors to noise model
    noise_model = noise.NoiseModel()
    noise_model.add_all_qubit_quantum_error(error_DP1, gates_1Q)
    noise_model.add_all_qubit_quantum_error(error_DP2, gates_2Q)
    noise_model.add_all_qubit_quantum_error(error_AD1, gates_1Q)
    noise_model.add_all_qubit_quantum_error(error_AD2.tensor(error_AD2), gates_2Q)
    noise_model.add_all_qubit_quantum_error(error_ADmeas, ['id'])
    noise_model.add_all_qubit_readout_error(error_meas)


    return (noise_model)

def compute_p_DP_from_p_AD_and_fQ(p_AD1, p_AD2, f1Q, f2Q):
    """
    Compute the depolarizing probabilities p_DP1 and p_DP2 from the amplitude damping probabilities p_AD1 and p_AD2
    and the fidelities f1Q and f2Q for single-qubit and two-qubit gates.
    """
    # 1Q
    y = np.sqrt(1 - p_AD1)
    p_DP1 = ( 2 * (3*f1Q - 1) - (1 + y)**2 ) / ( y**2 - (1 + y)**2 )
    p_DP1 = round(p_DP1, 3)

    # 2Q
    x = np.sqrt(1 - p_AD2)
    p_DP2 = 1 - ( 4 * (5*f2Q - 1) - 1 ) / ( x * ( 4 + 6*x + 4*x**2 + x**3) )
    p_DP2 = round(p_DP2, 3)

    return p_DP1, p_DP2

def get_noise_param_from_calibration_data (f1Q, f2Q, T1, time_1Q, time_2Q, fmeas):
    """
    returns the noise parameters (depolarising probabilities and amplitude damping probabilities) 
    for single-qubit and two-qubit gates from the calibration data obtained from a QPU
    """
    # Amplitude damping probabilities from T1 and gate times
    p_AD1 = round(1 - np.exp(-time_1Q/T1), 3)
    p_AD2 = round(1 - np.exp(-time_2Q/T1), 3)


    # Depolarising probabilities from fidelities
    p_DP1, p_DP2 = compute_p_DP_from_p_AD_and_fQ(p_AD1, p_AD2, f1Q, f2Q)

    # Readout error probabilities
    p_meas = [[fmeas, round(1-fmeas, 3)], [round(1-fmeas, 3), fmeas]]

    noise_params = NoiseParams(p_DP1, p_DP2, p_AD1, p_AD2, 0, p_meas)
    return noise_params

# Quantum circuits
def init_circuit(circuit_params, num_qubits):
    qc = QuantumCircuit(num_qubits)
    if circuit_params.choice == 'QAOA_RIGETTI':
        for i in range(0, num_qubits):
            apply_native_rigetti_h(qc, i)
        qc.barrier()

    for index in range(circuit_params.depth_min):
        qc = add_circuit_layer(circuit_params, num_qubits, qc, index)

    return qc


def add_circuit_layer(circuit_params, num_qubits, qc, layer_index):
    if circuit_params.choice == 'HEA_RIGETTI':
        circuit_angles = circuit_params.circuits_angles[int((num_qubits-circuit_params.num_qubits_min)/circuit_params.num_qubits_step)]
        params = circuit_angles[layer_index]
        qc = add_layer_HEA_RIGETTI(qc, num_qubits, params, circuit_params.rx_only)
    elif circuit_params.choice == 'HEA_IONQ':
        circuit_angles = circuit_params.circuits_angles[int((num_qubits-circuit_params.num_qubits_min)/circuit_params.num_qubits_step)]
        params = circuit_angles[layer_index]
        qc = add_layer_HEA_IONQ(qc, num_qubits, params)
    elif circuit_params.choice == 'QAOA_RIGETTI':
        G = create_cycle_graph(num_qubits)
        theta = 2*np.pi*np.random.rand(2)
        qc = add_layer_QAOA_RIGETTI(qc, G, theta)
    return qc

def add_layer_HEA_IONQ (qc, num_qubits, params):
    """
    This functions adds a new layer to random circuit qc 
    (where num_qubits is the width of the quantum circuit
    and params a np.array of 2*num_qubits angles in [0,2\pi)]).
    IONQ hardware-efficient VQA ansatz
    """
    for i in range(num_qubits):
        t = round(params[i]/(2*np.pi), 3)
        qc.append(GPIGate(t), [i])
    for k in range(num_qubits):
        t = round(params[k+num_qubits]/(2*np.pi), 3)
        qc.append(GPI2Gate(t), [k])
    for g in range(2):
        for f in range(g, num_qubits-1, 2):
            qc.append(MSGate(0,0), [f, f+1])
    qc.barrier()
    return(qc)

def add_layer_HEA_RIGETTI (qc, num_qubits, params, rx_only=False):
    """
    This functions adds a new layer to random circuit qc 
    (where num_qubits is the width of the quantum circuit
    and params a np.array of 2*num_qubits angles in [0,2\pi)]).
    """
    # X rotation layer with random params
    for i in range(num_qubits):
        qc.rx(params[i], i) 
    if not rx_only:
        # Y rotation layer with random params
        for k in range(num_qubits):
            qc.ry(params[k+num_qubits], k)
    # 2 layers of nearest neighbour CZ gates
    for g in range(2):
        for f in range(g, num_qubits-1, 2):
            qc.cz(f,f+1)
    # Barrier function prevents compiler changing circuit structure
    qc.barrier()
    #print(qc) # visualise quantum circuit
    return (qc)

def add_layer_QAOA_RIGETTI(qc, G, theta):
    """
    **Inspired from qiskit tutorials**
    Adds a layer of qaoa unitaries to a circuit using Rigetti native gates rx, ry, cz
    
    Args:  
        qc: qiskit circuit
        G: networkx graph
        theta: list
               unitary parameters (2-element list)
                     
    Returns:
        qc: qiskit circuit
    """
    
    nqubits = len(G.nodes())
    p = len(theta)//2  # number of alternating unitaries
    
    beta = theta[:p]
    gamma = theta[p:]

    # problem unitary
    for pair in list(G.edges()):
        apply_native_rigetti_rzz(qc, 2 * gamma[0], pair[0], pair[1]) #qc.rzz(2 * gamma[0], pair[0], pair[1])
   
    qc.barrier()

    # mixer unitary
    for i in range(0, nqubits):
        qc.rx(2 * beta[0], i)
    qc.barrier()
    
    return qc

def add_layer_qaoa(qc, G, theta, use_native_gates:bool):
    """
    **Inspired from qiskit tutorials**
    Adds a layer of qaoa unitaries to a circuit
    
    Args:  
        qc: qiskit circuit
        G: networkx graph
        theta: list
               unitary parameters (2-element list)
                     
    Returns:
        qc: qiskit circuit
    """
    
    nqubits = len(G.nodes())
    p = len(theta)//2  # number of alternating unitaries
    
    beta = theta[:p]
    gamma = theta[p:]

    # problem unitary
    for pair in list(G.edges()):
        if use_native_gates:
            apply_rzz_native(qc, 2 * gamma[0], pair[0], pair[1])
        else:
            qc.rzz(2 * gamma[0], pair[0], pair[1])
   
    qc.barrier()

    # mixer unitary
    for i in range(0, nqubits):
        qc.rx(2 * beta[0], i)
    qc.barrier()
    
    return qc

def add_layer_IONQ_VQC (qc, num_qubits, params):
    """
    This functions adds a new layer to random circuit qc 
    (where num_qubits is the width of the quantum circuit
    and params a np.array of 2*num_qubits angles in [0,2\pi)]).
    IONQ hardware-efficient VQA ansatz
    """
    for i in range(num_qubits):
        t = round(params[i]/(2*np.pi), 3)
        qc.append(GPIGate(t), [i])
    for k in range(num_qubits):
        t = round(params[k+num_qubits]/(2*np.pi), 3)
        qc.append(GPI2Gate(t), [k])
    for g in range(2):
        for f in range(g, num_qubits-1, 2):
            qc.append(MSGate(0,0), [f, f+1])
    qc.barrier()
    return(qc)

def add_layer_VQC (qc, num_qubits, params):
    """
    This functions adds a new layer to random circuit qc 
    (where num_qubits is the width of the quantum circuit
    and params a np.array of 2*num_qubits angles in [0,2\pi)]).
    """
    print (params)
    # X rotation layer with random params
    for i in range(num_qubits):
        print (i)
        qc.rx(params[i], i) 
    # Y rotation layer with random params
    for k in range(num_qubits):
            qc.ry(params[k+num_qubits], k)
    # 2 layers of nearest neighbour CZ gates
    for g in range(2):
        for f in range(g, num_qubits-1, 2):
            qc.cz(f,f+1)
    # Barrier function prevents compiler changing circuit structure
    qc.barrier()
    #print(qc) # visualise quantum circuit
    return (qc)

def add_layer_VQC_rx_only (qc, num_qubits, params):
    """
    This functions adds a new layer to random circuit qc 
    (where num_qubits is the width of the quantum circuit
    and params a np.array of 2*num_qubits angles in [0,2\pi)]).
    """
    # X rotation layer with random params
    for i in range(num_qubits):
        qc.rx(params[i], i)
    # 2 layers of nearest neighbour CZ gates
    for g in range(2):
        for f in range(g, num_qubits-1, 2):
            qc.cz(f,f+1)
    # Barrier function prevents compiler changing circuit structure
    qc.barrier()
    #print(qc) # visualise quantum circuit
    return (qc)

def VQC_Unit(num_qubits, depth, angles, rx_only:bool):
    """
    This function generates a randomised (parameterised) quantum circuit
     with depth depth and width num_qubits, and parameters params
    """
    qc = QuantumCircuit(num_qubits)
    #print('params = ', params)
    for dep in range(depth):
        if rx_only:
            qc = add_layer_VQC_rx_only(qc, num_qubits, angles[dep])
        else:
            qc = add_layer_VQC(qc, num_qubits, angles[dep])
    return qc

def apply_Pauli_meas_unit (qc, num_qubits, unit_index, backend):
    """
    Applies a unitary made of single-qubit unitaries (described by uni_index) corresponding to
    measurements in the X, Y or Z basis, to the quantum circuit qc (that has width num_qubits)
    """
    for i in range (num_qubits):
        if unit_index[i] == 0: # X-basis measurement
            if backend.name() == 'ionq_simulator' or backend.name() == 'ionq_qpu':
                qc = add_native_ionq_xbasis_meas_unit(qc, i)
            else:
                qc.h(i)
        elif unit_index[i] == 1: # Y-basis measurement
            if backend.name() == 'ionq_simulator' or backend.name() == 'ionq_qpu':
                qc = add_native_ionq_ybasis_meas_unit(qc, i)
            else:
                qc.sdg(i)
                qc.h(i)
        # Z-basis measurement - identity matrix is applied
    return (qc)

def measure_Zbasis (qc, K, backend, initial_layout, verbose=False, IONQ=False):
    """
    qc: a quantum circuit
    K: number of shots i.e. number of times we want to measure the quantum circuit qc
    backend: backend to run the quantum circuit on
    initial_layout: label of the qubits we want to use in the lattice (e.g. [100, 101, 107] for a 3-qubit circuit); set to None if we want to use the default layout

    output: counts, a dictionary with measurement outcomes and the corresponding number of times the outcome was obtained among the K measurements
    """
    # Perform computational basis measurements on output quantum register of circuit
    qc_copy = qc.copy("qc_copy")
    
    qc_copy.measure_all()

    if verbose: start_time = time.time()
    if IONQ:
        job = backend.run(qc_copy, shots=K, error_mitigation=ErrorMitigation.NO_DEBIASING, initial_layout=initial_layout)
    else:
        job = backend.run(qc_copy, shots=K, initial_layout=initial_layout)
    if verbose: 
        elapsed_time = time.time() - start_time
        print("=========================================")
        print("Compilation and measurement time (in seconds) : ", elapsed_time)
    result = job.result()
    counts = result.get_counts()
    return (counts)

def apply_hadamard_native(qc, q0):
    """
    applies hadamard gate decomposed into Rigetti's native gates RX and RZ
    """
    qc.rx(np.pi/2, q0)
    qc.rz(np.pi/2, q0)
    qc.rx(np.pi/2, q0)
    return

def apply_rzz_native(qc, theta, q0, q1):
    """
    applies rzz gate decomposed into Rigetti's native gates (RX, RZ, CZ)
    """
    apply_hadamard_native(qc, q1)
    qc.cz(q0, q1)
    apply_hadamard_native(qc, q1)
    qc.rz(theta, q1)
    apply_hadamard_native(qc, q1)
    qc.cz(q0, q1)
    apply_hadamard_native(qc, q1)
    return