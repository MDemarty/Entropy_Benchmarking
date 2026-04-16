"""
Useful functions to perform density matrix simulations
"""
import numpy as np
from qiskit.quantum_info import purity, entropy

from libQC import init_circuit, add_circuit_layer
from libQC import define_gates, define_backend
from libMetric import Metrics

def get_output_density_matrix(qc, backend):
    """
    returns the output density matrix of the quantum circuit qc under some noise model
    defined by the choice of backend, and using Qiskit density matrix simulator
    """

    qc_copy = qc.copy("qc_copy")
    qc_copy.save_density_matrix()

    result = backend.run(qc_copy).result()
    density_matrix = result.data()['density_matrix']

    return(density_matrix)

def get_metrics_DensMat_single_width (backend, circuit_params, num_qubits):
    """
    *returns 3 lists all_vNd, all_pur, all_R2d corresponding
    to the values of the von Neumann entropy density, Purity and second-order Renyi 
    entropy density for a quantum circuit of different depths and fixed width num_qubits.
    """

    all_vNd = []
    all_pur = []
    all_R2d = []

    #Initialise circuit
    qc = init_circuit(circuit_params, num_qubits)

    for depth_index in range(circuit_params.depth_min, circuit_params.depth_max+1, circuit_params.depth_step):
        print("depth = ", depth_index)

        if depth_index>circuit_params.depth_min:
            for index in range(circuit_params.depth_step):
                qc = add_circuit_layer(circuit_params, num_qubits, qc, depth_index - circuit_params.depth_step + index)
        # print("\n Quantum circuit \n", qc)

        density_matrix = get_output_density_matrix(qc, backend)

        # Metrics
        vNd = entropy(density_matrix, base=2) / num_qubits
        pur = np.real(purity(density_matrix))
        R2d = -1 * np.log2(pur) / num_qubits

        all_vNd.append(vNd)
        all_pur.append(pur)
        all_R2d.append(R2d)  

    return(all_vNd, all_pur, all_R2d)


def get_metrics_DensMat(experiment_params):
    """
    *same as get_metrics_DensMat_single_width but for different circuit widths
    (from num_qubits_min to num_qubits_max)
    """
    circuit_params = experiment_params.circuit_params
    noise_params = experiment_params.noise_params
    backend_params = experiment_params.backend_params


    basis_gates = define_gates(circuit_params.choice, 'DensMat')
    backend = define_backend(backend_params, noise_params, basis_gates)
    backend.set_options(method="density_matrix")

    metrics = Metrics(['all_vNd_diff_n', 'all_pur_diff_n', 'all_R2d_diff_n'])
 
    for num_qubits in range(circuit_params.num_qubits_min, circuit_params.num_qubits_max+1, circuit_params.num_qubits_step):
        print("-- \n Number of qubits : ", num_qubits)

        all_vNd, all_pur, all_R2d = get_metrics_DensMat_single_width(backend, circuit_params, num_qubits)
        metrics['all_vNd_diff_n'].append(all_vNd)
        metrics['all_pur_diff_n'].append(all_pur)
        metrics['all_R2d_diff_n'].append(all_R2d)

    return metrics

