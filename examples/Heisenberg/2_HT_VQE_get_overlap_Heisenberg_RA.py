#calc overlap between HF and TN_VQE
import qiskit 
from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp
from qiskit_nature.mappers.second_quantization.jordan_wigner_mapper import JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit.opflow import X, Y, Z, I

import numpy as np
from _HT_calc_energy_qiskit_main import hybrid_tensor_calc_energy

import itertools

import joblib
import os

#Tensor structure
network=[[7,6,5,4],[3,2,1,0]]

n_sites = len(list(itertools.chain.from_iterable(network)))

#create Hamiltonian
J_intra=1.0
J_inter=1.0

Ham_str = ""
for Pauli in ["X","Y","Z"]:
    for Pauli_index in range(n_sites):
        if Pauli_index == n_sites-1:
            break
        if (Pauli_index + 1) % 4 == 0:
            Ham_str_temp_coeff = str(J_inter)+"*"
        else:
            Ham_str_temp_coeff = str(J_intra)+"*"
        Ham_str_temp_pauli = ""
        for Qubit_index in range(n_sites-1):
            if Qubit_index != 0:
                Ham_str_temp_pauli += "^" 
            if Qubit_index != Pauli_index:   
                Ham_str_temp_pauli += "I"
            else:
                Ham_str_temp_pauli += Pauli+"^"+Pauli
        Ham_str += "(" + Ham_str_temp_coeff + Ham_str_temp_pauli + ")"
        if Pauli_index != n_sites-2:
            Ham_str += "+"
            #Ham_str += "\n"
    if Pauli=="X" or Pauli=="Y":
        Ham_str += "+"

#print(Ham_str)

Ham = eval(Ham_str)

#depth for each tensor
depth_network_list=[[2,2],2]


c=hybrid_tensor_calc_energy(network=network,Hamiltonian=Ham,depth_network_list=depth_network_list)

num_states_intra_list=[2,2]
H_intra_list = c.ham_intra_list 
if len(H_intra_list) != len(num_states_intra_list): 
    raise Exception("len(H_intra_list) != len(num_states_intra_list)")

ham_pauli_list=c.ham_pauli_list
ham_coeff_list=c.ham_coeff_list

num_subsystem = len(network)
num_qubits = len(list(itertools.chain.from_iterable(network)))

#import joblib
import time
init_t = time.time()
def get_matrix_element_H_HT(hd_vec, h_temp, depth_network_list, init_param_list, num_states_intra_list, num_qubits, num_subsystem, network): 
    h_bin = [format(h_temp, "0"+str(num_qubits)+"b")]
    overlap_h = c.get_overlap_HT_HF(hd_vec, h_bin, depth_network_list, init_param_list, num_states_intra_list , num_qubits, num_subsystem, network,  RealAmplitude=True, particle_number_preserving=False) 
    return h_temp, overlap_h

hd_vec_list = [c.get_h_vec(0, num_qubits, num_subsystem, network, num_states_intra_list)] 

ssvqeall_param = np.load(os.path.join(os.getcwd(), "npy/vqeall_Heisenberg_RA.npy"))
init_param_list = [array.tolist() for array in c.param_split(ssvqeall_param,depth_network_list, network, hd_vec_list, RealAmplitude=True, particle_number_preserving=False)]

matrix_size = 2**num_qubits

#calculate overlap in parallel
overlap_h_array = np.zeros((matrix_size))
TS_HF_array = joblib.Parallel(n_jobs=-1)(joblib.delayed(get_matrix_element_H_HT)(hd_vec_list[0], h_temp, depth_network_list, init_param_list, num_states_intra_list, num_qubits, num_subsystem, network) for h_temp in range(matrix_size))

#substitute element
for hd, element in TS_HF_array:
    overlap_h_array[hd] = element

#print result
print(overlap_h_array)
fin_para_t = time.time()
print(fin_para_t-init_t)

#save overlap
np.save(os.path.join(os.getcwd(), "npy/overlap_Heisenberg_RA_HT_HF"), overlap_h_array)

eigenvalue,eigenvec=np.linalg.eigh(np.array(Ham.to_matrix()))
print("Exact eigen energy {}".format(min(eigenvalue)))
eigenvecT = eigenvec.T[0]
overlapFCI = np.dot(eigenvecT,overlap_h_array)
fidelity = np.conj(overlapFCI)*overlapFCI
print("fidelity:",fidelity.real)
