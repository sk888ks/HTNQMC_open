from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp
from qiskit_nature.mappers.second_quantization.jordan_wigner_mapper import JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit.opflow import X, Y, Z, I

import numpy as np
from scipy.optimize import minimize
from _HT_calc_energy_qiskit_main import hybrid_tensor_calc_energy
import itertools

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
depth_network_list=[[2,2],2] #[[2,2],2] means that the depths for all the tensors are two.


#Exact value
eigenvalue,eigenvec=np.linalg.eigh(np.array(Ham.to_matrix()))
print("Exact eigen energy {}".format(min(eigenvalue)))

c=hybrid_tensor_calc_energy(network=network,Hamiltonian=Ham,depth_network_list=depth_network_list)

#Matrix size for each subsystems
num_states_intra_list=[2,2]

H_intra_list = c.ham_intra_list 
if len(H_intra_list) != len(num_states_intra_list): 
    raise Exception("len(H_intra_list) != len(num_states_intra_list)")

ham_pauli_list=c.ham_pauli_list
ham_coeff_list=c.ham_coeff_list

num_subsystem = len(network)
num_qubits = len(list(itertools.chain.from_iterable(network)))

#Preparing all zero indices
h_vec_list = [c.get_h_vec(0, num_qubits, num_subsystem, network, num_states_intra_list)] 

num_parameters = 0
for depth_m, network_m in zip(depth_network_list[0], network):
    num_qubits_m = len(network_m)
    num_parameters += depth_m * num_qubits_m +num_qubits_m
num_qubits_global = len(h_vec_list[0][-1])
num_parameters += depth_network_list[1] * num_qubits_global +num_qubits_global

weight_list=[i+1 for i in reversed(range(len(h_vec_list)))]
c.set_parameters_for_ssvqeall(depth_network_list=depth_network_list, weight_list=weight_list, network=network, h_vec_list=h_vec_list,num_states_intra_list=num_states_intra_list,RealAmplitude=True, particle_number_preserving=False )

np.random.seed(0)

#Execute HTN+VQE
ssvqe_result = minimize(c.SSVQE_cost_alltensor, x0 = np.random.rand(num_parameters) ,method="SLSQP", options={'maxiter':100000}, callback = c.callback_SSVQE_cost_alltensor)

#print result
print(ssvqe_result)
print(network,num_states_intra_list, depth_network_list, h_vec_list)

#save result
np.save(os.path.join(os.getcwd(), "npy/vqeall_Heisenberg_RA"), ssvqe_result.x)
