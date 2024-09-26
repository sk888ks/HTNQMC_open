import qiskit 
from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp
from qiskit_nature.mappers.second_quantization.jordan_wigner_mapper import JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit.opflow import X, Y, Z, I

import numpy as np

from _HT_QC_FCIQMC_main import QC_FCIQMC

from _transition_amplitude_plot import plt_matrix_colorbar
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

Ham = eval(Ham_str)

#Exact value
eigenvalue,eigenvec=np.linalg.eigh(np.array(Ham.to_matrix()))
print("Exact eigen energy {}".format(min(eigenvalue))) 
eigenvect=eigenvec.T[0]
abs_max_index = np.argmax(np.abs(eigenvect))
print("abs_max_index=",abs_max_index)

H_hd_h_array = np.array(Ham.to_matrix()).real
# diagH = np.diag(H_hd_h_array)
absH_hd_h_array=np.abs(H_hd_h_array)

overlap_vec = np.load(os.path.join(os.getcwd(),"npy/overlap_Heisenberg_RA_HT_HF.npy")) #Quantum QMC
#overlap_vec = None #Classical QMC

#parameters for QMC
dt=0.001
evol_time=dt*10000
iidx = abs_max_index
walker_set=[[iidx,1,1]]
shift=H_hd_h_array[iidx][iidx]
energy_init = shift

fixed_num_walker = 1000
damping = 0.1
shift_step_A = 5 

rho = 10**(-16) #0
mixed_energy_MD = True #Quantum QMC
# mixed_energy_MD = False #Classical QMC


#execute QMC
walker_set = QC_FCIQMC(evol_time, dt, H_hd_h_array, absH_hd_h_array, walker_set, shift,  fixed_num_walker = fixed_num_walker, damping = damping, shift_step_A = shift_step_A, init_energy_index=iidx, rho=rho, mixed_energy_MD=mixed_energy_MD, overlap_vec=overlap_vec)

#print walker
print(walker_set)
