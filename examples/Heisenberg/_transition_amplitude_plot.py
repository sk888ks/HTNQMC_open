#(SSVQEから)遷移振幅を取得

from qiskit.opflow import X, Z, I
#from _cython_vqe_handmade_ansatz import cStatevector 
from qiskit.quantum_info.states import Statevector

from _vqe_handmade_ansatz import SSVQE_handmade_HEA, handmade_HEA, VQE_handmade_HEA

import numpy as np

import matplotlib.pyplot as plt;plt.rcParams['figure.dpi'] = 300

from qiskit_nature.drivers import Molecule
from qiskit_nature.drivers.second_quantization import (
    ElectronicStructureDriverType,
    ElectronicStructureMoleculeDriver,
)
from qiskit_nature.problems.second_quantization import ElectronicStructureProblem
from qiskit_nature.converters.second_quantization import QubitConverter
from qiskit_nature.mappers.second_quantization import JordanWignerMapper
import os


def get_H_from_circuits(Hamiltonian, depth, result_param , num_states:int=None, RealAmplitude=False):
    #VQE基底でのHを計算
    init_H_array = np.array(Hamiltonian.to_matrix())
    if num_states is None:
        num_states = 2**Hamiltonian.num_qubits
    H_array = np.empty((num_states, num_states)).astype("complex128")
    for i in range(num_states):
        for j in range(num_states):
            #状態i,j準備
            state_i_list = list(map(int,format(i, "0"+str(Hamiltonian.num_qubits)+"b")))  #indexを二進数にしてリストに変換
            bound_circuit_i = handmade_HEA(Hamiltonian.num_qubits, depth, init_state=state_i_list, RealAmplitude=RealAmplitude).bind_parameters(result_param)
            statevector_i = Statevector(bound_circuit_i).data

            state_j_list = list(map(int,format(j, "0"+str(Hamiltonian.num_qubits)+"b")))  #indexを二進数にしてリストに変換
            bound_circuit_j = handmade_HEA(Hamiltonian.num_qubits, depth, init_state=state_j_list, RealAmplitude=RealAmplitude).bind_parameters(result_param)
            statevector_j = Statevector(bound_circuit_j).data

            #期待値取得
            Hij_temp = np.conjugate(statevector_i) @ init_H_array @ statevector_j
            H_array[i][j] = Hij_temp

    return H_array

def plt_matrix_colorbar(Hij, vmin=-2, vmax=0):
    #Hのカラープロット
    fig, ax = plt.subplots()
    im = ax.imshow(Hij, vmin=vmin, vmax=vmax, cmap = "viridis")
    fig.colorbar(im, ax=ax)
    plt.show()
    return

def main():
    pass

if __name__ == "__main__":
    main()