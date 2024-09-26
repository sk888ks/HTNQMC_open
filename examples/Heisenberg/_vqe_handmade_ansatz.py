#お手製のHEAで(SS)VQE
from curses import A_ALTCHARSET
from logging import exception
from mimetypes import init
from unicodedata import decomposition
from qiskit.algorithms import VQE, NumPyMinimumEigensolver
from qiskit.algorithms.optimizers import SLSQP

from qiskit import Aer
from qiskit.utils import QuantumInstance
from qiskit.opflow import X, Z, I
from qiskit.utils import algorithm_globals
from qiskit import QuantumCircuit

import numpy as np

from qiskit.opflow.state_fns.circuit_state_fn import CircuitStateFn
from qiskit.opflow.converters.circuit_sampler import CircuitSampler

from qiskit.opflow.expectations.pauli_expectation import PauliExpectation 
from qiskit.opflow.expectations.aer_pauli_expectation import AerPauliExpectation
from qiskit.opflow.expectations.matrix_expectation import MatrixExpectation
from qiskit.opflow.state_fns.state_fn import StateFn

import os

from qiskit.circuit import ParameterVector, QuantumCircuit

from scipy.optimize import minimize
import shutil
from qiskit.quantum_info.states import Statevector

import math

def handmade_HEA(num_qubits,num_depth,init_state = None, RealAmplitude = False):
    #手作りansatz
    # define parameters
    p = ParameterVector('p', num_qubits*2*num_depth + num_qubits*2 )
    circuit = QuantumCircuit(num_qubits)

    #初期状態準備
    if init_state is not None:
        if len(init_state) != num_qubits:
            raise Exception("len(init_state) != num_qubits")
        #circuit.initialize(init_state) #circuit.initialize did not work
        for qubit_index_prep, state in enumerate(reversed(init_state)):
            if state == 1:
                circuit.x(qubit_index_prep)
        circuit.barrier()

    param_index = 0
    for d in range(num_depth):
        #RyRz
        for qubit_index in range(num_qubits):
            circuit.ry(p[param_index], qubit_index)
            param_index += 1 
        if not RealAmplitude:
            for qubit_index in range(num_qubits):
                circuit.rz(p[param_index], qubit_index)
                param_index += 1 
        
        circuit.barrier()

        #CNOT
        for qubit_index in range(num_qubits-1):
            if qubit_index % 2 == 0:
                circuit.cnot(qubit_index, qubit_index+1)
        for qubit_index in range(num_qubits-1):
            if qubit_index % 2 == 1:
                circuit.cnot(qubit_index, qubit_index+1)
        
        circuit.barrier()

    #RyRz
    for qubit_index in range(num_qubits):
        circuit.ry(p[param_index], qubit_index)
        param_index += 1 
    if not RealAmplitude:
        for qubit_index in range(num_qubits):
            circuit.rz(p[param_index], qubit_index)
            param_index += 1 
    # print(init_state)
    # print(circuit.draw())
    return circuit


def handmade_particle_preserving_ansatz(num_qubits,num_depth,init_state = None):
    #手作りansatz
    # define parameters
    p = ParameterVector('p', (num_qubits-1)*num_depth)
    circuit = QuantumCircuit(num_qubits)

    #初期状態準備
    if init_state is not None:
        if len(init_state) != num_qubits:
            raise Exception("len(init_state) != num_qubits")
        #circuit.initialize(init_state) #circuit.initialize did not work
        for qubit_index_prep, state in enumerate(reversed(init_state)):
            if state == 1:
                circuit.x(qubit_index_prep)
        circuit.barrier()

    param_index = 0
    for d in range(num_depth):
        #CNOT
        for qubit_index in range(num_qubits-1):
            if qubit_index % 2 == 0:
                circuit.h(qubit_index)
                circuit.cnot(qubit_index, qubit_index+1)
                circuit.ry(p[param_index], qubit_index)
                circuit.ry(p[param_index], qubit_index+1)
                param_index += 1
                circuit.cnot(qubit_index, qubit_index+1)
                circuit.h(qubit_index)
        for qubit_index in range(num_qubits-1):
            if qubit_index % 2 == 1:
                circuit.h(qubit_index)
                circuit.cnot(qubit_index, qubit_index+1)
                circuit.ry(p[param_index], qubit_index)
                circuit.ry(p[param_index], qubit_index+1)
                param_index += 1
                circuit.cnot(qubit_index, qubit_index+1)
                circuit.h(qubit_index)
        circuit.barrier()
    if param_index != (num_qubits-1)*num_depth:
        raise Exception("param_index != (num_qubits-1)*num_depth")

    # print(init_state)
    #print(circuit.draw())
    return circuit


def VQE_handmade_HEA(depth:int = 2, Hamiltonian = None, seed = 50):
    #handmade ansatzでVQE
    qubit = Hamiltonian.num_qubits
    #VQE
    #settings
    algorithm_globals.random_seed = seed
    qi = QuantumInstance(Aer.get_backend('statevector_simulator'), seed_transpiler=seed, seed_simulator=seed)

    ansatz = handmade_HEA(qubit, depth)
    slsqp = SLSQP(maxiter=1000)

    #vqe
    vqe = VQE(ansatz, optimizer=slsqp, quantum_instance=qi)

    #result
    result = vqe.compute_minimum_eigenvalue(Hamiltonian)

    return result

def get_expectation_value_snapshot(circuit, Hamiltonian):
    #期待値計算
    psi = CircuitStateFn(circuit) 
    #B 状態ベクトルの場合##################
    backend = Aer.get_backend('statevector_simulator') #qasm_simulator使うとMatrix expectationが毎回値変わった

    # define the state to sample
    measurable_expression = StateFn(Hamiltonian, is_measurement=True).compose(psi) #オブザーバブルと状態ベクトルから、測定から期待値計算できるように準備する？

    # snapshot
    expectation = AerPauliExpectation().convert(measurable_expression)
    sampler = CircuitSampler(backend).convert(expectation)   

    return sampler.eval().real


def SSVQE_handmade_HEA(Hamiltonian, depth, weight_list = None, num_states = 2,  RealAmplitude = False, particle_preserving_ansatz = False):
    #handmade ansatz でSSVQE
    if type(Hamiltonian) is np.ndarray:
        num_qubits = int(math.log2(len(Hamiltonian[0])))
    else:
        num_qubits = Hamiltonian.num_qubits

    count_SSVQE = 0
    def SSVQE_cost(params):
        #SSVQEのコスト
        cost = 0
        #todo arrayの場合の対処法
        if type(Hamiltonian) is np.ndarray:
            for i in range(num_states):
                state_i_list = list(map(int,format(i, "0"+str(num_qubits)+"b")))  #indexを二進数にしてリストに変換
                if particle_preserving_ansatz:
                    bound_circuit_i = handmade_particle_preserving_ansatz(num_qubits, depth, init_state=state_i_list).bind_parameters(params)
                else:     
                    bound_circuit_i = handmade_HEA(num_qubits, depth, init_state=state_i_list, RealAmplitude=RealAmplitude).bind_parameters(params)
                statevector_i = Statevector(bound_circuit_i).data     
                #期待値取得
                cost += weight_list[i] * (np.conjugate(statevector_i) @ Hamiltonian @ statevector_i).real        
        else:
            for ansatz, weight in zip(ansatz_list, weight_list):
                cost += weight*get_expectation_value_snapshot(ansatz.bind_parameters(params), Hamiltonian)

        return cost

    def callback_SSVQE(params):
        #SSVQE callback
        nonlocal count_SSVQE
        if count_SSVQE == 0:
            print("Iter Cost", *["State_"+str(i) for i in range(num_states)])
            if os.path.exists('ssvqe_out.txt'):
                if os.path.exists("ssvqe_out_old.txt"):
                    os.remove("ssvqe_out_old.txt")
                shutil.move("ssvqe_out.txt","ssvqe_out_old.txt")
            with open(os.path.join(os.path.dirname(__file__), 'ssvqe_out.txt'), mode = "w") as f:
                f.write("{0} {1}".format("Iter", "Cost"))
                f.writelines([" State_"+str(i) for i in range(num_states)])
                f.write("\n")

        count_SSVQE += 1
        cost = 0
        expectation_list = []

        if type(Hamiltonian) is np.ndarray:
            for i in range(num_states):
                state_i_list = list(map(int,format(i, "0"+str(num_qubits)+"b")))  #indexを二進数にしてリストに変換
                bound_circuit_i = handmade_HEA(num_qubits, depth, init_state=state_i_list, RealAmplitude=RealAmplitude).bind_parameters(params)
                statevector_i = Statevector(bound_circuit_i).data     
                #期待値取得
                expectation_value = (np.conjugate(statevector_i) @ Hamiltonian @ statevector_i).real
                cost += weight_list[i] * expectation_value        
                expectation_list.append(expectation_value)
        else:
            for ansatz, weight in zip(ansatz_list, weight_list):
                expectation_value_snapchot = get_expectation_value_snapshot(ansatz.bind_parameters(params), Hamiltonian)
                cost += weight*expectation_value_snapchot
                expectation_list.append(expectation_value_snapchot)
        print(count_SSVQE, cost, *expectation_list)
        with open(os.path.join(os.path.dirname(__file__), 'ssvqe_out.txt'), mode = "a") as f:
            f.write("{0} {1}".format(count_SSVQE, cost))
            f.writelines([" "+i for i in list(map(str,expectation_list))])
            f.write("\n")
        return

    #SSVQE 
    ansatz_list=[]
    for state_index in range(num_states):
        state_index_binary = format(state_index, "0"+str(num_qubits)+"b")
        init_state_list = list(map(int,state_index_binary))
        ansatz_list.append(handmade_HEA(num_qubits ,depth, init_state=init_state_list, RealAmplitude=RealAmplitude))

    if weight_list is None:
        weight_list = [i+1 for i in reversed(range(len(ansatz_list)))] #weight_list = [...,3,2,1]

    np.random.seed(0)
    ssvqeresult = minimize(SSVQE_cost, x0 = np.random.rand(ansatz_list[0].num_parameters) ,method="SLSQP", options={'maxiter':100000}, callback=callback_SSVQE)
    return ssvqeresult

def main():
    pass

if __name__ == "__main__":
    main()
