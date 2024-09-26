import numpy as np
import time
import copy
import itertools
import os
import shutil

from _transition_amplitude_plot import  get_H_from_circuits
from functools import reduce
import math

#from _cython_vqe_handmade_ansatz import chandmade_HEA as handmade_HEA 
from _vqe_handmade_ansatz import handmade_HEA, handmade_particle_preserving_ansatz 
#from _cython_vqe_handmade_ansatz import cStatevector as Statevector 
from qiskit.quantum_info.states import Statevector

from qiskit.opflow import X, Z, I, Y


class hybrid_tensor_calc_energy:
    def __init__(self,network,Hamiltonian,depth_network_list): #tested
        #初期化
        self.time_start=time.time()
        
        #ハミルトニアンの係数とパウリをリスト形式にする
        ham_coeff_list, ham_pauli_list_flatten = self.term_to_list(Hamiltonian)

        #ネットワークの指数がサイト数と異なる
        if len(list(itertools.chain.from_iterable(network)))!=len(ham_pauli_list_flatten[0]):
            raise Exception("inccorect network indices")
        
        #ネットワークで指定した順番にパウリ演算子を変更し、パウリ演算子をネットワークと同じリスト構造にする
        ham_pauli_list=self.get_pauli_network_split(ham_pauli_list_flatten,network)

        #色々設定
        self.ham_pauli_list=ham_pauli_list 
        self.ham_coeff_list=ham_coeff_list
        self.network=network
        self.depth_network_list=depth_network_list
        self.count_SSVQE = 0
        self.init_t = time.time()


        #localをまたがる項削除（つまりHintra出力）
        self.extracted_pauli_list, self.extracted_coeff_list=self.get_extracted_local_pauli_coeff_list(self.ham_pauli_list, self.ham_coeff_list)
        
        #todo計算するパウリと係数のリストをローカルテンソルごとの要素となるよう入れ替え、I削除
        ham_intra_pauli_list, ham_intra_coeff_list = self.get_ham_intra_list(self.extracted_pauli_list, self.extracted_coeff_list)
        #Qiskitのハミルトニアンを作成
        self.ham_intra_list=[]
        for ham_intra_pauli, ham_intra_coeff in self.zip_equal(ham_intra_pauli_list, ham_intra_coeff_list):
            self.ham_intra_list.append(self.list_to_term(ham_intra_pauli, ham_intra_coeff))
    
    def zip_equal(self, *iterables):
        for i in range(len(iterables)):
            if i==0:
                pass
            elif len(iterables[i-1]) != len(iterables[i]):
                raise Exception("lengths in the iterable are different")
        return zip(*iterables)

        
    def get_pauli_network_split(self,ham_pauli_list_flatten,network):#tested
        #ネットワークで指定した順番にパウリ演算子を変更し、パウリ演算子をネットワークと同じリスト構造にする

        temp_network_flatten=list(itertools.chain.from_iterable(network)) 
        
        ham_pauli_list=[]
        for plist in ham_pauli_list_flatten:
            ham_pauli_list_temp_temp=[]
            ##ネットワークで指定した順番にパウリ演算子を変更
            for site_index in temp_network_flatten:
                ham_pauli_list_temp_temp.append(plist[-site_index-1])
            n_network_idx=0
            n_splitted_idx=0
            ham_pauli_list_temp_temp_splitted=[]

            #パウリ演算子をネットワークと同じリスト構造にする
            while True:
                ham_pauli_list_temp_temp_splitted.append(ham_pauli_list_temp_temp[n_splitted_idx:n_splitted_idx + len(network[n_network_idx])])
                n_splitted_idx+=len(network[n_network_idx])
                n_network_idx+=1
                if n_splitted_idx==len(temp_network_flatten):
                    break
                elif n_splitted_idx>len(temp_network_flatten): #index数合わせ
                    raise Exception("n_splitted_idx>len(temp_network_flatten)")
            ham_pauli_list.append(ham_pauli_list_temp_temp_splitted)
        return ham_pauli_list

    def get_H_tilde(self, ham_pauli_list, ham_coeff_list, depth_list, result_list, num_states_list, RealAmplitude=False): #test #todo particle_number_preserving
        #M^m(u)を計算し、\tilde{H}も計算
        H_tilde = 0
        for ham_pauli_one_term, ham_coeff in self.zip_equal(ham_pauli_list, ham_coeff_list):
            M_mu_temp_list = []            
            for ham_pauli_one_tensor, depth, result , num_states in self.zip_equal(ham_pauli_one_term, depth_list, result_list, num_states_list): #e.g., ["I", "Z"]
                #リスト形式のパウリを演算子に変更
                ham_pauli_list_for_TF = [ham_pauli_one_tensor]
                ham_coeff_list_for_TF = [1.0]
                ope_temp = self.list_to_term(ham_pauli_list_for_TF, ham_coeff_list_for_TF)
                #M^m(u)生成
                M_mu_temp_list.append(get_H_from_circuits(Hamiltonian=ope_temp, depth=depth, result_param=result.x , num_states=num_states, RealAmplitude=RealAmplitude))
            #M^m(u)のテンソル積とって係数かける。それらを足す
            H_tilde += ham_coeff * reduce(np.kron, M_mu_temp_list)
                
        return H_tilde

    def get_H_hd_h_tilde(self, hd_local_vec, h_local_vec, ham_pauli_list, ham_coeff_list, depth_list, init_param_list_local, num_states_list, RealAmplitude=False , particle_number_preserving=False): #tested
        #H_hd_h_tildeを計算
        H_hd_h_tilde = 0
        for ham_pauli_one_term, ham_coeff in self.zip_equal(ham_pauli_list, ham_coeff_list):
            N_mu_temp_list = []            
            for hd_local, h_local, ham_pauli_one_tensor, depth, init_param , num_states in self.zip_equal(hd_local_vec, h_local_vec,ham_pauli_one_term, depth_list, init_param_list_local, num_states_list): #e.g., ["I", "Z"]
                #リスト形式のパウリを演算子に変更
                ham_pauli_list_for_TF = [ham_pauli_one_tensor]
                ham_coeff_list_for_TF = [1.0]
                ope_temp = self.list_to_term(ham_pauli_list_for_TF, ham_coeff_list_for_TF)
                #Todo N^m(u)生成
                N_mu_temp_list.append(self.get_Nmu_from_circuits(hd_local, h_local, Hamiltonian=ope_temp, depth=depth, param=init_param , num_states=num_states, RealAmplitude=RealAmplitude, particle_number_preserving=particle_number_preserving))
            #N^m(u)のテンソル積とって係数かける。それらを足す
            H_hd_h_tilde += ham_coeff * reduce(np.kron, N_mu_temp_list)
                
        return H_hd_h_tilde

    def get_Nmu_from_circuits(self, hd_local, h_local, Hamiltonian, depth, param, num_states, RealAmplitude=False , particle_number_preserving=False): #tested #PNny
        #Nmuの計算
        #todo num_qubitsから引く
        init_H_array = np.array(Hamiltonian.to_matrix())
        if num_states is None:
            num_states = 2**Hamiltonian.num_qubits
        Nmu_array = np.empty((num_states, num_states)).astype("complex128") 
        for id in range(num_states):
            for i in range(num_states):
                #状態h',h準備
                state_hd_list = hd_local + list(map(int,format(id, "0"+str(Hamiltonian.num_qubits-len(hd_local))+"b")))  #indexを二進数にしてリストに変換
                if particle_number_preserving:
                    bound_circuit_id = handmade_particle_preserving_ansatz(Hamiltonian.num_qubits, depth, init_state=state_hd_list).bind_parameters(param)
                else:
                    bound_circuit_id = handmade_HEA(Hamiltonian.num_qubits, depth, init_state=state_hd_list, RealAmplitude=RealAmplitude).bind_parameters(param)                    
                statevector_id = Statevector(bound_circuit_id).data

                state_h_list =  h_local + list(map(int,format(i, "0"+str(Hamiltonian.num_qubits-len(h_local))+"b")))  #indexを二進数にしてリストに変換
                if particle_number_preserving:
                    bound_circuit_i = handmade_particle_preserving_ansatz(Hamiltonian.num_qubits, depth, init_state=state_h_list).bind_parameters(param)
                else:
                    bound_circuit_i = handmade_HEA(Hamiltonian.num_qubits, depth, init_state=state_h_list, RealAmplitude=RealAmplitude).bind_parameters(param)                    
                statevector_i = Statevector(bound_circuit_i).data

                #期待値取得
                H_hd_h_temp = np.conjugate(statevector_id) @ init_H_array @ statevector_i
                Nmu_array[id][i] = H_hd_h_temp

        return Nmu_array

    def set_parameters_for_ssvqeall(self, weight_list, depth_network_list, network, h_vec_list, num_states_intra_list, RealAmplitude, particle_number_preserving=False): #tested
        self.weight_list = weight_list
        self.depth_network_list = depth_network_list
        self.network = network
        self.h_vec_list = h_vec_list
        self.num_states_intra_list = num_states_intra_list
        self.RealAmplitude = RealAmplitude
        self.particle_number_preserving = particle_number_preserving

    def get_extracted_local_pauli_coeff_list(self,ham_pauli_list,ham_coeff_list): #tested
        #localをまたがるパウリがある場合削除。すべてIの場合も削除
        local_pauli_num=0

        pauli_local_list=[]
        coeff_local_list=[]

        for pauli_list, coeff in self.zip_equal(ham_pauli_list,ham_coeff_list):
            local_pauli_num=0
            for single_pauli_list in pauli_list:
                if ("X" in single_pauli_list) or ("Y" in single_pauli_list) or ("Z" in single_pauli_list):
                    local_pauli_num+=1
                else:
                    pass
            if local_pauli_num==1:
                pauli_local_list.append(pauli_list)
                coeff_local_list.append(coeff)
        
        return pauli_local_list,coeff_local_list

    
    def get_ham_intra_list(self, extracted_pauli_list, extracted_coeff_list, passI=True): #tested
        #計算するパウリと係数のリストをローカルテンソルごとの要素となるよう入れ替え、I削除
        #リストの行列入れ替え
        transposed_pauli_list=[list(x) for x in self.zip_equal(*extracted_pauli_list)]

        pauli_local_list=[]
        coeff_local_list=[]
        
        #すべての要素がIのものを削除し、係数を付加
        for pauli_list in transposed_pauli_list:
            pauli_local_list_temp=[]
            coeff_local_list_temp=[]
            for single_pauli_list,coeff in self.zip_equal(pauli_list,extracted_coeff_list):
                #すべての要素がIならパス
                if passI:
                    if not all([pauli=="I" for pauli in single_pauli_list]):
                        pauli_local_list_temp.append(single_pauli_list)
                        coeff_local_list_temp.append(coeff)
                else:
                    pauli_local_list_temp.append(single_pauli_list)
                    coeff_local_list_temp.append(coeff)
            pauli_local_list.append(pauli_local_list_temp) 
            coeff_local_list.append(coeff_local_list_temp)
            if len(pauli_local_list_temp) == 0:
                raise Exception("The operator element in a tensor is empty")

        return pauli_local_list, coeff_local_list


    def get_h_vec(self,h_temp, num_qubits, num_subsystem, network, num_states_intra_list): #tested
        #10進数から対応する\vec{h}を生成
        h_temp_list = list(map(int,format(h_temp, "0"+str(num_qubits)+"b")))
        h_splitted_list =[]

        idx = 0
        m = 0
        while True:
            if m == num_subsystem:
                h_splitted_list.append(h_temp_list[idx:])
                break
            num_qubits_m = len(network[m])
            h_splitted_list.append(h_temp_list[idx:idx+num_qubits_m-int(math.log2(num_states_intra_list[m]))])
            idx += num_qubits_m-int(math.log2(num_states_intra_list[m]))
            m += 1
        return h_splitted_list
            
    def get_H_hd_h_global(self,hd_global, h_global, H_hd_h_tilde, depth, init_param_global, RealAmplitude=False, particle_number_preserving=False): #tested
        #グローバル縮約

        num_qubits = int(math.log2(len(H_hd_h_tilde[0])))

        state_hd_list = hd_global
        if particle_number_preserving:
            bound_circuit_hd = handmade_particle_preserving_ansatz(num_qubits, depth, init_state=state_hd_list).bind_parameters(init_param_global)
        else:
            bound_circuit_hd = handmade_HEA(num_qubits, depth, init_state=state_hd_list, RealAmplitude=RealAmplitude).bind_parameters(init_param_global)            
        statevector_hd = Statevector(bound_circuit_hd).data

        state_h_list = h_global  #indexを二進数にしてリストに変換
        if particle_number_preserving:
            bound_circuit_h = handmade_particle_preserving_ansatz(num_qubits, depth, init_state=state_h_list).bind_parameters(init_param_global)            
        else:
            bound_circuit_h = handmade_HEA(num_qubits, depth, init_state=state_h_list, RealAmplitude=RealAmplitude).bind_parameters(init_param_global)
        statevector_h = Statevector(bound_circuit_h).data

        #期待値取得
        H_hd_h = np.conjugate(statevector_hd) @ H_hd_h_tilde @ statevector_h
        return H_hd_h 
        
    def get_matrix_element_H_HT(self,hd_vec, h_vec, ham_pauli_list, ham_coeff_list, depth_network_list, init_param_list, num_states_intra_list, RealAmplitude=False, particle_number_preserving=False): #tested
        #HとパラメーターからH_h'hを得る
        hd_vec_temp = copy.deepcopy(hd_vec)
        hd_global = hd_vec_temp.pop(-1) #hd_global_vecにはglobal, hd_vecはlocalな情報のみのこる
        hd_vec_local = hd_vec_temp

        h_vec_temp = copy.deepcopy(h_vec)
        h_global = h_vec_temp.pop(-1)
        h_vec_local = h_vec_temp

        init_param_list_temp = copy.deepcopy(init_param_list)
        init_param_global = init_param_list_temp.pop(-1)
        init_param_list_local = init_param_list_temp

        #Nmuを作成し、そこから非エルミート有効演算子\tilde{H_h'h}を得る
        H_hd_h_tilde = self.get_H_hd_h_tilde(hd_vec_local, h_vec_local, ham_pauli_list, ham_coeff_list, depth_list=depth_network_list[0], init_param_list_local=init_param_list_local, num_states_list=num_states_intra_list, RealAmplitude=RealAmplitude, particle_number_preserving=particle_number_preserving)

        #グローバル縮約
        H_hd_h = self.get_H_hd_h_global(hd_global, h_global, H_hd_h_tilde, depth=depth_network_list[1], init_param_global=init_param_global, RealAmplitude=RealAmplitude, particle_number_preserving=particle_number_preserving)
        return H_hd_h

    def get_overlap_HT_HF(self, hd_vec, h_bin, depth_network_list, init_param_list, num_states_intra_list, num_qubits, num_subsystem, network,  RealAmplitude=False, particle_number_preserving=False): #tested
        #HTとFock状態の重なりを得る
        hd_vec_temp = copy.deepcopy(hd_vec)
        hd_global = hd_vec_temp.pop(-1) #hd_global_vecにはglobal, hd_vecはlocalな情報のみのこる
        hd_vec_local = hd_vec_temp

        init_param_list_temp = copy.deepcopy(init_param_list)
        init_param_global = init_param_list_temp.pop(-1)
        init_param_list_local = init_param_list_temp

        #射影状態をテンソルの形式に直す
        proj_network = self.get_pauli_network_split(h_bin, network)

        #Nmuを作成し、そこから非エルミート有効演算子\tilde{overlap_h'}を得る
        overlap_h_tilde = self.get_overlap_h_tilde(hd_vec_local, proj_network, depth_list=depth_network_list[0], init_param_list_local=init_param_list_local, num_states_list=num_states_intra_list, RealAmplitude=RealAmplitude, particle_number_preserving=particle_number_preserving)

        #グローバル縮約
        overlap = self.get_overlap_h_global(hd_global,  overlap_h_tilde, depth=depth_network_list[1], init_param_global=init_param_global, RealAmplitude=RealAmplitude, particle_number_preserving=particle_number_preserving)
        return overlap

    def get_overlap_h_tilde(self, hd_local, proj_network, depth_list, init_param_list_local, num_states_list, RealAmplitude=False, particle_number_preserving=False):
        #overlap_h_tildeを計算
        overlap_h_tilde = 0

        N_mu_temp_list = []            
        for hd_local,  proj_list, depth, init_param, num_states in self.zip_equal(hd_local, proj_network[0], depth_list, init_param_list_local, num_states_list): #e.g., ["0", "1"]
            #N^m生成
            #psi_Tのproj_ope_bin番目の成分を取り出す
            N_mu_temp_list.append(self.get_overlap_Nmu_from_circuits(hd_local, proj_list, depth=depth, param=init_param , num_states=num_states, RealAmplitude=RealAmplitude, particle_number_preserving=particle_number_preserving))
        #N^m(u)のテンソル積とって係数かける。それらを足す
        overlap_h_tilde = reduce(np.kron, N_mu_temp_list)
                
        return overlap_h_tilde

    def get_overlap_Nmu_from_circuits(self, hd_local, proj_list, depth, param, num_states, RealAmplitude=False , particle_number_preserving=False): #tested #PNny
        #Nmuの計算
        if num_states is None:
            num_states = 2**len(proj_list)
        Nmu_array = np.empty((num_states)).astype("complex128") 
        for id in range(num_states):
            #状態h',h準備
            state_hd_list = hd_local + list(map(int,format(id, "0"+str(len(proj_list)-len(hd_local))+"b")))  #indexを二進数にしてリストに変換
            if particle_number_preserving:
                bound_circuit_id = handmade_particle_preserving_ansatz(len(proj_list), depth, init_state=state_hd_list).bind_parameters(param)
            else:
                bound_circuit_id = handmade_HEA(len(proj_list), depth, init_state=state_hd_list, RealAmplitude=RealAmplitude).bind_parameters(param)                    
            statevector_id = Statevector(bound_circuit_id).data
            
            #期待値取得
            overlap_hd_temp = np.conjugate(statevector_id)[int("0b"+"".join(proj_list),2)]
            Nmu_array[id] = overlap_hd_temp

        return Nmu_array


    def get_overlap_h_global(self, hd_global,  overlap_h_tilde, depth, init_param_global, RealAmplitude=False, particle_number_preserving=False):
        #グローバル縮約

        num_qubits = int(math.log2(len(overlap_h_tilde)))

        state_hd_list = hd_global
        if particle_number_preserving:
            bound_circuit_hd = handmade_particle_preserving_ansatz(num_qubits, depth, init_state=state_hd_list).bind_parameters(init_param_global)
        else:
            bound_circuit_hd = handmade_HEA(num_qubits, depth, init_state=state_hd_list, RealAmplitude=RealAmplitude).bind_parameters(init_param_global)            
        statevector_hd = Statevector(bound_circuit_hd).data

        #期待値取得
        overlap = np.conjugate(statevector_hd) @ overlap_h_tilde
        return overlap


    def term_to_list(self,Hamiltonian): #tested
        #Qiskitのハミルトニアンの係数とパウリをリスト形式にする
        try:
            ham_coeff_list=list(Hamiltonian.coeffs)
            ham_pauli_list=list(Hamiltonian.primitive.paulis.settings["data"])
        except: 
            print("only one parameter is included in the Hamiltonian")
            ham_coeff_list=[Hamiltonian.coeff]
            ham_pauli_list=[Hamiltonian.primitive.settings["data"]]
        
        
        return ham_coeff_list, ham_pauli_list

    def list_to_term(self,ham_pauli_list, ham_coeff_list): #tested
        #ハミルトニアンの係数とパウリのリストをQiskit形式にする
        ham_char = ""
        for i, (ham_pauli, ham_coeff) in enumerate(self.zip_equal(ham_pauli_list, ham_coeff_list)):
            ham_char = ham_char + "(" + str(ham_coeff) + "*"
            for j, pauli in enumerate(ham_pauli):
                ham_char = ham_char + pauli
                if  j != len(ham_pauli) -1:
                    ham_char = ham_char + "^"
                else:
                    ham_char = ham_char + ")"
            if  i != len(ham_pauli_list) -1:
                ham_char = ham_char + "+"
        return eval(ham_char)


    def SSVQE_cost_alltensor(self, params): #tested
        #全テンソル一気にSSVQE
        if type(params) is np.ndarray:
            params = params.tolist()

        params_splitted_list = self.param_split(params, self.depth_network_list, self.network, self.h_vec_list, self.RealAmplitude, particle_number_preserving=self.particle_number_preserving)

        cost = 0
        for h_vec, weight in self.zip_equal(self.h_vec_list, self.weight_list):
            cost +=  weight * self.get_matrix_element_H_HT(h_vec, h_vec, ham_pauli_list=self.ham_pauli_list, ham_coeff_list=self.ham_coeff_list, depth_network_list=self.depth_network_list, init_param_list=params_splitted_list, num_states_intra_list=self.num_states_intra_list, RealAmplitude = self.RealAmplitude, particle_number_preserving = self.particle_number_preserving) 
        return cost.real

    
    def callback_SSVQE_cost_alltensor(self, params): #tested
        #SSVQEのcallback
        if self.count_SSVQE == 0:
            num_states = len(self.h_vec_list)
            print("Iter Time Cost", *["State_"+str(i) for i in range(num_states)])

        if type(params) is np.ndarray:
            params = params.tolist()

        params_splitted_list = self.param_split(params, self.depth_network_list, self.network, self.h_vec_list,RealAmplitude = self.RealAmplitude, particle_number_preserving=self.particle_number_preserving)

        cost = 0
        expectation_list = []
        for h_vec, weight in self.zip_equal(self.h_vec_list, self.weight_list):
            expectation = self.get_matrix_element_H_HT(h_vec, h_vec,ham_pauli_list=self.ham_pauli_list, ham_coeff_list=self.ham_coeff_list, depth_network_list=self.depth_network_list, init_param_list=params_splitted_list, num_states_intra_list=self.num_states_intra_list, RealAmplitude = self.RealAmplitude, particle_number_preserving=self.particle_number_preserving) 
            cost +=  weight * expectation
            expectation_list.append(expectation.real)
        print(self.count_SSVQE, time.time()-self.init_t ,cost.real, *expectation_list)  
        self.count_SSVQE += 1  
        return None


    def param_split(self, params, depth_network_list, network, h_vec_list, RealAmplitude=False, particle_number_preserving=False): #tested
        #パラメーターをネットワーク形式に分割
        #todo realamplitude デフォルトをselfに変える
        param_splitted = []
        num_parameters_before = 0
        num_parameters_after = 0
        if particle_number_preserving:
            for depth_m, network_m in self.zip_equal(depth_network_list[0], network):
                num_qubits_m = len(network_m)
                num_parameters_after += depth_m * (num_qubits_m-1)
                param_splitted.append(np.array(params[num_parameters_before:num_parameters_after]))
                num_parameters_before = num_parameters_after

            num_qubits_global = len(h_vec_list[0][-1])
            num_parameters_after += depth_network_list[1] * (num_qubits_global-1)
            param_splitted.append(np.array(params[num_parameters_before:num_parameters_after]))
        elif RealAmplitude:
            for depth_m, network_m in self.zip_equal(depth_network_list[0], network):
                num_qubits_m = len(network_m)
                num_parameters_after += depth_m * num_qubits_m +num_qubits_m
                param_splitted.append(np.array(params[num_parameters_before:num_parameters_after]))
                num_parameters_before = num_parameters_after

            num_qubits_global = len(h_vec_list[0][-1])
            num_parameters_after += depth_network_list[1] * num_qubits_global +num_qubits_global
            param_splitted.append(np.array(params[num_parameters_before:num_parameters_after]))
        else:
            for depth_m, network_m in self.zip_equal(depth_network_list[0], network):
                num_qubits_m = len(network_m)
                num_parameters_after += depth_m * num_qubits_m*2 +num_qubits_m*2
                param_splitted.append(np.array(params[num_parameters_before:num_parameters_after]))
                num_parameters_before = num_parameters_after

            num_qubits_global = len(h_vec_list[0][-1])
            num_parameters_after += depth_network_list[1] * num_qubits_global*2 +num_qubits_global*2
            param_splitted.append(np.array(params[num_parameters_before:num_parameters_after]))
        if num_parameters_after != len(params):
            raise Exception("number of splitted parameters != len(params)")
        return param_splitted

