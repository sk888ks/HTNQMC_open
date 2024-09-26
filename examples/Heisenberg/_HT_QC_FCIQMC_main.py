import qiskit 
from qiskit_nature.operators.second_quantization.fermionic_op import FermionicOp
from qiskit_nature.mappers.second_quantization.jordan_wigner_mapper import JordanWignerMapper
from qiskit_nature.converters.second_quantization import QubitConverter

import numpy as np

import math

import itertools
import os

def get_mixed_energy(walker_set,H_hd_h_array, energy_init=None,iidx=0): #iidx init energy index
    #get mixed energy

    #set initial walker
    iidx_walker_set = None
    for walker in walker_set:
        if walker[0] == iidx:
            iidx_walker_set = walker

    if energy_init is None: #通常の場合
        energy = H_hd_h_array[iidx][iidx] #VQE result
    else: #HFなどの基準エネルギーを別に持っている場合
        energy = energy_init 
    for walker_h in walker_set: #walker_h: [wf idx, sign, num walker]
        if walker_h[0] == iidx: 
            pass
        else:
            try:
                energy += H_hd_h_array[walker_h[0]][iidx] * (walker_h[1]*walker_h[2]) / iidx_walker_set[2] 
            except TypeError: #if iidx_walker_set is None, return inf
                energy = "inf"

            #energy += H_hd_h_array[walker_h[0]][0] * walker_h[2] / walker_set[0][2] 
    return energy

def get_mixed_energy_multi_determinant(walker_set, H_hd_h_array, overlap_vec, rho=0.0): #iidx init energy index
    #get mixed energy
    
    denom = 0 #denominator
    numer = 0 #numerator
    for h, sign_h, num_walker_h in walker_set:
        # if abs(overlap_vec[h]) < rho:
        #     pass
        # else:
        #     denom += sign_h * num_walker_h * overlap_vec[h]
        denom += sign_h * num_walker_h * overlap_vec[h]

        #H_hd_h
        for hd, H_hd_h in enumerate(H_hd_h_array[:,h]):
            # if abs(H_hd_h) < rho and abs(overlap_vec[hd]) < rho:
            #     pass
            # else:
            #     numer +=  sign_h * num_walker_h * overlap_vec[hd] * H_hd_h
            if abs(H_hd_h) < rho:
                pass
            else:
                numer +=  sign_h * num_walker_h * overlap_vec[hd] * H_hd_h
    
    mixed_energy = numer/denom
    return mixed_energy

def get_num_walker_total(walker_set):
    #get number of walker total
    num_walker_total = 0
    for walker in walker_set:
        num_walker_total += walker[2]
    return num_walker_total 

def get_spawned_walker_set(dt, h, sign_h, num_walker_h, H_hd, absH_hd, rho):
    #get spawned walker

    spawned_walker_set = []
    for hd in range(len(absH_hd)):
        if h == hd: #対角成分はパス
            pass
        elif absH_hd[hd] < rho: #|Hh'h|<rhoならパス
            pass
        else:
            for _ in range(num_walker_h):
                p_s = dt * absH_hd[hd]  #/np.linalg.norm(absH_hd*dt)
                sign_hd = int(np.sign(H_hd[hd]))
                if  p_s > 1: #psが1以上ならfloor(ps)だけspawnして、psからpsの整数部分を引く                    
                    spawned_walker_set.append([hd, (-1)*(sign_h*sign_hd), math.floor(p_s)])
                    p_s -= math.floor(p_s) 
                if np.random.rand() < p_s: #確率以上ならspawn
                    spawned_walker_set.append([hd, (-1)*(sign_h*sign_hd), 1])
    return spawned_walker_set

def get_died_cloned_waler_set(dt, h, sign_h, num_walker_h,  H_hd, shift):
    #get died cloned walker

    died_cloned_waler = []
    p_h = (H_hd[h]-shift)*dt
    if p_h < 0:
        #clone each walker with probability |p_h|
        for _ in range(num_walker_h): 
            #todo? abs(p_h) >1の処理
            died_cloned_waler.append([h, sign_h, 1]) #コピー
            if np.random.rand() < abs(p_h): #ある確率で増やす
                died_cloned_waler.append([h, sign_h, 1])
    else:
        #kill each walker with probability p_h
        for _ in range(num_walker_h): 
            #todo? abs(p_h) >1の処理
            if np.random.rand() < p_h: #ある確率で殺す=コピーしない
                pass
            else:
                #コピー
                died_cloned_waler.append([h, sign_h, 1])
    return died_cloned_waler

def get_merged_walker_set(init_walker_set):
    #merge walker set
    
    #group by idx
    unique_idx = set([row[0] for row in init_walker_set])
    annihilated_walker_set = []
    annihilated_walker_set_temp = [[row for row in init_walker_set if row[0] == id] for id in unique_idx]

    for walker_one_idx in annihilated_walker_set_temp:
    #merge value of walkers in the same idx 
        walker_val_temp = 0
        for walker_with_sign_val in walker_one_idx:
            walker_val_temp += walker_with_sign_val[1] * walker_with_sign_val[2]

        if abs(walker_val_temp) != 0: #if walker value ==0, the walker is removed
            annihilated_walker_set.append([walker_with_sign_val[0],np.sign(walker_val_temp), abs(walker_val_temp)])

    return annihilated_walker_set

def QC_FCIQMC(evol_time, dt, H_hd_h_array, absH_hd_h_array, walker_set, shift = 0 ,numpy_seed = 0, fixed_num_walker = 10000, damping = 0.05, shift_step_A = 5, energy_init = None, init_energy_index = 0, rho = 0.0, mixed_energy_MD = False, overlap_vec = None):
    variable_shift = False
    result_list = []
    np.random.seed(numpy_seed)

    for iter in range(int(evol_time/dt)):
        if iter == 0:
            print("Iter Energy Num_walker Shift")
            with open(os.path.join(os.path.dirname(__file__),"QC_FCIQMC_result.txt"), mode = "w") as f:
                f.write("Iter Energy Num_walker Shift\n")
            with open(os.path.join(os.path.dirname(__file__),"QC_FCIQMC_walker_result.txt"), mode = "w") as f:
                f.write("Iter Walker\n")
        if mixed_energy_MD:
            if overlap_vec is None:
                raise Exception("overlap_vec is None")
            energy = get_mixed_energy_multi_determinant(walker_set,H_hd_h_array, overlap_vec, rho)
        elif energy_init is None:
            energy = get_mixed_energy(walker_set, H_hd_h_array, iidx=init_energy_index)
        else:
            energy = get_mixed_energy(walker_set, H_hd_h_array, energy_init, iidx=init_energy_index)
        num_walker_total = get_num_walker_total(walker_set)
        result_list.append([iter, energy, num_walker_total, shift])
        print(*result_list[-1])
        with open(os.path.join(os.path.dirname(__file__),"QC_FCIQMC_result.txt"), mode = "a") as f:
            for i in result_list[-1]:
                f.write(str(i))
                f.write(" ")
            f.write("\n")
        with open(os.path.join(os.path.dirname(__file__),"QC_FCIQMC_walker_result.txt"), mode = "a") as f:
            f.write(str(iter))
            f.write(" ")
            f.write(str(walker_set))
            f.write("\n")

        if (not variable_shift) and num_walker_total > fixed_num_walker: #check variable shift mode
            variable_shift = True
        if variable_shift and iter % shift_step_A == 0: #shift
            previous_num_walker = result_list[-1-shift_step_A][2]
            current_num_walker = result_list[-1][2]
            shift = shift - (damping/(shift_step_A*dt))*math.log((current_num_walker/previous_num_walker) ,math.e)

        spawned_walker_set = []
        died_cloned_waler = []
        for h, sign_h, num_walker_h in walker_set:
            #spawning
            H_hd = H_hd_h_array[:,h]
            absH_hd = absH_hd_h_array[:,h] #estimate |H_h'h|
            spawned_walker_set += get_spawned_walker_set(dt, h, sign_h, num_walker_h, H_hd, absH_hd, rho)
            spawned_walker_set = get_merged_walker_set(spawned_walker_set)

            #die clone
            died_cloned_waler += get_died_cloned_waler_set(dt, h, sign_h, num_walker_h,  H_hd, shift) 
            died_cloned_waler = get_merged_walker_set(died_cloned_waler)

        #annihilate the walkers psi_i with opposite signs
        walker_set_temp = spawned_walker_set + died_cloned_waler
        walker_set = get_merged_walker_set(walker_set_temp)

    return walker_set


def main():
    pass

if __name__ == "__main__":
    main()







