import numpy as np
# from jax import grad
from ase.io import *
from ase.data import covalent_radii
from ase.units import Bohr, Hartree
import random
from descriptors import *
# from scipy import sparse
import sparse
from GAP import *
from matscipy.neighbours import neighbour_list


def get_cutoff_padding(sigma):
    threshold = 0.001
    cutoff_padding = sigma * np.sqrt(-2 * np.log(threshold))
    return cutoff_padding

def build_nl(mol,r_cut,sigma):
    i, j = neighbour_list('ij', mol, cutoff=r_cut+get_cutoff_padding(sigma))
    for at in range(len(mol)):
        i = np.append(i,at)
        j = np.append(j,at)
    return i,j


def calculateEFSparse(train,mol,weights,pars,zeta = 2):
    # K = kernel_matrix(pars = pars,mol_set1 = [mol],mol_set2=train)
    der1, desc1 = DerDescriptorSparse(mol,pars)
    desc2 = descriptor_atoms(train,pars)
    print("Derivatives are done")
    K = np.matmul(desc1,np.transpose(desc2))**zeta
    dKdr = kernel_derSparse(desc1,der1,desc2,mol, zeta=2)
    print("Kernel derivatives are done")
    energy = np.matmul(K,weights)
    nAt  = len(mol)
    f_k  = np.zeros((nAt,3))
    for j in range(nAt):
        for direction in range(3):
            eneg_drj = np.matmul(dKdr[j][direction],weights)
            for i in range(nAt):
                f_k[j,direction] += -eneg_drj[i]
    results = {'energy':np.sum(energy)*Hartree,'forces':f_k*Hartree}
    return results

def kernel_derSparse(desc1,der1,train_desc,mol_set1, zeta=2):
    t1 = (np.matmul(desc1,np.transpose(train_desc))**(zeta-1))
    dKdr = []
    for iatom in range(len(mol_set1)):
        dKdri = []
        for direction in range(3):
            curDer = (der1[iatom][direction])
            curDer = curDer.todense()
            t2 = np.matmul(curDer,np.transpose(train_desc))
            K = (zeta*np.multiply(t1,t2))
            K = sparse.COO.from_numpy(K)
            dKdri.append(K)
        dKdr.append(dKdri)
#        print(dKdr_final)
    return dKdr