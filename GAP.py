import numpy as np
# from jax import grad
from ase.io import *
from ase.data import covalent_radii
from ase.units import Bohr, Hartree
import random
from descriptors import *
# from scipy import sparse

atom_energy = {"Zn": -49117.02929728, "O": -2041.3604,
               "H": -13.63393, "Li": -7.467060138769*Hartree,
               "Na": -4421.77421255, "Cl": -12577.38265088}

def train(L,E_ref,K, lambda_reg =0.1, atom_energy = None, energy_keyword = "energy"):
    Lambda = lambda_reg*np.eye((K.shape[0]))
    up = np.linalg.multi_dot([L,E_ref])
    down = np.linalg.multi_dot([L,L.T,K]) + Lambda
    weights = np.linalg.solve(down,up)
    return weights

def calculateE(train,mol,weights,pars):
    K = kernel_matrix(pars = pars,mol_set1 = [mol],mol_set2=train)
    energy = np.matmul(K,weights)
    nAt  = len(mol)
    results = np.sum(energy)*Hartree
    return results




def calculateEF(train,mol,weights,pars,zeta = 2):
    # K = kernel_matrix(pars = pars,mol_set1 = [mol],mol_set2=train)
    desc1,der1 = DerDescriptor(mol,pars)
    desc2 = descriptor_atoms(train,pars)
    print("Derivatives are done")
    # print(der1.shape)
    K = kernel_matrix(pars = pars,mol_set1 = [mol],mol_set2=train)
    # dKdr = an_kernel_gradient(mol,train,pars,zeta=zeta)#kernel_der(desc1,der1,desc2,mol, zeta=2)
    dKdr = kernel_der(desc1,der1,desc2,mol, zeta=2)
    # print(dKdr.shape)
    print("Kernel derivatives are done")
    energy = np.matmul(K,weights)
    nAt  = len(mol)
    # dKdr = kernel_der
    f_k  = np.zeros((nAt,3))
    for j in range(nAt):
        for direction in range(3):
            # print(dKdr[j][direction].shape)
            eneg_drj = np.matmul(dKdr[j][direction],weights)
            # print("eneg_drj",eneg_drj.shape)
            # print("diggy",eneg_drj[0])
            # print("hole",np.matmul(dKdr[j][direction][0],self.weights))
            for i in range(nAt):
                f_k[j,direction] += -eneg_drj[i]
    results = {'energy':np.sum(energy)*Hartree,'forces':f_k*Hartree}
    return results

def prepLgap(dim1,dim2,mols):#,charges):
    # Qkqeq = np.zeros((len(self.Kernel.training_set_descriptors),len(self.Kernel.training_set)))
    Lgap = np.zeros((dim1,dim2))
    count_r, count_c = 0, 0
    for mol in mols:
        for at in mol:
            Lgap[count_r,count_c] = 1
            count_r += 1
        count_c += 1
    return Lgap


def get_energies(mols, atom_energy, energy_keyword = "energy"):
    energy = []
    for mol in mols:
        mol_energy = mol.info[energy_keyword]
        for element in atom_energy:
            no_of_A = np.count_nonzero(mol.symbols == element)
            mol_energy -= (no_of_A*atom_energy[element])
        energy.append(mol_energy/Hartree)
    return np.array(energy)



def kernel_matrix(pars = None, mol_set1=None,mol_set2=None,zeta=2):
    desc1 = descriptor_atoms(mol_set1,pars)
    desc2 = descriptor_atoms(mol_set2,pars)
    K = np.matmul(desc1,np.transpose(desc2))**zeta
    return K

def GAPLoss(w,K,L,Eref,sigma):
    Lt = L.T
    # print(Lt.shape, K.shape,w.shape)
    
    bra = np.linalg.multi_dot([Lt,K,w]) - Eref
    R = sigma*np.linalg.multi_dot([w.T,K,w])
    l = np.matmul(bra.T,bra)
    Loss = np.subtract(l,R)
    return Loss 



def get_energies_perAtom(mols, atom_energy, energy_keyword = "energy"):
    energy = []
    for mol in mols:
        mol_energy = mol.info[energy_keyword]
        for element in atom_energy:
            no_of_A = np.count_nonzero(mol.symbols == element)
            mol_energy -= (no_of_A*atom_energy[element])
        mol_energy = mol_energy#/Hartree
        energy.append(mol_energy/len(mol))
    return energy


def kernel_der(desc1,der1,train_desc,mol_set1, zeta=2):
    t1 = (np.matmul(desc1,np.transpose(train_desc))**(zeta-1))
    dKdr = []
    for iatom in range(len(mol_set1)):
        dKdri = []
        for direction in range(3):
            curDer = (der1[iatom][direction])
            t2 = np.matmul(curDer,np.transpose(train_desc))
            K = (zeta*np.multiply(t1,t2))
            dKdri.append(K)
        dKdr.append(dKdri)
#        print(dKdr_final)
    return np.array(dKdr)
