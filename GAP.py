import jax.numpy as np
from jax import grad
from ase.io import *
from ase.data import covalent_radii
from ase.units import Bohr, Hartree
import random
from dscribe.descriptors import SOAP


atom_energy = {"Zn": -49117.02929728, "O": -2041.3604,
               "H": -13.63393, "Li": -7.467060138769*Hartree,
               "Na": -4421.77421255, "Cl": -12577.38265088}

def train(L,E_ref,K, lambda_reg =0.1, atom_energy = None, energy_keyword = "energy"):
    Lambda = lambda_reg*np.eye((K.shape[0]))
    up = np.linalg.multi_dot([L,E_ref])
    down = np.linalg.multi_dot([L,L.T,K]) + Lambda
    weights = np.linalg.solve(down,up)
    return weights

def calculate(train,mol,weights,pars):
    K = kernel_matrix(pars = pars,mol_set1 = [mol],mol_set2=train)
    energy = np.matmul(K,weights)
    nAt  = len(mol)
    results = np.sum(energy)*Hartree
    return results

def prepLgap(dim1,dim2,mols):#,charges):
    # Qkqeq = np.zeros((len(self.Kernel.training_set_descriptors),len(self.Kernel.training_set)))
    Lgap = np.zeros((dim1,dim2))
    count_r, count_c = 0, 0
    for mol in mols:
        for at in mol:
            Lgap = Lgap.at[count_r,count_c].set(1)
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

def descriptor_atoms(molecules,pars):
    soap = SOAP(
        periodic=pars["periodic"],
        r_cut=pars["rcut"],  
        n_max=pars["nmax"],
        sigma=pars["sigma"],
        species = pars["species"],
        l_max=pars["lmax"])
    soap_atoms = []
    element_list = []
    soaps_new = soap.create(molecules)
    if len(molecules) == 1:
        soaps_new = [soaps_new]
    for ids,m in enumerate(soaps_new):
        element_list.extend([a.symbol for a in molecules[ids]])
        norms = np.linalg.norm(m,axis=-1)
        soap_atoms.extend(m/norms[:,None])
    return np.array(soap_atoms)#, np.array(element_list)

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