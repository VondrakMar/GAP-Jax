import numpy as np
from ase.io import *
from ase.data import covalent_radii
from ase.units import Bohr
import random
from qpac.data import atomic_numbers
from dscribe.descriptors import SOAP

np.random.seed(10)


class soapClass():
    def __init__(self):
        self.precomputed = {"training":[]}
        self.precomputedEl = {"training":[]}

    def compute(self,mols,name):
        soaps = self.descriptor_atoms(mols)
        self.precomputed[name] = soaps[0]
        self.precomputedEl[name] = soaps[1]

    def saveSOAP(self,name,nameSave):
        np.save(nameSave,self.precomputed[name])
        np.save(f'El{nameSave}',self.precomputedEl[name])
    
    def loadSOAP(self,name,nameSave):
        self.precomputed[name] = np.load(nameSave)
        self.precomputedEl[name] = np.load(f'El{nameSave}')

    def descriptor_atoms(mols):
        raise NotImplementedError

class dscribeSOAP(soapClass):
    def __init__(self, nmax, lmax,rcut, sigma,species,periodic):
        self.species = species
        self.soap = SOAP(
            periodic=periodic,
            r_cut=rcut,  
            n_max=nmax,
            sigma=sigma,
            species = species,
            l_max=lmax,
            sparse = True)
        super().__init__()


    def descriptor_atoms(self,molecules,n_jobs=1):
        soap_atoms = []
        element_list = []
        soaps_new = self.soap.create(molecules,n_jobs = n_jobs)
        soaps_new = soaps_new.todense()
        print("hell",type(soaps_new))
        if len(molecules) == 1:
            soaps_new = [soaps_new]
        for ids,m in enumerate(soaps_new):
            element_list.extend([a.symbol for a in molecules[ids]])
            norms = np.linalg.norm(m,axis=-1)
            soap_atoms.extend(m/norms[:,None])
        
        return np.array(soap_atoms), np.array(element_list)

    def _descr_deriv_atomsSparse(self,molecule, deriv = "numerical"):
        der, desc = self.soap.derivatives(molecule,method=deriv,attach=True) #analytical derivatives in SOAPs are not usable with 1.2.0 version        '''
        element_list = []
        element_list.extend([a.symbol for a in molecule])
        element_list = np.array(element_list)
        norms = np.linalg.norm(desc,axis=-1)
        norm_desc = desc/norms[:,None]
        norms2 = norms**2
        deriv_final = np.moveaxis(der,[0,1],[2,0])
        # normal_soap_vector = np.array(normal_soap_vector)
        hold = (np.einsum('ij,arkj->arik',desc,deriv_final,optimize="greedy"))
        vd = np.einsum('akii->aki',hold,optimize="greedy")
        r1 = np.einsum('akl,lj,l->aklj',vd,desc,1/norms,optimize="greedy")
        f1 = np.einsum('aklj,l->aklj',deriv_final,norms,optimize="greedy")
        normDer = np.einsum('aklj,l->aklj',f1-r1,1/norms2,optimize="greedy")
        
        return norm_desc, normDer, element_list

