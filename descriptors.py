import numpy as np
# import scipy.sparse as sp
from dscribe.descriptors import SOAP
import sparse

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

def DerDescriptorSparse(molecule,pars):
    soap = SOAP(
        periodic=pars["periodic"],
        r_cut=pars["rcut"],  
        n_max=pars["nmax"],
        sigma=pars["sigma"],
        species = pars["species"],
        l_max=pars["lmax"],
        sparse=True)
    soap_atoms = []
    # element_list = []
    der, desc = soap.derivatives([molecule],method="auto",attach=True) #analytical derivatives in SOAPs are not usable with 1.2.0 version        '''
    desc = desc.todense()
    norms = np.linalg.norm(desc,axis=-1)
    norm_desc = desc/norms[:,None]
    norm_desc = sparse.COO.from_numpy(norm_desc)
    # norms2 = norms**2
    deriv_final = sparse.moveaxis(der,[0,1],[2,0])
    # normal_soap_vector = np.array(normal_soap_vector)
    # hold = sparse.COO.from_numpy(hold)
    invnorms = 1/norms
    norms2 = norms**2
    invnorms2 = 1/norms2
    desc = sparse.COO.from_numpy(desc)
    norms = sparse.COO.from_numpy(norms)
    norms2 = sparse.COO.from_numpy(norms2)
    invnorms = sparse.COO.from_numpy(invnorms)
    invnorms2 = sparse.COO.from_numpy(invnorms2)
    print("I am fucking here")
    hold = sparse.einsum('ij,arkj->arik',desc,deriv_final)
    vd = sparse.einsum('akii->aki',hold)
    r1 = sparse.einsum('akl,lj,l->aklj',vd,desc,invnorms)
    f1 = sparse.einsum('aklj,l->aklj',deriv_final,norms)
    normDer = sparse.einsum('aklj,l->aklj',f1-r1,invnorms2)
    # print(type(hold), type(vd), type(r1), type(f1), type(normDer))
    return normDer,norm_desc 

def DerDescriptor(molecule,pars):
    soap = SOAP(
        periodic=pars["periodic"],
        r_cut=pars["rcut"],  
        n_max=pars["nmax"],
        sigma=pars["sigma"],
        species = pars["species"],
        l_max=pars["lmax"],
        sparse=False)
    soap_atoms = []
    # element_list = []
    der, desc = soap.derivatives([molecule],method="auto",attach=True) #analytical derivatives in SOAPs are not usable with 1.2.0 version        '''
    # desc = desc.todense()
    norms = np.linalg.norm(desc,axis=-1)
    norm_desc = desc/norms[:,None]
    norms2 = norms**2
    deriv_final = np.moveaxis(der,[0,1],[2,0])
    # normal_soap_vector = np.array(normal_soap_vector)
    print("I am fucking here")
    hold = (np.einsum('ij,arkj->arik',desc,deriv_final,optimize="greedy"))
    vd = np.einsum('akii->aki',hold,optimize="greedy")
    r1 = np.einsum('akl,lj,l->aklj',vd,desc,1/norms,optimize="greedy")
    f1 = np.einsum('aklj,l->aklj',deriv_final,norms,optimize="greedy")
    normDer = np.einsum('aklj,l->aklj',f1-r1,1/norms2,optimize="greedy")
    return norm_desc, normDer


def an_kernel_gradient(mol_set1,train_set, pars, zeta=2):
    soap = SOAP(
        periodic=pars["periodic"],
        r_cut=pars["rcut"],  
        n_max=pars["nmax"],
        sigma=pars["sigma"],
        species = pars["species"],
        l_max=pars["lmax"],
        sparse=False)
    desc2 = descriptor_atoms(train_set,pars)
    soap_atoms = []
    der, desc = soap.derivatives([mol_set1],method="auto",attach=True) #analytical derivatives in SOAPs are not usable with 1.2.0 version        '''
    norms = np.linalg.norm(desc,axis=-1)
    norm_desc = desc/norms[:,None]
    norms2 = norms**2
    deriv_final = np.moveaxis(der,[0,1],[2,0])
    # deriv_final = []    
    dKdr = [] 
    for iatom in range(len(mol_set1)):
        # print("new atom")
        dKdri = []
        for direction in range(3):
            inter = deriv_final[iatom][direction] # derivation of atom in 3 direction based on all other atoms
            tt1 = (np.matmul(desc,np.transpose(inter))) 
            vd = np.diag(tt1)
            tt2 = np.multiply(desc,vd[:,None])
            r1 = tt2/norms[:,None] # each atomistic SOAP descriptor is normalized 
            f1 = np.multiply(inter,norms[:,None]) 
            norms2 = norms**2
            normDer=(f1-r1)/norms2[:,None]
            t1 = (np.matmul(norm_desc,np.transpose(desc2))**(zeta-1))
            t2 = (np.matmul(normDer,np.transpose(desc2)))
            K = zeta*np.multiply(t1,t2)
            dKdri.append(K)
        dKdr.append(dKdri)
    return dKdr
