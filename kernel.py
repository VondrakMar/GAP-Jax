import numpy as np
from ase.io import *
from ase.data import covalent_radii
from ase.units import Bohr
from qpac.funct import CURdecomposition
import random
# from quippy.descriptors import Descriptor
from dscribe.descriptors import SOAP
np.random.seed(10)

import time

class kQEqKernel():
    def __init__(self,
                 multi_SOAP=False, 
                 Descriptor=None, 
                 training_set=None, 
                 training_system_charges = None, 
                 perEl = True, 
                 sparse=False, 
                 sparse_count=None, 
                 sparseSelect = None,
                 sparse_method="CURel"):

        self.sparse=None
        self.Descriptor = Descriptor
        self.multi_SOAP = multi_SOAP
        self.elements = self.Descriptor.species
        self.sparse = sparse
        self.sparse_count = sparse_count
        self.perEl = perEl
        if training_set is not None:
            self.training_set = training_set
            if len(self.Descriptor.precomputed["training"]) == 0:
                self.Descriptor.compute(self.training_set,"training")
            self.training_set_descriptors,self.training_set_elements = self.Descriptor.precomputed["training"],self.Descriptor.precomputedEl["training"],
            self.nAt_train = len(self.training_set_descriptors)
            if training_system_charges == None:
                self.training_system_charges = [0 for temp in training_set] 
            else: 
                self.training_system_charges = training_system_charges
        else:
            self.training_set_descriptors = None
            self.training_set = None
            self.training_system_charges = None
            self.training_set_elements = None
        if self.sparse == True and training_set is not None:
            self.sparse_method = sparse_method
            if sparse_count == None:
                if  self.nAt_train  < 1000:
                    self.sparse_count =  self.nAt_train
                    print("Training set has less than 1000 points, training will be done with the full set")
                else:
                    self.sparse_count = 1000
            else:
                self.sparse_count = int(sparse_count)
            if sparse_method == "CURel":
                self.sparse_points = self._create_representationCUR_elements(self.sparse_count)
            elif sparse_method == "prepicked":
                self.sparse_points = sparseSelect
                self._create_representationPrepicked(nameS="KQEqSOAP.npy",nameEl="KQEqEl.npy")
            elif sparse_method == None:
                pass    
            else:
                print("Specify sparse method!")
        elif self.sparse == False and training_set is not None        :
            self.representing_set_descriptors = self.training_set_descriptors
            self.representing_set_elements = self.training_set_elements
        
        #######################################################################
        if self.training_set is not None:
            self.train_descs = {}
            self.repres_descs = {}
            ldesc = len(self.training_set_descriptors[0])
            for el in self.elements:

                self.repres_descs[el] = []
                if self.perEl == True:
                    self.train_descs[el] = []
                    for countTrain in range(len(self.training_set_descriptors)):
                        if el == self.training_set_elements[countTrain]:
                            self.train_descs[el].append(self.training_set_descriptors[countTrain])
                        else:
                            self.train_descs[el].append(np.zeros(ldesc))
                    self.train_descs[el] = np.array(self.train_descs[el])
                else:
                    self.train_descs[el] = self.training_set_descriptors
                for countRepre in range(len(self.representing_set_descriptors)):
                    if el == self.representing_set_elements[countRepre]:
                        self.repres_descs[el].append(self.representing_set_descriptors[countRepre])
                    else:
                        self.repres_descs[el].append(np.zeros(ldesc))
                self.repres_descs[el] = np.array(self.repres_descs[el])
        #######################################################################



    def _create_representationPrepicked(self, nameS="KQEqSOAP.npy",nameEl="KQEqEl.npy"):
        self.representing_set_descriptors = np.load(nameS)
        self.representing_set_elements = np.load(nameEl)

    def _create_representationCUR_elements(self,sparse_count):
        specs = []
        for mol in self.training_set:
            for atom in mol:
                specs.append(atom.symbol)
        soapVecs = {a:[] for a in self.Descriptor.species}
        soapIds = {a:[] for a in self.Descriptor.species}
        for id, el in enumerate(specs):
            soapVecs[el].append(self.training_set_descriptors[id])
            soapIds[el].append(id)
        self.representing_set_descriptors = []
        self.representing_set_elements = []
        pickedAll = []
        for element in self.Descriptor.species:
            picked = CURdecomposition(sparse_count=sparse_count,mat=soapVecs[element])
            self.representing_set_descriptors.extend([soapVecs[element][pos] for pos in picked]) # this has to be changed for multi SOAP
            self.representing_set_elements.extend([element for pos in picked])
            pickedAll.extend([soapIds[element][pos] for pos in picked])
        self.sparse = True
        return pickedAll
    
    def saveSparse(self, nameS="KQEqSOAP.npy",nameEl="KQEqEl.npy"):
        np.save(nameS, self.representing_set_descriptors)
        np.save(nameEl, self.representing_set_elements)

        
class SOAPKernel(kQEqKernel):
    def __init__(self,multi_SOAP=False,
                 Descriptor=None,
                 training_set=None, 
                 training_system_charges = None,
                 perEl = True, 
                 sparse=False, 
                 sparse_count=None, 
                 sparseSelect = None,
                 sparse_method="CUR",
                 zeta=2):
        super().__init__(multi_SOAP=multi_SOAP,
                         Descriptor=Descriptor,
                         training_set=training_set, 
                         training_system_charges = training_system_charges,
                         perEl = perEl,
                         sparse = sparse,
                         sparse_count=sparse_count,
                         sparseSelect = sparseSelect,
                         sparse_method=sparse_method)
        self.zeta = zeta

    def kernel_matrix(self,mol_set1=None,mol_set2=None,kerneltype='general'):
        zeta = self.zeta
        if kerneltype=='general':
            if mol_set2 == None:
                desc1 = self.Descriptor.descriptor_atoms(mol_set1)
                dim1  = len(desc1)
                K = np.matmul(desc1,np.transpose(desc1))**zeta
            else:
                desc1 = self.Descriptor.descriptor_atoms(mol_set1)
                desc2 = self.Descriptor.descriptor_atoms(mol_set2)
                dim1  = len(desc1)
                dim2  = len(desc2)
                K = np.matmul(desc1,np.transpose(desc2))**zeta
            return K
        elif kerneltype=='predicting':
            desc1,elems1 = self.Descriptor.descriptor_atoms(mol_set1)
            ldesc = len(desc1[0])
            el_descs = {}

            for el in self.elements:
                el_descs[el] = []
                if self.perEl == True:
                    for countRepre in range(len(desc1)):
                        if el == elems1[countRepre]:
                            el_descs[el].append(desc1[countRepre])
                        else:
                            el_descs[el].append(np.zeros(ldesc))
                    el_descs[el] = np.array(el_descs[el])
                elif self.perEl == False:
                    el_descs[el] = desc1
            K = np.zeros((len(desc1),len(self.representing_set_descriptors)))
            for el in self.elements:
                K_temp = np.matmul(np.array(el_descs[el]),np.transpose(np.array(self.repres_descs[el])))**zeta
                K += K_temp
            return K
        elif kerneltype=='training':
            K_nm = np.zeros((len(self.training_set_descriptors),len(self.representing_set_descriptors)))
            K_mm = np.zeros((len(self.representing_set_descriptors),len(self.representing_set_descriptors)))
            for el in self.elements:
                K_nm += np.matmul(self.train_descs[el],np.transpose(self.repres_descs[el]))**zeta
                K_mm += np.matmul(self.repres_descs[el],np.transpose(self.repres_descs[el]))**zeta   
            
            return K_nm, K_mm
            # print(train_descs)

    def _calculate_function(self,mol_set1,zeta=2):
        desc1,der1,elems1 = self.Descriptor._descr_deriv_atoms(mol_set1)#,deriv="numerical")
        K_final = np.zeros((len(desc1),len(self.representing_set_descriptors)))
        ldesc = len(desc1[0])
        lder = der1[0].shape
        el_descs = {}
        el_ders = {}
        dKdr = {} #  np.zeros((len(desc1), 3, len(desc1), len(self.representing_set_descriptors)))
        
        for el in self.elements:
            el_descs[el] = []
            el_ders[el] = []
            dKdr[el] = []
            #dKdri[el] = []
            if self.perEl == True:
                for countRepre in range(len(desc1)):
                    if el == elems1[countRepre]:
                        el_descs[el].append(desc1[countRepre])
                        el_ders[el].append(der1[countRepre])
                    else:
                        el_descs[el].append(np.zeros(ldesc))
                        el_ders[el].append(np.zeros(lder))
            elif self.perEl == False:
                el_descs[el] = desc1
                el_ders[el] = der1
            el_descs[el] = np.array(el_descs[el])
            el_ders[el] = np.array(el_ders[el])
            K_temp = np.matmul(np.array(el_descs[el]),np.transpose(np.array(self.repres_descs[el])))**zeta
            K_final += K_temp
            t1 = (np.matmul(el_descs[el],np.transpose(self.repres_descs[el]))**(zeta-1))
            for iatom in range(len(mol_set1)):
                dKdri = []
                for direction in range(3):
                    # K = np.zeros((len(desc1),len(self.representing_set_descriptors)))
                    curDer = (der1[iatom][direction])
                    t2 = np.matmul(curDer,np.transpose(self.repres_descs[el]))
                    K = (zeta*np.multiply(t1,t2))
                    dKdri.append(K)
                dKdr[el].append(dKdri)
        dKdr_final = np.zeros((len(desc1), 3, len(desc1), len(self.representing_set_descriptors)))
        for a in self.elements:
            dKdr_final += dKdr[a]
#        print(dKdr_final)
        return K_final,dKdr_final
        '''
        normal_soap_vector = []
        norms = []
        for sv in desc1:
            norm = np.linalg.norm(sv)
            normal_soap_vector.append(sv/norm)
            norms.append(norm)
        norms = np.array(norms)
        norms2 = norms**2
        K_final = np.matmul(normal_soap_vector,np.transpose(desc_train))**zeta     
        deriv_final = np.moveaxis(der1,[0,1],[2,0])

        normal_soap_vector = np.array(normal_soap_vector)
        hold = (np.einsum('ij,arkj->arik',desc1,deriv_final,optimize="greedy"))
        vd = np.einsum('akii->aki',hold,optimize="greedy")
        r1 = np.einsum('akl,lj,l->aklj',vd,desc1,1/norms,optimize="greedy")
        f1 = np.einsum('aklj,l->aklj',deriv_final,norms,optimize="greedy")
        normDer = np.einsum('aklj,l->aklj',f1-r1,1/norms2,optimize="greedy")
        #t1 = np.einsum('ij,lj->il',normal_soap_vector,desc_train,optimize="greedy")**(zeta-1)
        #t2 = np.einsum('akij,lj->akil',normDer,desc_train,optimize="greedy")
        #dKdr = zeta*np.einsum('akij,ij->akij',t2,t1,optimize="greedy")
#        t1 = np.matmul(normal_soap_vector,np.transpose(desc_train))**(zeta-1)
        t1 = np.einsum('ij,lj->il',normal_soap_vector,desc_train,optimize="greedy")**(zeta-1)
        dKdr = zeta*np.einsum('akij,lj,il->akil',normDer,desc_train,t1,optimize=['einsum_path', (0, 1), (0, 1)])        
        return K_final,dKdr
        '''

