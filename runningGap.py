from GAP import *
import matplotlib.pyplot as plt

import numpy as np
import random
random.seed(10)


def num_grad(mol,train,h=0.0001,direction=0,iatom=0):
    tmpmol = mol.copy()
    pos = tmpmol.get_positions()#/Bohr
    pos[iatom][direction] += h
    tmpmol.set_positions(pos)#*Bohr)
    Eplus = kernel_matrix(pars,mol_set1 = [tmpmol], mol_set2 = train, zeta=2)
    pos[iatom][direction] += -2.0*h
    tmpmol.set_positions(pos)#*Bohr)
    Eminus = kernel_matrix(pars,mol_set1 = [tmpmol], mol_set2 = train, zeta=2)
    pos[iatom][direction] += h
    tmpmol.set_positions(pos)#*Bohr)
    energy = (Eplus-Eminus)/(2.0*h)#*Bohr)
    return energy#*Hartree#/Bohr


# mols = read("water8Force.xyz@:",format="extxyz")
mols = read("water8Force.xyz@:",format="extxyz")
for mol in mols:
    mol.center(vacuum=10)

random.shuffle(mols)
print(len(mols))
train_set = mols[:100]
test_set = mols[100:103]
pars = {"periodic":False,
        "rcut":3.5,
        "nmax":4,
        "sigma":3.5/8,
        "species":['H','O'],
        "lmax":3}
lambda_reg = 0.1
# lambda_reg = 0.01
K = kernel_matrix(pars,mol_set1 = train_set, mol_set2 = train_set, zeta=2)
w = np.ones(K.shape[0])*0.3
E_ref = get_energies(train_set,atom_energy)
M = prepLgap(K.shape[0],len(train_set),train_set)
Loss = GAPLoss(w,K,M,E_ref,lambda_reg)
w = train(M,E_ref,K, lambda_reg =lambda_reg, atom_energy = atom_energy, energy_keyword = "energy")

dK = num_grad(test_set[0],train_set,iatom = 3,direction=2)
print(dK[5])
desc1,der1 = DerDescriptor(test_set[0],pars)
desc2 = descriptor_atoms(train_set,pars)
# print(der1.shape)
# dKdr = kernel_der(desc1,der1,desc2,test_set[0], zeta=2)
dKdr = an_kernel_gradient(test_set[0],train_set,pars,zeta=2)
print(dKdr[3][2][5])

# print("loss ", Loss)
'''
print(w)
LossMin = GAPLoss(w,K,M,E_ref,lambda_reg)
print("loss ", LossMin)
print("starting")
'''





# w = w + 0.2
'''
dLoss = grad(GAPLoss)
for i in range(len(w)):
    wtemp = w.copy()
    tempN = wtemp[i]+0.001
    wtemp = wtemp.at[i].set((tempN))
    Loss1 = GAPLoss(wtemp,K,M,E_ref,0.1)
    wtemp = wtemp.at[i].set(wtemp[i]-0.002)
    Loss2 = GAPLoss(wtemp,K,M,E_ref,0.1)
    anLoss = dLoss(w,K,M,E_ref,0.1)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
    print(anLoss[i])
    print((Loss1-Loss2)/0.002)
    print("$$$$$$$$$$$$$$$$$$$$$$$$$$")
'''
E_ref = get_energies_perAtom(mols=test_set,atom_energy = atom_energy)
E_test = []
F_test = []
F_ref = []


for a in test_set:
    res = calculateEF(train_set,a,w,pars)
    # E_test.append(res/len(a))
    E_test.append(res["energy"]/len(a))
    # E_test.append(res/len(a))
    F_test.extend(res["forces"])
    F_ref.extend(a.arrays["forces"])
print(E_ref)
print(E_test)
'''
plt.figure(figsize=(10,8), dpi=100)
plt.plot(E_ref,E_ref,alpha = 0.5,color="black")
plt.scatter(E_test,E_ref,alpha = 0.5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
plt.close()

plt.figure(figsize=(10,8), dpi=100)
plt.plot(F_ref,F_ref,alpha = 0.5,color="black")
plt.scatter(F_test,F_ref,alpha = 0.5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
plt.close()
'''
from buildStuff import make_water
print("createing water")
water = make_water(1.0, [5, 5, 5])
print(len(water))
res = calculateEF(train_set,water,w,pars)
print(res["energy"])
print(res["forces"][:10])
# E_test.append(res/len(a))
# E_test.append(res["energy"]/len(a))
# E_test.append(res/len(a))
# F_test.extend(res["forces"])
# F_ref.extend(a.arrays["forc//es"])