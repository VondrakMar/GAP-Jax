from GAP import *
from GAPsparse import *
import matplotlib.pyplot as plt

import numpy as np
import random
random.seed(10)


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
K = kernel_matrix(pars,mol_set1 = train_set, mol_set2 = train_set, zeta=2)
w = np.ones(K.shape[0])*0.3
E_ref = get_energies(train_set,atom_energy)
M = prepLgap(K.shape[0],len(train_set),train_set)
Loss = GAPLoss(w,K,M,E_ref,lambda_reg)
# print("loss ", Loss)
w = train(M,E_ref,K, lambda_reg =lambda_reg, atom_energy = atom_energy, energy_keyword = "energy")
# print(w)
LossMin = GAPLoss(w,K,M,E_ref,lambda_reg)
print("loss ", LossMin)
print("starting")



# E_ref = get_energies_perAtom(mols=test_set,atom_energy = atom_energy)
# E_test = []
# for a in test_set:
#     calculateEFSparse(train_set,a,w,pars)
E_ref = get_energies_perAtom(mols=test_set,atom_energy = atom_energy)
E_test = []
F_test = []
F_ref = []


for a in test_set:
    # res = calculateEFSparse(train_set,a,w,pars)
    # E_test.append(res/len(a))
    res = calculateEFSparse(train_set,a,w,pars)
    E_test.append(res["energy"]/len(a))
    # E_test.append(res/len(a))
    F_test.extend(res["forces"])
    F_ref.extend(a.arrays["forces"])

from buildStuff import make_water
print("createing water")
# water = make_water(1.0, [10, 10, 10])
water = make_water(1.0, [5, 5, 5])
res = calculateEFSparse(train_set,water,w,pars)
# res = calculateEFSparse(train_set,a,w,pars)

'''
print(E_ref)
print(E_test)
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