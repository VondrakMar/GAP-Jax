from GAP import *
import matplotlib.pyplot as plt

import jax.numpy as np
import random
random.seed(10)


mols = read("LiH.xyz@:",format="extxyz")
random.shuffle(mols)
print(len(mols))
train_set = mols[:100]
test_set = mols[:100]
pars = {"periodic":False,
        "rcut":3.5,
        "nmax":4,
        "sigma":3.5/8,
        "species":['Li','H'],
        "lmax":3}
lambda_reg = 0.1
K = kernel_matrix(pars,mol_set1 = train_set, mol_set2 = train_set, zeta=2)
w = np.ones(K.shape[0])*0.3
E_ref = get_energies(train_set,atom_energy)
M = prepLgap(K.shape[0],len(train_set),train_set)
Loss = GAPLoss(w,K,M,E_ref,lambda_reg)
print("loss ", Loss)
w = train(M,E_ref,K, lambda_reg =lambda_reg, atom_energy = atom_energy, energy_keyword = "energy")
print(w)
LossMin = GAPLoss(w,K,M,E_ref,lambda_reg)
print("loss ", LossMin)
print("starting")





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
for a in test_set:
    mol_energy = (calculate(train_set,a,w,pars))
    # for element in atom_energy:
        # no_of_A = np.count_nonzero(a.symbols == element)
        # mol_energy += (no_of_A*atom_energy[element])
    # print(mol_energy,a.info["energy"])
    E_test.append(mol_energy/len(a))


plt.figure(figsize=(10,8), dpi=100)
plt.plot(E_ref,E_ref,alpha = 0.5,color="black")
plt.scatter(E_test,E_ref,alpha = 0.5)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()


'''
print("loss ", Loss)

Loss = GAPLoss(w-0.001,K,M,E_ref,0.1)
print("loss ", Loss)
Loss = GAPLoss(w+0.001,K,M,E_ref,0.1)
print("loss ", Loss)


for a in test_set:
    mol_energy = (calculate(train_set,a,w,pars))
    for element in atom_energy:
        no_of_A = np.count_nonzero(a.symbols == element)
        mol_energy += (no_of_A*atom_energy[element])
    print(mol_energy,a.info["energy"])
    E.append(mol_energy)


'''
def minimize(w,K,M,E_ref,reg = 0.1,learning_rate=0.001,momentum=0.4):
    dloss = grad(GAPLoss)
    # dcons = grad(Q_constraint)
    # oldstep = np.zeros(len(w))
    for istep in range(5):
        oldw = w
        #print(istep,'loss',loss(q,xyz,eneg,hard,atsize,Qtot))
        #print('dloss',    dloss(q,xyz,eneg,hard,atsize,Qtot))
        loss_grad = dloss(w,K,M,E_ref,reg)
        proj_grad = loss_grad 
        print(w)
        print("proj_grad",proj_grad)
        w -= learning_rate*proj_grad 
        # oldstep = newstep
        # w = w + step
        if np.linalg.norm(w-oldw)<1e-4:
            return w
        # print(istep,'w',w)
    return 'not converged'
# w = np.ones(K.shape[0])*0.3
w = minimize(w,K,M,E_ref,reg =0.1)
L = GAPLoss(w,K,M,E_ref,0.1)
E = []
for a in test_set:
    mol_energy = (calculate(train_set,a,w,pars))
    for element in atom_energy:
        no_of_A = np.count_nonzero(a.symbols == element)
        mol_energy += (no_of_A*atom_energy[element])
    print(mol_energy,a.info["energy"])
    E.append(mol_energy)