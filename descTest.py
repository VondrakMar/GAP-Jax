from GAP import *
import matplotlib.pyplot as plt

import numpy as np
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


desc, der = DerDescriptor(mols[0],pars)
descSparse, derSparse = DerDescriptorSparse(mols[0],pars)