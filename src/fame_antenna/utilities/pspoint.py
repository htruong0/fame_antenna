from itertools import combinations
import tensorflow as tf
from collections import OrderedDict
from fame_antenna.utilities import utility_functions

def idot(p1, p2):
    return p1[0]*p2[0] - p1[1]*p2[1] - p1[2]*p2[2] - p1[3]*p2[3]

class PSpoint():
    '''A phase-space point class which can be used to calculate all Mandelstam invariants.'''

    def __init__(self, four_moms, y_global_cut=0.0001):
        self.four_moms = four_moms
        self.E = four_moms[:, 0]
        self.px = four_moms[:, 1]
        self.py = four_moms[:, 2]
        self.pz = four_moms[:, 3]
        self.w = self.E[0] + self.E[1]
        self.num_jets = four_moms.shape[0]-2
        self.combs = list(combinations(range(1, self.num_jets+1), 2))
        self.keys = [''.join(str(comb[0]) + str(comb[1])) for comb in self.combs]
        self.sij = self.calculate_sij()
        self.minsij = tf.reduce_min(list(self.sij.values()))
        tf.debugging.Assert(tf.less(y_global_cut*self.w**2, self.minsij), ["min sij below cut", self.minsij])
        self.minsij, self.minkey = min([(s[1], s[0]) for s in list(self.sij.items())])

        
    def __eq__(self, other):
        if type(self) == type(other):
            return self.four_moms == other.four_moms
        else:
            return False
        
    def __repr__(self):
        return "min sij = {:.1f} (s{})".format(self.minsij, self.minkey)
        
    def __getitem__(self, key):
        return self.four_moms[key]
    
    def __setitem__(self, key, item):
        self.four_moms[key] = item
    
    def __len__(self):
        return len(self.four_moms)
    
    def __neg__(self):
        negative = PSpoint(self.four_moms)
        negative.four_moms = -self.four_moms
        return negative
    
    def __add__(self, other):
        if len(self.four_moms) == len(other.four_moms):
            return self.four_moms + other.four_moms
        else:
            raise utility_functions.myException("Momenta don't have the same shape.")
            
    def __sub__(self, other):
        if len(self.four_moms) == len(other.four_moms):
            return self.four_moms - other.four_moms
        else:
            raise utility_functions.myException("Momenta don't have the same shape.")
            
    def __mul__(self, other):
        if len(self.four_moms) == len(other.four_moms):
            return self.four_moms * other.four_moms
        else:
            raise utility_functions.myException("Momenta don't have the same shape.")
            
    def __div__(self, other):
        if len(self.four_moms) == len(other.four_moms):
            return self.four_moms / other.four_moms
        else:
            raise utility_functions.myException("Momenta don't have the same shape.")
            
    def __truediv__(self, other):
        if len(self.four_moms) == len(other.four_moms):
            return self.four_moms / other.four_moms
        else:
            raise utility_functions.myException("Momenta don't have the same shape.")

    def calculate_sij(self):
        sij = OrderedDict()
        for i, comb in enumerate(self.combs):
            pi = self.four_moms[comb[0]+1]
            pj = self.four_moms[comb[1]+1]
            sij[self.keys[i]] = 2*idot(pi, pj)
        return sij
