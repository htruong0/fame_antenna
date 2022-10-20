import numpy as np
import math
from itertools import combinations
from tqdm.auto import tqdm

def calculate_sij(p):
    '''Calculate all mandelstam invariant pairs.'''
    num_jets = p.shape[0]-2
    combs = list(combinations(range(1, num_jets+1), 2))
    keys = [f"{comb[0]}{comb[1]}" for comb in combs]
    sij = dict()
    for i, comb in enumerate(combs):
        pi = p[comb[0]+1]
        pj = p[comb[1]+1]
        sij[keys[i]] = 2*(pi[0]*pj[0]-pi[1]*pj[1]-pi[2]*pj[2]-pi[3]*pj[3])
    return sij

class Mapper():
    """
    Class to calculate mapping variables.
    """

    def __init__(self, permutations):
        self.permutations = permutations


    def calculate_all_map_variables(self, p, s):
        """
        Calculate map variables for all permutations and transform them.
        """
        self.set_momenta(p, s)
        N = len(self.permutations)
        rs = np.empty(N)
        rhos = np.empty(N)
        for i, perm in enumerate(self.permutations):
            r, rho = self.calculate_map_variables(*perm)
            rs[i] = r
            rhos[i] = rho
        map_variables = self.transform_map_variables(rs, rhos)
        return map_variables
    
    def calculate_map_variables(self, i, j, k):
        """
        Calculate r and rho from [arXiv:hep-ph/0212097] Eq. (5.4) and (5.7).
        """
        self.set_indices(i, j, k)
        r = self.s_jk / (self.s_ij + self.s_jk)
        rho = math.sqrt(1 + 4*r*(1-r)*self.s_ij*self.s_jk/(self.s_ijk*self.s_ik))
        return r, rho

    def transform_map_variables(self, rs, rhos, epsilon=1E-7):
        """
        Preprocess map variables to aid in training.
        """   
        rs = np.log(rs)
        rhos = np.log(rhos-(1-epsilon))
        return np.hstack([rs, rhos])
        
    def set_indices(self, i, j, k):
        """
        Set indices of antenna legs.
        """
        self.pi = self.p[i+1]
        self.pj = self.p[j+1]
        self.pk = self.p[k+1]
        ij = self._get_pair_string(i, j)
        ik = self._get_pair_string(i, k)
        jk = self._get_pair_string(j, k)
        self.s_ij = self.s[ij]
        self.s_ik = self.s[ik]
        self.s_jk = self.s[jk]
        self.s_ijk = self.s_ij + self.s_ik + self.s_jk
        
    def set_momenta(self, p, s):
        """
        Set momenta of choice.
        """
        self.p = p
        self.num_jets = self.p.shape[0]-2
        self.w = p[0][0] + p[1][0]
        # quicker to calculate all s_ij here as we can re-use them for all permutations
        self.s = s
        
    def _get_pair_string(self, i, j):
        return f"{i}{j}" if i < j else f"{j}{i}"


class AntennaGenerator():
    """
    Class to compute antenna functions.
    """
    
    def __init__(self, permutations, antenna):
        self.permutations = permutations
        self._antenna = antenna

    def calculate_Xs_ratio(self, inputs):
        '''Calculate antennae ratios.'''
        p, s, mu, alpha_s = inputs
        self._antenna.set_mu(mu)
        self._antenna.set_alpha_s(alpha_s)
        self._antenna.set_momenta(p, s)
        X = []
        for perm in self.permutations:
            X.append(self._antenna.XR(*perm))
        return np.array(X)   

        
    def calculate_Xs_loop(self, inputs):
        '''Calculate loop antennae.'''
        p, s, mu, alpha_s = inputs
        self._antenna.set_mu(mu)
        self._antenna.set_alpha_s(alpha_s)
        self._antenna.set_momenta(p, s)
        X = []
        for perm in self.permutations:
            X.append(self._antenna.X_3_1(*perm))
        return np.array(X)
        
    def calculate_Xs_tree(self, inputs):
        '''Calculate tree antennae.'''
        p, s, mu, alpha_s = inputs
        self._antenna.set_mu(mu)
        self._antenna.set_alpha_s(alpha_s)
        self._antenna.set_momenta(p, s)
        X = []
        for perm in self.permutations:
            X.append(self._antenna.X_3_0(*perm))
        return np.array(X)

    def get_alpha_s(self):
        return self._antenna.alpha_s

    def get_mu_r(self):
        return self._antenna.mu


class InputGenerator():
    """
    Class to compute antenna ratios,
    mapping variables and mandelstam variables.
    """
    def __init__(self, antennaGenerator, mapper):
        self._antenna = antennaGenerator
        self._mapper = mapper

    def calculate_inputs(self, p, mu_r=None, alpha_s=None, mode="tree"):
        sijs = calculate_sij(p)
        if mu_r is None:
            mu_r = self._antenna.get_mu_r()
            alpha_s = self._antenna.get_alpha_s()
            
        if mode == "tree":
            X = self._antenna.calculate_Xs_tree([p, sijs, mu_r, alpha_s])
        elif mode == "loop":
            X = self._antenna.calculate_Xs_loop([p, sijs, mu_r, alpha_s])
        elif mode == "ratio":
            X = self._antenna.calculate_Xs_ratio([p, sijs, mu_r, alpha_s])
        
        RF = self._mapper.calculate_all_map_variables(p, sijs)
        return X, RF, np.array(list(sijs.values()))
    
    def calculate_inputs_array(self, p_array, mu_r_array=None, alpha_s_array=None, mode="ratio", pbar=False):
        N = len(p_array)
        num_jets = p_array.shape[1]-2
        n_perms = len(self._antenna.permutations)
        n_s = {3: 3, 4: 6, 5: 10}
        Xs = np.zeros((N, n_perms))
        RFs = np.zeros((N, 2*n_perms))
        Sijs = np.zeros((N, n_s[num_jets]))
        
        if mu_r_array is None:
            mu_r_array = np.empty(N, dtype="object")
            alpha_s_array = np.empty(N, dtype="object")
        
        if pbar:
            for i, p in tqdm(enumerate(p_array), total=N):
                X, RF, S = self.calculate_inputs(p, mu_r_array[i], alpha_s_array[i], mode=mode)
                Xs[i] = X
                RFs[i] = RF
                Sijs[i] = S
        else:
            for i, p in enumerate(p_array):
                X, RF, S = self.calculate_inputs(p, mu_r_array[i], alpha_s_array[i], mode=mode)
                Xs[i] = X
                RFs[i] = RF
                Sijs[i] = S
        
        return Xs, RFs, Sijs
