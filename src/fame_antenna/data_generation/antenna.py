import math
import numpy as np
import sys
sys.path.append("/mt/home/htruong/Documents/fame/src/fame/data_generation/")
from ftdilog import dli2 # https://github.com/Expander/polylogarithm/blob/master/src/fortran/Li2.f90

frac = lambda x, y : x / y
pisqo6 = 1.6449340668482264
pisqo24 = 0.4112335167120566
twopi = 6.283185307179586

class Antenna():
    def __init__(self, mu=91.188, alpha_s=0.118, N_c=3):
        self.mu = mu
        self.alpha_s = alpha_s
        self.N_c = N_c
        self.quark_indices = [1, 2]
        self.gluon_indices = [3, 4, 5]


    def XR(self, i, j, k):
        self.set_indices(i, j, k)
        prefactor = self.N_c*self.alpha_s/twopi
        if (i in self.quark_indices and k in self.quark_indices):
            XR = prefactor*self.AR(i, j, k)
        elif (i in self.quark_indices and j in self.gluon_indices):
            XR = prefactor*self.DR(i, j, k)
        elif (i in self.gluon_indices and j in self.gluon_indices):
            XR = prefactor*self.FR(i, j, k)
        return XR


    def X_3_1(self, i, j, k):
        self.set_indices(i, j, k)
        prefactor = self.N_c**2*self.alpha_s/twopi
        if (i in self.quark_indices and k in self.quark_indices):
            X_3_1 = prefactor*self.A_3_1(i, j, k)
        elif (i in self.quark_indices and j in self.gluon_indices):
            X_3_1 = prefactor*self.D_3_1(i, j, k)
        elif (i in self.gluon_indices and j in self.gluon_indices):
            X_3_1 = prefactor*self.F_3_1(i, j, k)
        return X_3_1


    def X_3_0(self, i, j, k):
        self.set_indices(i, j, k)
        if (i in self.quark_indices and k in self.quark_indices):
            X_3_0 = self.N_c*self.A_3_0(i, j, k)
        elif (i in self.quark_indices and j in self.gluon_indices):
            X_3_0 = self.N_c*self.D_3_0(i, j, k)
        elif (i in self.gluon_indices and j in self.gluon_indices):
            X_3_0 = self.N_c*self.F_3_0(i, j, k)
        return X_3_0


    def FR(self, i, j, k):
        fr_ijk = self.fr(i, j, k)
        fr_jki = self.fr(j, k, i)
        fr_kij = self.fr(k, i, j)
        FR = (fr_ijk[0] + fr_jki[0] + fr_kij[0]) / (fr_ijk[1] + fr_jki[1] + fr_kij[1])
        return FR


    def F_3_1(self, i, j, k):
        F_3_1 = self.f_3_1(i, j, k) + self.f_3_1(j, k, i) + self.f_3_1(k, i, j)
        return F_3_1


    def all_f_3_1(self, i, j, k):
        return np.array([self.f_3_1(i, j, k), self.f_3_1(j, k, i), self.f_3_1(k, i, j)])


    def fr(self, i, j, k):
        f_3_0 = self.f_3_0(i, j, k)
        
        R_ij_ik = self.R(self.y_ij, self.y_ik)
        R_ik_jk = self.R(self.y_ik, self.y_jk)
        R_ij_jk = self.R(self.y_ij, self.y_jk)
        R = R_ij_ik + R_ik_jk + R_ij_jk
        
        Y = 11/6 * (math.log(self.y_ij) + math.log(self.y_ik) + math.log(self.y_jk))
        S = frac(1, 6*self.s_ij) + frac(1, 6*self.s_jk) + frac(1, 9*self.s_ijk)
        finite_from_pole = 2*(self.I_gg(self.s_ij)+self.I_gg(self.s_ik)+self.I_gg(self.s_jk)-2*self.I_gg(self.s_ijk))
        
        f_3_1 = -(Y+R)*f_3_0 + S + self.T*f_3_0 + finite_from_pole*f_3_0
        return f_3_1, f_3_0


    def f_3_1(self, i, j, k):
        f_3_0 = self.f_3_0(i, j, k)
        
        R_ij_ik = self.R(self.y_ij, self.y_ik)
        R_ik_jk = self.R(self.y_ik, self.y_jk)
        R_ij_jk = self.R(self.y_ij, self.y_jk)
        R = R_ij_ik + R_ik_jk + R_ij_jk
        
        Y = 11/6 * (math.log(self.y_ij) + math.log(self.y_ik) + math.log(self.y_jk))
        S = frac(1, 6*self.s_ij) + frac(1, 6*self.s_jk) + frac(1, 9*self.s_ijk)
        finite_from_pole = 2*(self.I_gg(self.s_ij)+self.I_gg(self.s_ik)+self.I_gg(self.s_jk)-2*self.I_gg(self.s_ijk))
        
        f_3_1 = -(Y+R)*f_3_0 + S + self.T*f_3_0 + finite_from_pole*f_3_0
        return f_3_1


    def F_3_0(self, i, j, k):
        F_3_0 = self.f_3_0(i, j, k) + self.f_3_0(j, k, i) + self.f_3_0(k, i, j)
        return F_3_0


    def all_f_3_0(self, i, j, k):
        return np.array([self.f_3_0(i, j, k), self.f_3_0(j, k, i), self.f_3_0(k, i, j)])


    def f_3_0(self, i, j, k):
        self.set_indices(i, j, k)
        
        a = 2*frac(self.s_ijk**2*self.s_ik, self.s_ij*self.s_jk)
        b = frac(self.s_ik*self.s_ij, self.s_jk)
        c = frac(self.s_ik*self.s_jk, self.s_ij)
        d = 8/3*self.s_ijk

        f_3_0 = frac(a+b+c+d, self.s_ijk**2)
        return f_3_0


    def DR(self, i, j, k):
        dr_ijk = self.dr(i, j, k)
        dr_ikj = self.dr(i, k, j)
        DR = (dr_ijk[0] + dr_ikj[0]) / (dr_ijk[1] + dr_ikj[1])
        return DR


    def D_3_1(self, i, j, k):
        D_3_1 = self.d_3_1(i, j, k) + self.d_3_1(i, k, j)        
        return D_3_1


    def all_d_3_1(self, i, j, k):
        return np.array([self.d_3_1(i, j, k), self.d_3_1(i, k, j)])


    def dr(self, i, j, k):
        d_3_0 = self.d_3_0(i, j, k)
            
        R_ij_jk = self.R(self.y_ij, self.y_jk)
        R_ik_jk = self.R(self.y_ik, self.y_jk)
        R_ij_ik = self.R(self.y_ij, self.y_ik)
        
        a = R_ij_jk + R_ik_jk + R_ij_ik
        b = 5/3*(math.log(self.y_ij) + math.log(self.y_ik)) + 11/6*math.log(self.y_jk)
        c = frac(1, 6*self.s_jk)
        finite_from_pole = 2*(self.I_qg(self.s_ij)+self.I_qg(self.s_ik)+self.I_gg(self.s_jk)-2*self.I_qg(self.s_ijk))

        d_3_1 = (-(a+b)*d_3_0 + c) + self.T*d_3_0 + finite_from_pole*d_3_0
        return d_3_1, d_3_0


    def d_3_1(self, i, j, k):
        d_3_0 = self.d_3_0(i, j, k)
            
        R_ij_jk = self.R(self.y_ij, self.y_jk)
        R_ik_jk = self.R(self.y_ik, self.y_jk)
        R_ij_ik = self.R(self.y_ij, self.y_ik)
        
        a = R_ij_jk + R_ik_jk + R_ij_ik
        b = 5/3*(math.log(self.y_ij) + math.log(self.y_ik)) + 11/6*math.log(self.y_jk)
        c = frac(1, 6*self.s_jk)
        finite_from_pole = 2*(self.I_qg(self.s_ij)+self.I_qg(self.s_ik)+self.I_gg(self.s_jk)-2*self.I_qg(self.s_ijk))

        d_3_1 = (-(a+b)*d_3_0 + c) + self.T*d_3_0 + finite_from_pole*d_3_0
        return d_3_1


    def D_3_0(self, i, j, k):
        D_3_0 = self.d_3_0(i, j, k) + self.d_3_0(i, k, j)
        return D_3_0


    def all_d_3_0(self, i, j, k):
        return np.array([self.d_3_0(i, j, k), self.d_3_0(i, k, j)])


    def d_3_0(self, i, j, k):
        self.set_indices(i, j, k)
        
        a = 2*frac(self.s_ijk**2*self.s_ik, self.s_ij*self.s_jk)
        b = frac(self.s_ij*self.s_ik, self.s_jk) + frac(self.s_ik*self.s_jk, self.s_ij)
        c = frac(self.s_jk**2, self.s_ij)
        d = frac(5*self.s_ijk, 2) + frac(self.s_jk, 2)

        d_3_0 = frac(a+b+c+d, self.s_ijk**2)
        return d_3_0


    def AR(self, i, j, k):
        ar_ijk = self.ar(i, j, k)
        ar_kji = self.ar(k, j, i)
        AR = (ar_ijk[0] + ar_kji[0]) / (ar_ijk[1] + ar_kji[1])
        return AR


    def A_3_1(self, i, j, k):
        A_3_1 = self.a_3_1(i, j, k) + self.a_3_1(k, j, i)
        return A_3_1


    def all_a_3_1(self, i, j, k):
        return np.array([self.a_3_1(i, j, k), self.a_3_1(k, j, i)])


    def ar(self, i, j, k):
        a_3_0 = self.a_3_0(i, j, k)
        
        R_ij_kj = self.R(self.y_ij, self.y_jk)
        a = R_ij_kj + 5/3*(math.log(self.y_ij)+math.log(self.y_jk))
        b = frac(1, 2*self.s_ijk) + frac(self.s_ik+self.s_jk, 2*self.s_ijk*self.s_ij) - frac(self.s_ij, 2*self.s_ijk*(self.s_ij+self.s_ik))
        c = frac(math.log(self.y_ij), self.s_ijk)*(2-frac(self.s_ij*self.s_jk, 2*(self.s_ik+self.s_jk)**2)+2*frac(self.s_ij-self.s_jk, self.s_ik+self.s_jk))
        finite_from_pole = 2*(self.I_qg(self.s_ij)+self.I_qg(self.s_jk)-self.I_qq(self.s_ijk))

        a_3_1 = -a*a_3_0+b+c + self.T*a_3_0 + finite_from_pole*a_3_0
        return a_3_1, a_3_0


    def a_3_1(self, i, j, k):
        a_3_0 = self.a_3_0(i, j, k)
        
        R_ij_kj = self.R(self.y_ij, self.y_jk)
        a = R_ij_kj + 5/3*(math.log(self.y_ij)+math.log(self.y_jk))
        b = frac(1, 2*self.s_ijk) + frac(self.s_ik+self.s_jk, 2*self.s_ijk*self.s_ij) - frac(self.s_ij, 2*self.s_ijk*(self.s_ij+self.s_ik))
        c = frac(math.log(self.y_ij), self.s_ijk)*(2-frac(self.s_ij*self.s_jk, 2*(self.s_ik+self.s_jk)**2)+2*frac(self.s_ij-self.s_jk, self.s_ik+self.s_jk))
        finite_from_pole = 2*(self.I_qg(self.s_ij)+self.I_qg(self.s_jk)-self.I_qq(self.s_ijk))

        a_3_1 = -a*a_3_0+b+c + self.T*a_3_0 + finite_from_pole*a_3_0
        return a_3_1


    def A_3_0(self, i, j, k):
        A_3_0 = self.a_3_0(i, j, k) + self.a_3_0(k, j, i)
        return A_3_0


    def all_a_3_0(self, i, j, k):
        return np.array([self.a_3_0(i, j, k), self.a_3_0(k, j, i)])


    def a_3_0(self, i, j, k):
        self.set_indices(i, j, k)

        a = frac(self.s_jk, self.s_ij)
        b = frac(self.s_ik*self.s_ijk, self.s_ij*self.s_jk)

        a_3_0 = frac(a + b, self.s_ijk)
        return a_3_0


    def I_gg(self, sij):
        return self.i_gg(sij, self.mu**2)


    def I_qq(self, sij):
        return self.i_qq(sij, self.mu**2)


    def I_qg(self, sij):
        return self.i_qg(sij, self.mu**2)


    def i_gg(self, sij, mu2, ieorder=0):
        z = self.zlog(mu2) - self.zlog(sij)
        
        e2 = -1/2
        e1 = -11/12
        e0 = pisqo24
        
        I_gg = self.I_op(z, e0, e1, e2, ieorder)
        return I_gg


    def i_qq(self, sij, mu2, ieorder=0):
        z = self.zlog(mu2) - self.zlog(sij)
        
        e2 = -1/2
        e1 = -3/4
        e0 = pisqo24
        
        I_qq = self.I_op(z, e0, e1, e2, ieorder)
        return I_qq


    def i_qg(self, sij, mu2, ieorder=0):
        z = self.zlog(mu2) - self.zlog(sij)
        
        e2 = -1/2
        e1 = -5/6
        e0 = pisqo24
        
        I_qg = self.I_op(z, e0, e1, e2, ieorder)
        return I_qg


    def I_op(self, z, e0, e1, e2, ieorder):
        if ieorder == -2:
            I = e2
        elif ieorder == -1:
            I = e1 + z*e2
        elif ieorder == 0:
            # minus pi^2 is a performance trick to avoid using complex numbers in zlog
            z2 = z**2 - np.pi**2
            I = e0 + z*e1 + 0.5*z2*e2
        return I


    def zlog(self, z):
        return math.log(z)


    def R(self, y, z):
        a = math.log(y)*math.log(z)
        b = math.log(y)*math.log(1-y)
        c = math.log(z)*math.log(1-z)
        d = pisqo6
        e = dli2(y) + dli2(z)
        return a-b-c+d-e


    def kosower_mapping(self, i, j, k):
        self.set_indices(i, j, k)

        r = self.s_jk / (self.s_ij + self.s_jk)
        rho = math.sqrt(1 + 4*r*(1-r)*self.s_ij*self.s_jk/(self.s_ijk*self.s_ik))
        
        x = ((1+rho)*self.s_ijk - 2*r*self.s_jk) / (2*(self.s_ij + self.s_ik))
        z = ((1-rho)*self.s_ijk - 2*r*self.s_ij) / (2*(self.s_jk + self.s_ik))

        q_ij = x*self.pi + r*self.pj + z*self.pk
        q_jk = (1-x)*self.pi + (1-r)*self.pj + (1-z)*self.pk

        # delete p_j
        q = np.delete(self.p, j+1, axis=0)
        # insert p_ij
        if i >= j+1:
            q[i] = q_ij
        else:
            q[i+1] = q_ij
        # insert p_jk
        if k > j:
            q[k] = q_jk
        else:
            q[k+1] = q_jk
        return q


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
        self.y_ij = self._y(self.s_ij)
        self.y_ik = self._y(self.s_ik)
        self.y_jk = self._y(self.s_jk)
        self.T = 11/6*math.log(self.mu**2/self.s_ijk)


    def set_mu(self, mu):
        self.mu = mu


    def set_alpha_s(self, alpha_s):
        self.alpha_s = alpha_s


    def set_momenta(self, p, s):
        """
        Set momenta of choice.
        """
        self.p = p
        self.num_jets = self.p.shape[0]-2
        self.w = p[0][0] + p[1][0]
        # quicker to calculate all s_ij here as we can re-use them for all permutations
        self.s = s


    def _y(self, s):
        return s/self.s_ijk


    def _get_pair_string(self, i, j):
        return f"{i}{j}" if i < j else f"{j}{i}"
