#!/usr/bin/env python3
import sys, os, itertools
import numpy as np
from sympy.physics.quantum.spin import Rotation
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j, clebsch_gordan

if(__package__==None or __package__==""):
    import LECs
    import constants
else:
    from . import constants
    from . import LECs

class ChEFTNNPotential:
    def __init__(self):
        self.NMesh = 60
        self.costh = None
        self.weights = None
        self.Jmax = -1
        self.wigner_d = None
        self.helicity_to_pw = None
        self.Regulator = "NonLocal"
        self.Power = 4
        self.Lambda =394
        self.Lambda_SFR = 700.0
        self.g_A = 1.29
        self.include_delta = True
        self.n_mesh_loop_integral = 25
        self.iso = [-3,1] # T=0,1

        # parameters
        self.ch_order = 0
        self.Ct_1S0_pp = 0
        self.Ct_1S0_pn = 0
        self.Ct_1S0_nn = 0
        self.Ct_3S1 = 0
        self.C_1S0 = 0
        self.C_3P0 = 0
        self.C_1P1 = 0
        self.C_3P1 = 0
        self.C_3S1 = 0
        self.C_3S1_3D1 = 0
        self.C_3P2 = 0
        self.c1 = 0
        self.c2 = 0
        self.c3 = 0
        self.c4 = 0

    def set_pot_parameters(self, params):
        if("NMesh" in params): self.NMesh = params["NMesh"]
        if("Jmax" in params): self.Jmax = params["Jmax"]
        if("Regulator" in params): self.Regulator = params["Regulator"]
        if("Power" in params): self.Power = params["Power"]
        if("Lambda" in params): self.Lambda = params["Lambda"]
        if("ch_order" in params): self.ch_order = params["ch_order"]
        if("include_delta" in params): self.include_delta = params["include_delta"]
        if("Lambda_SFR" in params): self.Lambda_SFR = params["Lambda_SFR"]

        # LO LECs
        if("Ct_1S0_pp" in params): self.Ct_1S0_pp = params["Ct_1S0_pp"] * 1.e-2 # MeV-2
        if("Ct_1S0_pn" in params): self.Ct_1S0_pn = params["Ct_1S0_pn"] * 1.e-2 # MeV-2
        if("Ct_1S0_nn" in params): self.Ct_1S0_nn = params["Ct_1S0_nn"] * 1.e-2 # MeV-2
        if("Ct_3S1" in params): self.Ct_3S1 = params["Ct_3S1"] * 1.e-2 # MeV-2
        # NLO LECs
        if("C_1S0" in params): self.C_1S0 = params["C_1S0"] * 1.e-8 # MeV-4
        if("C_3P0" in params): self.C_3P0 = params["C_3P0"] * 1.e-8 # MeV-4
        if("C_1P1" in params): self.C_1P1 = params["C_1P1"] * 1.e-8 # MeV-4
        if("C_3P1" in params): self.C_3P1 = params["C_3P1"] * 1.e-8 # MeV-4
        if("C_3S1" in params): self.C_3S1 = params["C_3S1"] * 1.e-8 # MeV-4
        if("C_3S1_3D1" in params): self.C_3S1_3D1 = params["C_3S1_3D1"] * 1.e-8 # MeV-4
        if("C_3P2" in params): self.C_3P2 = params["C_3P2"] * 1.e-8 # MeV-4
        # NNLO piN LECs
        if("c1" in params): self.c1 = params["c1"] * 1.e-3 # MeV-1
        if("c2" in params): self.c2 = params["c2"] * 1.e-3 # MeV-1
        if("c3" in params): self.c3 = params["c3"] * 1.e-3 # MeV-1
        if("c4" in params): self.c4 = params["c4"] * 1.e-3 # MeV-1

        self.costh, self.weights = np.polynomial.legendre.leggauss(self.NMesh)
        self.legengre = np.zeros((self.Jmax+1, self.NMesh), dtype=np.float64)
        self.zJ = np.zeros((self.Jmax+1, self.NMesh), dtype=np.float64)
        for j in range(self.Jmax+1):
            self.legengre[j] = np.polynomial.legendre.legval(self.costh, [0]*j + [1])
            self.zJ[j,:] = self.costh**j

    # partial-wave decomposition for central force
    def do_pwd_C(self, v, pbra, pket, s, j, coupled):
        """
        Eq. (4.20) from K. Erkelenz et al., Nucl. Phys. A 176, 413 (1971).
        """
        mes = np.zeros(6, dtype=np.float64)
        if(s==0 and not coupled): mes[0] += 2.0 * self.pwd_integral(v, 0, j)
        elif(s==1 and not coupled): mes[1] += 2.0 * self.pwd_integral(v, 0, j)
        elif(coupled):
            mes[2] += 2.0 * self.pwd_integral(v, 0, j+1)
            if(j==0): return mes
            mes[3] += 2.0 * self.pwd_integral(v, 0, j-1)
        return mes

    # partial-wave decomposition for spin-spin force
    def do_pwd_S(self, v, pbra, pket, s, j, coupled):
        """
        Eq. (4.21) from K. Erkelenz et al., Nucl. Phys. A 176, 413 (1971).
        """
        mes = np.zeros(6, dtype=np.float64)
        if(s==0 and not coupled): mes[0] -= 6.0 * self.pwd_integral(v, 0, j)
        elif(s==1 and not coupled): mes[1] += 2.0 * self.pwd_integral(v, 0, j)
        elif(coupled):
            mes[2] += 2.0 * self.pwd_integral(v, 0, j+1)
            if(j==0): return mes
            mes[3] += 2.0 * self.pwd_integral(v, 0, j-1)
        return mes

    # partial-wave decomposition for spin-orbit force
    def do_pwd_LS(self, v, pbra, pket, s, j, coupled):
        """
        Eq. (4.22) from K. Erkelenz et al., Nucl. Phys. A 176, 413 (1971).
        """
        mes = np.zeros(6, dtype=np.float64)
        if(s==0 and not coupled): mes[0] += 0.0
        elif(s==1 and not coupled): mes[1] += 2.0 * pbra * pket * (self.pwd_integral(v, 0, j+1) - self.pwd_integral(v, 0, j-1)) / (2.0*j + 1.0)
        elif(coupled):
            mes[2] += 2.0 * pbra * pket * (j+2)*(self.pwd_integral(v, 0, j+2) - self.pwd_integral(v, 0, j) ) / (2.0*j + 3.0)
            if(j==0): return mes
            mes[3] += 2.0 * pbra * pket * (j-1)*(self.pwd_integral(v, 0, j-2) - self.pwd_integral(v, 0, j) ) / (2.0*j - 1.0)
        return mes

    def do_pwd_sigL(self, v, pbra, pket, s, j, coupled):
        """
        Eq. (4.23) from K. Erkelenz et al., Nucl. Phys. A 176, 413 (1971).
        """
        mes = np.zeros(6, dtype=np.float64)
        if(s==0 and not coupled): mes[0] += 2.0 * pbra**2 * pket**2 * (self.pwd_integral(v, 2, j) - self.pwd_integral(v, 0, j))
        elif(s==1 and not coupled): mes[1] += 2.0 * pbra**2 * pket**2 * (-1.0*self.pwd_integral(v, 0, j) + (j-1)/(2*j+1)*self.pwd_integral(v, 1, j+1) + (j+2)/(2*j+1)*self.pwd_integral(v, 1, j-1))
        elif(coupled):
            mes[2] += 2.0 * pbra**2 * pket**2 * ( (2*j+3)/(2*j+1)*self.pwd_integral(v, 0, j+1) -2/(2*j+1)*self.pwd_integral(v, 1, j) - self.pwd_integral(v, 2, j+1))
            if(j==0): return mes
            mes[3] += 2.0 * pbra**2 * pket**2 * ( (2*j-1)/(2*j+1)*self.pwd_integral(v, 0, j-1) +2/(2*j+1)*self.pwd_integral(v, 1, j) - self.pwd_integral(v, 2, j-1))
            mes[4] -= 4.0 * pbra**2 * pket**2 * np.sqrt(j*(j+1)) / (2*j+1)**2 * (self.pwd_integral(v, 0, j+1) - self.pwd_integral(v, 0, j-1))
            mes[5] -= 4.0 * pbra**2 * pket**2 * np.sqrt(j*(j+1)) / (2*j+1)**2 * (self.pwd_integral(v, 0, j+1) - self.pwd_integral(v, 0, j-1))
        return mes

    def do_pwd_T(self, v, pbra, pket, s, j, coupled):
        """
        Eq. (4.24) from K. Erkelenz et al., Nucl. Phys. A 176, 413 (1971).
        """
        mes = np.zeros(6, dtype=np.float64)
        if(s==0 and not coupled): mes[0] += 2.0 * (-(pbra**2 + pket**2) * self.pwd_integral(v, 0, j) + 2.0*pbra*pket*self.pwd_integral(v, 1, j))
        elif(s==1 and not coupled): mes[1] += 2.0 * ((pbra**2 + pket**2) * self.pwd_integral(v, 0, j) - 2.0*pbra*pket*(j*self.pwd_integral(v,0,j+1) + (j+1)*self.pwd_integral(v,0,j-1))/(2*j+1))
        elif(coupled):
            mes[2] += 2.0 /(2*j+1) * (-(pbra**2+pket**2) * self.pwd_integral(v,0,j+1) + 2.0*pbra*pket*self.pwd_integral(v,0,j))
            if(j==0): return mes
            mes[3] += 2.0 /(2*j+1) * ((pbra**2+pket**2) * self.pwd_integral(v,0,j-1) - 2.0*pbra*pket*self.pwd_integral(v,0,j))
            mes[4] -= 4.0 * np.sqrt(j*(j+1)) / (2*j+1) * (pbra**2 * self.pwd_integral(v,0,j-1) + pket**2 * self.pwd_integral(v,0,j+1) - 2.0*pbra*pket*self.pwd_integral(v,0,j))
            mes[5] -= 4.0 * np.sqrt(j*(j+1)) / (2*j+1) * (pbra**2 * self.pwd_integral(v,0,j+1) + pket**2 * self.pwd_integral(v,0,j-1) - 2.0*pbra*pket*self.pwd_integral(v,0,j))
        return mes

    def do_pwd_sigk(self, v, pbra, pket, s, j, coupled):
        """
        Eq. (4.25) from K. Erkelenz et al., Nucl. Phys. A 176, 413 (1971).
        """
        return self.do_pwd_T(v, -pbra, pket, s, j, coupled) * 0.25

    def pwd_integral(self, v, l, j):
        """
        The equation above Eq. (4.14) from K. Erkelenz et al., Nucl. Phys. A 176, 413 (1971).
        """
        if(j<0): return 0.0
        integrand_coeff = self.weights * self.zJ[l] * self.legengre[j]
        return np.dot(integrand_coeff, v) * np.pi

    def multiply_regulator(self, v, pbra, pket):
        freg = np.exp(-(pbra**2/self.Lambda**2)**self.Power) * np.exp(-(pket**2/self.Lambda**2)**self.Power)
        return v * freg

    def LO_contacts(self, pbra, pket, s, j, tz, coupled):
        """
        Eq. (4.39) from R. Machleidt, Phys. Rep. 503, 1 (2011).
        """
        mes = np.zeros(6, dtype=np.float64)
        if(s==0 and j==0 and not coupled):
            if(tz==-1): mes[0] += self.Ct_1S0_pp
            if(tz== 0): mes[0] += self.Ct_1S0_pn
            if(tz== 1): mes[0] += self.Ct_1S0_nn
        elif(s==1 and j==1 and coupled):
            mes[3] += self.Ct_3S1
        return mes

    def NLO_contacts(self, pbra, pket, s, j, coupled):
        """
        Eq. (4.41) from R. Machleidt, Phys. Rep. 503, 1 (2011).
        """
        mes = np.zeros(6, dtype=np.float64)
        if(s==0 and j==0 and not coupled):
            # 1S0
            mes[0] += (pbra**2 + pket**2) * self.C_1S0
        elif(s==1 and j==0 and coupled):
            # 3P0
            mes[2] += pbra*pket * self.C_3P0
        elif(s==0 and j==1 and not coupled):
            # 1P1
            mes[0] += pbra*pket * self.C_1P1
        elif(s==1 and j==1 and not coupled):
            # 3P1
            mes[1] += pbra*pket * self.C_3P1
        elif(s==1 and j==1 and coupled):
            # 3S1-3D1
            mes[3] += (pbra**2 + pket**2) * self.C_3S1
            mes[4] += pbra**2 * self.C_3S1_3D1
            mes[5] += pket**2 * self.C_3S1_3D1
        elif(s==1 and j==2 and coupled):
            # 3P2-3F2
            mes[3] += pbra*pket * self.C_3P2
        return mes

    def OPE_potential(self, pbra, pket, s, j, t, tz, coupled):
        """
        Eq. (4.5) from R. Machleidt, Phys. Rep. 503, 1 (2011).
        for pn, the CIB effect is included as introduced in Eq. (4.28) of the same reference.
        """
        if(tz==0):
            WT = 1.0 * (self.g_A**2) / (4.0*constants.f_pi**2) * self.ope_propagator(pbra, pket, constants.m_pi0) \
                  + 2.0 * (-1)**t * (self.g_A**2) / (4.0*constants.f_pi**2) * self.ope_propagator(pbra, pket, constants.m_pic)
        elif(abs(tz)==1):
            WT = -(self.g_A**2) / (4.0*constants.f_pi**2) * self.ope_propagator(pbra, pket, constants.m_pi0)
        mes = self.do_pwd_T(WT, pbra, pket, s, j, coupled)
        return mes

    def ope_propagator(self, pbra, pket, meson_mass):
        return 1.0 / ( self.q2 + meson_mass**2)

    def TPE_potential(self, pbra, pket, s, j, t, coupled):
        self.set_loop_function_W()
        self.set_loop_function_Wtilde()
        self.set_loop_function_S()
        self.set_loop_function_L()
        self.set_loop_function_A()
        self.set_loop_function_H()
        self.set_loop_function_D()
        V_C, W_C, V_S, W_S, V_LS, W_LS, V_T, W_T = np.zeros((8, self.NMesh), dtype=np.float64)

        if(self.ch_order >= 1):
            # from Eqs. (4.9) - (4.11) of R. Machleidt, Phys. Rep. 503, 1 (2011)
            W_C = -self.l / (384.0 * np.pi**2 * constants.f_pi**4) * \
                (4.0*constants.m_pi**2*(5.0*self.g_A**4-4.0*self.g_A**2-1.0) + \
                  self.q2*(23.0*self.g_A**4-10.0*self.g_A**2-1.0) + \
                      48.0*self.g_A**4*constants.m_pi**4/(self.w**2))
            V_T = -3.0 * self.g_A**4 * self.l / (64.0 * np.pi**2 * constants.f_pi**4)
            V_S = self.q2 * 3.0 * self.g_A**4 * self.l / (64.0 * np.pi**2 * constants.f_pi**4)

        if(self.include_delta and self.ch_order >= 1):
            # from Eqs. (2.5) - (2.7) of H. Krebs et al, Eur. Phys. J. A 32, 127 (2007)
            delta = constants.m_delta - constants.m_nucl
            delta2 = delta*delta
            sigma = 2.0 * constants.m_pi**2 + self.q2 - 2.0 * delta2
             # delta single triangle diagrams:
            W_C += (- constants.hA**2) / (216.0 * np.pi**2 * constants.f_pi**4) * \
                ((6.0 * sigma - self.w**2)*self.l + 12.0*delta2*sigma*self.d)

            # delta leading single box diagrams:
            V_C += - self.g_A**2 * constants.hA**2 / (12.0*np.pi*constants.f_pi**4*delta) * \
                (2.0*constants.m_pi**2 + self.q2) * (2.0*constants.m_pi**2 + self.q2) * self.a
            W_C += - self.g_A**2 * constants.hA**2 / (216.0*np.pi**2*constants.f_pi**4) * \
                ((12.0*delta2-20.0*constants.m_pi**2-11.0*self.q2)*self.l + 6.0*sigma**2*self.d)
            V_T += - self.g_A**2 * constants.hA**2 / (48.0*np.pi**2*constants.f_pi**4) * \
                (-2.0*self.l + (self.w**2-4.0*delta2)*self.d)
            V_S += self.q2 * self.g_A**2 * constants.hA**2 / (48.0*np.pi**2*constants.f_pi**4) * \
                (-2.0*self.l + (self.w**2-4.0*delta2)*self.d)
            W_T += - self.g_A**2 * constants.hA**2 / (144.0*np.pi*constants.f_pi**4*delta) * self.w**2*self.a
            W_S += self.q2 * self.g_A**2 * constants.hA**2 / (144.0*np.pi*constants.f_pi**4*delta) * self.w**2*self.a

            # delta leading double box diagrams:
            V_C += - constants.hA**4 / (27.0*np.pi**2 * constants.f_pi**4) * \
                (-4.0*delta2*self.l + sigma*(self.h + (sigma+8.0*delta2)*self.d))
            W_C += - constants.hA**4 / (486.0*np.pi**2 * constants.f_pi**4) * \
                ((12.0*sigma-self.w**2)*self.l + 3.0*sigma*(self.h+(8.0*delta2-sigma)*self.d))
            V_T += - constants.hA**4 / (216.0*np.pi**2 * constants.f_pi**4) * \
                (6.0*self.l + (12.0*delta2-self.w**2)*self.d)
            V_S += self.q2 * constants.hA**4 / (216.0*np.pi**2 * constants.f_pi**4) * \
                (6.0*self.l + (12.0*delta2-self.w**2)*self.d)
            W_T += - constants.hA**4 / (1296.0*np.pi**2 * constants.f_pi**4) * \
                (2.0*self.l + (4.0*delta2+self.w**2)*self.d)
            W_S += self.q2 * constants.hA**4 / (1296.0*np.pi**2 * constants.f_pi**4) * \
                (2.0*self.l + (4.0*delta2+self.w**2)*self.d)

        if(self.ch_order >= 2):
            # LEC dependent terms
            # from Eqs. (4.13) - (4.18) of R. Machleidt, Phys. Rep. 503, 1 (2011)
            V_C += 3.0 * self.g_A**2 / (16.0 * np.pi * constants.f_pi**4) * \
                (- (2.0 * constants.m_pi**2 * (2.0*self.c1 - self.c3) - self.q2*self.c3) * self.w_tilde * self.w_tilde * self.a)
            W_T += -self.g_A**2 * self.a / (32.0 * np.pi * constants.f_pi**4) * (self.c4 * self.w * self.w)
            W_S += -self.q2 * (-self.g_A**2 * self.a / (32.0 * np.pi * constants.f_pi**4) * (self.c4 * self.w * self.w))
            if(not self.include_delta):
                # relativistic correction (1/m dependent)
                # from Eqs. (4.13) - (4.18) of R. Machleidt, Phys. Rep. 503, 1 (2011)
                V_C += 3.0 * self.g_A**2 / (16.0 * np.pi * constants.f_pi**4) \
                    * (self.g_A**2 * constants.m_pi**5 / (16.0 * constants.m_nucl * self.w**2) + 3.0*self.q2*self.g_A**2 / (16.0*constants.m_nucl)*self.w_tilde**2 *self.a)
                W_C += self.g_A**2 / (128.0 * np.pi * constants.m_nucl * constants.f_pi**4) \
                    * (3.0*self.g_A**2*constants.m_pi**5/(self.w*self.w) - (4.0*constants.m_pi**2 + 2.0*self.q2 - self.g_A**2*(4.0*constants.m_pi**2+3.0*self.q2)) * self.w_tilde*self.w_tilde*self.a)
                V_T += 9.0*self.g_A**4*self.w_tilde*self.w_tilde*self.a / (512.0*np.pi*constants.m_nucl*constants.f_pi**4)
                W_T += -self.g_A**2*self.a / (32.0 * np.pi * constants.f_pi**4) * (self.w*self.w/(4.0*constants.m_nucl) - self.g_A**2/(8.0*constants.m_nucl) * (10.0*constants.m_pi**2 + 3.0*self.q2))
                V_S += -self.q2 * 9.0*self.g_A**4*self.w_tilde*self.w_tilde*self.a / (512.0*np.pi*constants.m_nucl*constants.f_pi**4)
                W_S += self.q2*self.g_A**2*self.a / (32.0 * np.pi * constants.f_pi**4) * (self.w*self.w/(4.0*constants.m_nucl) - self.g_A**2/(8.0*constants.m_nucl) * (10.0*constants.m_pi**2 + 3.0*self.q2))
                V_LS += 3.0*self.g_A**4*self.w_tilde*self.w_tilde*self.a / (32.0*np.pi*constants.m_nucl*constants.f_pi**4)
                W_LS += self.g_A**2*(1.0 - self.g_A**2)*self.w*self.w*self.a / (32.0*np.pi*constants.m_nucl*constants.f_pi**4)

                # further corrections from iterative diagrams
                # from Eqs. (4.21) - (4.24) of R. Machleidt, Phys. Rep. 503, 1 (2011)
                V_C += (-3) * self.g_A**4 / (256.0 * np.pi * constants.f_pi**4 * constants.m_nucl) \
                    * (constants.m_pi*self.w**2 + self.w_tilde**4 * self.a)
                W_C += self.g_A**4 / (128.0 * np.pi * constants.f_pi**4 * constants.m_nucl) \
                    * (constants.m_pi*self.w**2 + self.w_tilde**4 * self.a)
                tmp_V_T = 3 * self.g_A**4 / (512.0 * np.pi * constants.f_pi**4 * constants.m_nucl) \
                    * (constants.m_pi + self.w**2 * self.a)
                tmp_W_T = (-1) * self.g_A**4 / (256.0 * np.pi * constants.f_pi**4 * constants.m_nucl) \
                    * (constants.m_pi + self.w**2 * self.a)
                V_T += tmp_V_T
                W_T += tmp_W_T
                V_S += -self.q2 * tmp_V_T 
                W_S += -self.q2 * tmp_W_T

        if(self.include_delta and self.ch_order >= 2):
             # delta subleading triangle diagrams
            V_C += - constants.hA**2 * delta / (18.0*np.pi**2 * constants.f_pi**4) * \
                (6.0*sigma*(4.0*self.c1*constants.m_pi**2-2.0*self.c2*delta**2-self.c3*(2*delta**2+sigma))*self.d + \
                (-24.0*self.c1*constants.m_pi**2+self.c2*(self.w**2-6.0*sigma)+6.0*self.c3*(2.0*delta2+sigma))*self.l)
            W_T += - self.c4 * constants.hA**2 * delta / (72.0*np.pi**2 * constants.f_pi**4) * \
                ((self.w**2-4.0*delta2)*self.d - 2.0*self.l)
            W_S += self.q2 * self.c4*constants.hA**2*delta / (72.0*np.pi**2 * constants.f_pi**4) * \
                ((self.w*self.w-4.0*delta**2)*self.d - 2.0*self.l)
            # other terms are proportinal to b3+b8, which can be renormalized.
        mes = self.do_pwd_C( V_C +self.iso[t]*W_C,  pbra, pket, s, j, coupled)
        mes+= self.do_pwd_S( V_S +self.iso[t]*W_S,  pbra, pket, s, j, coupled)
        mes+= self.do_pwd_T( V_T +self.iso[t]*W_T,  pbra, pket, s, j, coupled)
        mes+= self.do_pwd_LS(V_LS+self.iso[t]*W_LS, pbra, pket, s, j, coupled)
        return mes

    def calc_loop_function_W(self, q):
        return np.sqrt(q**2 + 4.0 * constants.m_pi**2)

    def set_loop_function_W(self):
        self.w = np.sqrt(self.q**2 + 4.0 * constants.m_pi**2)

    def set_loop_function_Wtilde(self):
        self.w_tilde = np.sqrt(self.q2 + 2.0 * constants.m_pi**2)

    def set_loop_function_S(self):
        self.s = np.sqrt(self.Lambda_SFR**2 - 4.0 * constants.m_pi**2)

    def calc_loop_function_L(self, q):
        eps = 1.e-8
        if(self.Lambda_SFR < 2*constants.m_pi):
            # dimensional regularization
            if(q < eps): return 1.0
            else: return self.calc_loop_function_W(q)/q * np.log( (self.calc_loop_function_W(q) + q) / (2.0*constants.m_pi) )
        # spectral function regularization
        if(q < eps): return self.s / self.Lambda_SFR
        else: return self.calc_loop_function_W(q)/(2.0*q) * \
            np.log( (self.Lambda_SFR**2 * self.calc_loop_function_W(q)**2 + q**2 * self.s**2 + 2.0*self.Lambda_SFR*q*self.calc_loop_function_W(q)*self.s) / (4.0*constants.m_pi**2 * (self.Lambda_SFR**2 + q**2)))

    def set_loop_function_L(self):
        self.l = np.zeros(self.NMesh, dtype=np.float64)
        for idx, q in enumerate(self.q):
            self.l[idx] = self.calc_loop_function_L(q)

    def set_loop_function_A(self):
        self.a = np.zeros(self.NMesh, dtype=np.float64)
        eps = 1.e-8

        if(self.Lambda_SFR < 2*constants.m_pi):
            # dimensional regularization
            for idx, q in enumerate(self.q):
                if(q < eps): self.a[idx] = 1.0 / (4.0*constants.m_pi)
                elif(q >= eps): self.a[idx] = 1.0/(2.0*q) * np.arctan( q / (2.0*constants.m_pi) )
            return

        # spectral function regularization
        for idx, q in enumerate(self.q):
            if(q < eps):
                self.a[idx] = (self.Lambda_SFR - 2.0*constants.m_pi) / (4.0 * self.Lambda_SFR * constants.m_pi)
            elif(q >= eps):
                self.a[idx] = (1.0/(2.0*q)) * np.arctan( q*(self.Lambda_SFR - 2.0*constants.m_pi) / (q**2 + 2.0*self.Lambda_SFR*constants.m_pi))
        return

    def set_loop_function_H(self):
        self.h = np.zeros(self.NMesh, dtype=np.float64)
        delta = constants.m_delta - constants.m_nucl
        delta2 = delta**2
        for idx, q in enumerate(self.q):
            sigma = 2.0 * constants.m_pi**2 + q**2 - 2.0 * delta2
            self.h[idx] = 2.0 * sigma / (self.w[idx]**2 - 4.0*delta2) * \
                (self.l[idx] - self.calc_loop_function_L(2.0 * np.sqrt(delta2 - constants.m_pi**2)))
        return

    def set_loop_function_D(self):
        self.d = np.zeros(self.NMesh, dtype=np.float64)
        delta = constants.m_delta - constants.m_nucl
        a, b = 2.0*constants.m_pi, self.Lambda_SFR
        self.mu, self.mu_weights = np.polynomial.legendre.leggauss(self.n_mesh_loop_integral)
        for idx, q in enumerate(self.q):
            for i in range(self.n_mesh_loop_integral):
                mu = 0.5 * (b - a) * self.mu[i] + 0.5 * (a + b)
                w = 0.5 * (b - a) * self.mu_weights[i]
                self.d[idx] += w * np.arctan( np.sqrt(mu*mu - 4.0*constants.m_pi**2) / (2.0*delta) ) / (mu*mu + q*q)
            self.d[idx] /= delta

    def calc_pot(self, pbra, pket, s, j, t, tz):
        mass = 0
        if(tz==-1): mass = constants.m_p
        elif(tz== 0): mass = (constants.m_p+constants.m_n) * 0.5
        elif(tz== 1): mass = constants.m_n
        rel_factor = mass / np.sqrt(np.sqrt(pbra**2 + mass**2) * np.sqrt(pket**2 + mass**2))

        self.q = np.sqrt(pbra**2 + pket**2 - 2.0*pbra*pket*self.costh)
        self.q2 = self.q**2
        coupled = False
        if((-1)**(s+t+j) == 1): coupled = True
        if(s==0 and coupled): raise ValueError("This channel is not allowed")
        LOCt = self.LO_contacts(pbra, pket, s, j, tz, coupled)
        NLOC = self.NLO_contacts(pbra, pket, s, j, coupled)
        V1PE = self.OPE_potential(pbra, pket, s, j, t, tz, coupled)
        V2PE = self.TPE_potential(pbra, pket, s, j, t, coupled)
        mes = LOCt + NLOC + V1PE + V2PE
        mes = self.multiply_regulator(mes, pbra, pket)
        mes *= rel_factor
        mes /= (2.0*np.pi)**3
        return mes

if(__name__=="__main__"):
    parameters = LECs.N2LO_opt
    #parameters = LECs.DN2LOGO394
    nnpot = ChEFTNNPotential()
    nnpot.set_pot_parameters(parameters)
    pbra, pket = 50.0, 100.0
    s, j, t, tz = 1, 1, 1, 0
    me = nnpot.calc_pot(pbra, pket, s, j, t, tz)
    print("Matrix elements: ", me)
