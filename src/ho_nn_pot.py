#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, itertools, copy
import numpy as np
import scipy
from scipy.special import gamma, assoc_laguerre

if(__package__==None or __package__==""):
    import constants
    from partial_wave_basis import PartialWaveChannel, PartialWaveChannels
    from ho_partial_wave_basis import HOPartialWaveChannel, HOPartialWaveChannels
    from nn_pot import NNPotential
else:
    from . import constants
    from .partial_wave_basis import PartialWaveChannel, PartialWaveChannels
    from .ho_partial_wave_basis import HOPartialWaveChannel, HOPartialWaveChannels
    from .nn_pot import NNPotential

class HONNPotential:
    def __init__(self, Nmax, Jmax, hw):
        self.Nmax = Nmax
        self.Jmax = Jmax
        self.hw = hw
        self.HOPWChannels = HOPartialWaveChannels(Nmax, Jmax)
        self.vpot = {}
        for channel in self.HOPWChannels.Channels:
            J, Prty, S, Tz = channel.J, channel.Parity, channel.S, channel.Tz
            self.vpot[(J,Prty,S,Tz)] = np.zeros((channel.n_dim, channel.n_dim), dtype=np.float64)
        return
    
    def set_nn_potential(self, vpot_mom, add_coulomb=True):
        ho_radial_functions = {}
        channel = vpot_mom.pw.Channels[0]
        Mesh = channel.kmesh[:vpot_mom.pw.NMesh]
        Weight = channel.weights[:vpot_mom.pw.NMesh]
        for L in range(self.Nmax+1):
            for N in range((self.Nmax-L//2+1)):
                ho_radial_functions[(N,L)] = np.zeros((Mesh.size), dtype=np.float64)
                for i, k in enumerate(Mesh):
                    ho_radial_functions[(N,L)][i] = self._ho_radial_function_mom(N, L, k, self.hw)
        
        for channel in self.HOPWChannels.Channels:
            J, Prty, S, Tz = channel.J, channel.Parity, channel.S, channel.Tz
            for i in range(channel.n_dim):
                N1, L1, S1, J1, Tz1 = channel.get_quantum_numbers(i)
                for j in range(channel.n_dim):
                    N2, L2, S2, J2, Tz2 = channel.get_quantum_numbers(j)
                    mat = vpot_mom.get_block_matrix(J, Prty, S, Tz, L1, L2) * constants.hc**3
                    tmp_bra = ho_radial_functions[(N1,L1)] * Mesh**2 * Weight
                    tmp_ket = ho_radial_functions[(N2,L2)] * Mesh**2 * Weight
                    vpot_ij = tmp_bra @ mat @ tmp_ket
                    self.vpot[(J,Prty,S,Tz)][i,j] = vpot_ij

        if(not add_coulomb): return
        from scipy.integrate import quad
        for channel in self.HOPWChannels.Channels:
            J, Prty, S, Tz = channel.J, channel.Parity, channel.S, channel.Tz
            if(Tz!=-1): continue
            for i in range(channel.n_dim):
                N1, L1, S1, J1, Tz1 = channel.get_quantum_numbers(i)
                for j in range(channel.n_dim):
                    N2, L2, S2, J2, Tz2 = channel.get_quantum_numbers(j)
                    if(L1!=L2): continue
                    me = quad(lambda r: self._ho_radial_function(N1, L1, r, self.hw) * self._ho_radial_function(N2, L2, r, self.hw) * r * constants.hc * constants.alpha, 0.0, np.inf)[0]
                    self.vpot[(J,Prty,S,Tz)][i,j] += me

    def get_me(self, n1, l1, n2, l2, S, J, Tz):
        if((l1+l2)%2 == 1):
            raise ValueError(f"Parity is not conserved for l1={l1} and l2={l2}.")
        Prty = (-1)**l1
        channel = self.HOPWChannels.get_channel(J, Prty, S, Tz)
        i = channel.get_index(n1, l1)
        j = channel.get_index(n2, l2)
        return self.vpot[(J,Prty,S,Tz)][i,j]
    
    def _ho_radial_function_mom(self, n, l, k, hw):
        mu = constants.m_p * constants.m_n / (constants.m_p + constants.m_n)
        b = np.sqrt(constants.hc**2 / (mu * hw))
        x = k * b
        return (-1)**n * np.sqrt( 2.0*b * (gamma(n+1) / gamma(n+l+1.5)) ) * b * (x**l) * np.exp(-0.5*x*x) * assoc_laguerre(x*x, n, l+0.5)

    def _ho_radial_function(self, n, l, r, hw):
        mu = constants.m_p * 0.5 # this is because we are using this for the Coulomb potential
        b = np.sqrt(constants.hc**2 / (mu * hw))
        x = r / b
        return np.sqrt( (2.0/b) * (gamma(n+1) / gamma(n+l+1.5)) ) * (1.0/b) * (x**l) * np.exp(-0.5*x*x) * assoc_laguerre(x*x, n, l+0.5)
    
    def __str__(self):
        s = ""
        for channel in self.HOPWChannels.Channels:
            J, Prty, S, Tz = channel.J, channel.Parity, channel.S, channel.Tz
            for i in range(channel.n_dim):
                N1, L1, S1, J1, Tz1 = channel.get_quantum_numbers(i)
                for j in range(channel.n_dim):
                    N2, L2, S2, J2, Tz2 = channel.get_quantum_numbers(j)
                    s += f"J={J:3d}, Parity={Prty:3d}, S={S:3d}, Tz={Tz:3d}, N1={N1:3d}, L1={L1:3d}, N2={N2:3d}, L2={L2:3d}: {self.vpot[(J,Prty,S,Tz)][i,j]:12.6f}\n"
            
        return s
    
    def solve_deuteron(self):
        channel = self.HOPWChannels.get_channel(1, 1, 1, 0)
        vpot = self.vpot[(1,1,1,0)]
        tkin = np.zeros_like(vpot)
        for i in range(channel.n_dim):
            N1, L1, S1, J1, Tz1 = channel.get_quantum_numbers(i)
            for j in range(channel.n_dim):
                N2, L2, S2, J2, Tz2 = channel.get_quantum_numbers(j)
                if(L1!=L2): continue
                me = 0.0
                if(N1 == N2): me = (2*N1 + L1 + 1.5) * 0.5
                elif(N1 == N2-1): me = 0.5 * np.sqrt( (N1+1) * (N1+L1+1.5) )
                elif(N1 == N2+1): me = 0.5 * np.sqrt( (N2+1) * (N2+L2+1.5) )
                tkin[i,j] = me * self.hw
        ham = tkin + vpot
        eigvals, eigvecs = np.linalg.eigh(ham)
        idx = np.argmin(eigvals)
        energy = eigvals[idx]
        print(f"Deuteron energy: {energy:.6f} MeV")

def main():
    import LECs
    Nmax = 10
    Jmax = 1
    hw = 20.0
    pw = PartialWaveChannels(Jmax=Jmax, NMesh=40, kmax=4)
    vpot_mom = NNPotential(pw)
    parameters = LECs.N2LO_opt
    vpot_mom.set_potential(parameters)
    vpot_ho = HONNPotential(Nmax, Jmax, hw)
    vpot_ho.set_nn_potential(vpot_mom, add_coulomb=True)
    vpot_ho.solve_deuteron()

if(__name__=="__main__"):
    main()