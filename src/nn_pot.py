#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, itertools, copy
import numpy as np

if(__package__==None or __package__==""):
    import constants
    import LECs
    from partial_wave_basis import PartialWaveChannel, PartialWaveChannels
    from chiral_eft import ChEFTNNPotential
else:
    from . import constants
    from . import LECs
    from .partial_wave_basis import PartialWaveChannel, PartialWaveChannels
    from .chiral_eft import ChEFTNNPotential

class NNPotential:
    def __init__(self, pw):
        self.pw = pw
        self.vpot = {}
        for channel in pw.Channels:
            J, Prty, S, Tz = channel.J, channel.Parity, channel.S, channel.Tz
            self.vpot[(J,Prty,S,Tz)] = np.zeros((channel.n_dim, channel.n_dim), dtype=np.float64)

    def set_potential(self, pot_parameters):
        nnpot = ChEFTNNPotential()
        nnpot.set_pot_parameters(pot_parameters)
        channel = self.pw.Channels[0]
        Mesh = channel.kmesh[:self.pw.NMesh]
        print("NN potential")
        for j, s, tz in itertools.product(range(0, self.pw.Jmax+1), [0, 1], [-1, 0, 1]):
            if(not (j,1,s,tz) in self.vpot and not (j,-1,s,tz) in self.vpot): continue
            print(f"Computing NN potential for Channel: J={j:2d}, S={s:2d}, Tz={tz:2d}")
            for ibra, ki in enumerate(Mesh):
                for iket, kj in enumerate(Mesh):

                    pbra = ki * constants.hc
                    pket = kj * constants.hc
                    vt0 = np.zeros(6, dtype=np.float64)
                    vt1 = np.zeros(6, dtype=np.float64)

                    # T=0 case
                    if(tz==0):
                        prty = (-1)**(s+1)
                        if((j,prty,s,tz) in self.vpot):
                            vt0 = nnpot.calc_pot(pbra, pket, s, j, 0, tz)
                            channel = self.pw.get_channel(j, prty, s, tz)
                            coupled = False
                            if((-1)**(s+0+j) == 1): coupled = True
                            if(s==0): self.vpot[(j,prty,s,tz)][channel.get_index(ibra,j), channel.get_index(iket,j)] += vt0[0]
                            if(s==1 and not coupled): self.vpot[(j,prty,s,tz)][channel.get_index(ibra,j), channel.get_index(iket,j)] += vt0[1]
                            if(coupled):
                                self.vpot[(j,prty,s,tz)][channel.get_index(ibra,j+1), channel.get_index(iket,j+1)] += vt0[2]
                                if(j>0):
                                    self.vpot[(j,prty,s,tz)][channel.get_index(ibra,j-1), channel.get_index(iket,j-1)] += vt0[3]
                                    self.vpot[(j,prty,s,tz)][channel.get_index(ibra,j+1), channel.get_index(iket,j-1)] += vt0[4]
                                    self.vpot[(j,prty,s,tz)][channel.get_index(ibra,j-1), channel.get_index(iket,j+1)] += vt0[5]

                    # T=1 case
                    prty = (-1)**s
                    if((j,prty,s,tz) in self.vpot):
                        vt1 = nnpot.calc_pot(pbra, pket, s, j, 1, tz)
                        channel = self.pw.get_channel(j, prty, s, tz)
                        coupled = False
                        if((-1)**(s+1+j) == 1): coupled = True
                        if(s==0): self.vpot[(j,prty,s,tz)][channel.get_index(ibra,j), channel.get_index(iket,j)] += vt1[0]
                        if(s==1 and not coupled): self.vpot[(j,prty,s,tz)][channel.get_index(ibra,j), channel.get_index(iket,j)] += vt1[1]
                        if(coupled):
                            self.vpot[(j,prty,s,tz)][channel.get_index(ibra,j+1), channel.get_index(iket,j+1)] += vt1[2]
                            if(j>0):
                                self.vpot[(j,prty,s,tz)][channel.get_index(ibra,j-1), channel.get_index(iket,j-1)] += vt1[3]
                                self.vpot[(j,prty,s,tz)][channel.get_index(ibra,j+1), channel.get_index(iket,j-1)] += vt1[4]
                                self.vpot[(j,prty,s,tz)][channel.get_index(ibra,j-1), channel.get_index(iket,j+1)] += vt1[5]


    def save_npz(self, filename, compressed=True):
        # Save potential matrices and the partial-wave channel data into one .npz
        kwargs = {}
        # store number of channels from pw
        kwargs['n_channels'] = np.int64(len(self.pw.Channels))
        kwargs['pw_Jmax'] = np.int64(self.pw.Jmax)
        kwargs['pw_NMesh'] = np.int64(self.pw.NMesh)
        kwargs['pw_kmin'] = np.float64(self.pw.kmin)
        kwargs['pw_kmax'] = np.float64(self.pw.kmax)
        for i, ch in enumerate(self.pw.Channels):
            J, Prty, S, Tz = ch.J, ch.Parity, ch.S, ch.Tz
            key = (J,Prty,S,Tz)
            kwargs[f'vpot_{i}'] = np.asarray(self.vpot[key], dtype=np.float64)
        if compressed:
            np.savez_compressed(filename, **kwargs)
        else:
            np.savez(filename, **kwargs)

    @staticmethod
    def load_npz(filename):
        data = np.load(filename, allow_pickle=False)
        n_channels = int(data['n_channels'])
        # Reconstruct pw (PartialWaveChannels-like object)
        Jmax = int(data['pw_Jmax'])
        NMesh = int(data['pw_NMesh'])
        kmin = float(data['pw_kmin'])
        kmax = float(data['pw_kmax'])
        pw = PartialWaveChannels(Jmax, kmin, kmax, NMesh)
        obj = object.__new__(NNPotential)
        obj.pw = pw
        obj.vpot = {}
        for i, ch in enumerate(pw.Channels):
            J, Prty, S, Tz = ch.J, ch.Parity, ch.S, ch.Tz
            key = (J, Prty, S, Tz)
            vname = f'vpot_{i}'
            obj.vpot[key] = np.asarray(data[vname], dtype=np.float64)
        return obj


    def load_npz_into(self, filename):
        """Load data from .npz and overwrite this instance's `pw` and `vpot`.

        This is provided for convenience so callers can do
            nnpot.load_npz_into('file.npz')
        instead of replacing the variable with the return value of
        `NNPotential.load_npz('file.npz')`.
        """
        loaded = NNPotential.load_npz(filename)
        self.pw = loaded.pw
        self.vpot = loaded.vpot
        return None
    
    def get_block_matrix(self, J, Parity, S, Tz, lbra, lket):
        channel = self.pw.get_channel(J, Parity, S, Tz)
        Mesh = channel.kmesh[:channel.NMesh]
        block_matrix = np.zeros((Mesh.size, Mesh.size), dtype=np.float64)
        for ibra in range(Mesh.size):
            for iket in range(Mesh.size):
                block_matrix[ibra, iket] = self.vpot[(J, Parity, S, Tz)][channel.get_index(ibra,lbra), channel.get_index(iket,lket)]
        return block_matrix

    def solve_deuteron(self):
        if(self.pw.Jmax < 1): raise ValueError("J=1 channels is not computed!")
        J, Prty, S, Tz = 1, 1, 1, 0
        channel = self.pw.get_channel(J, Prty, S, Tz)
        mu = constants.m_p * constants.m_n / (constants.m_p + constants.m_n)
        V = copy.copy(self.vpot[(J, Prty, S, Tz)])
        Tkin = np.zeros(V.shape, dtype=np.float64)
        for ibra in range(V.shape[0]):
            for iket in range(V.shape[1]):
                if(ibra==iket): Tkin[ibra,iket] = channel.kmesh[iket]**2 * constants.hc**2 / (2*mu)
                V[ibra,iket] *= constants.hc**3 * np.sqrt(channel.weights[ibra] * channel.weights[iket]) * channel.kmesh[ibra] * channel.kmesh[iket]
        hamil = Tkin + V
        eig, vec = np.linalg.eigh(hamil)
        print(f"Deuteron ground-state energy: {eig[0]:12.6f} MeV")

if(__name__=="__main__"):
    parameters = LECs.N2LO_opt
    pw = PartialWaveChannels(Jmax=4, NMesh=30, kmax=6)
    nnpot = NNPotential(pw)
    nnpot.set_potential(parameters)
    nnpot.solve_deuteron()
