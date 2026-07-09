#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, itertools
import numpy as np
import pickle

class PartialWaveChannel:
    def __init__(self, J, Parity, S, Tz, kmin=0, kmax=6, NMesh=100):
        self.J = J
        self.Parity = Parity
        self.S = S
        self.Tz = Tz
        self.Llist = []
        for L in range(abs(J-S), J+S+1):
            if((-1)**L != self.Parity): continue
            if(abs(self.Tz)==1 and (-1)**(L+S) == -1): continue
            self.Llist.append(L)
        self.n_blocks = len(self.Llist)

        self.kmin = kmin
        self.kmax = kmax
        self.NMesh = NMesh
        self.kmesh = np.zeros(self.NMesh*self.n_blocks, dtype=np.float64)
        self.weights = np.zeros(self.NMesh*self.n_blocks, dtype=np.float64)
        self.n_dim = self.NMesh*self.n_blocks
        self.set_gauss_legendre_mesh(self.kmin, self.kmax, self.NMesh, self.Llist)
        return

    def set_gauss_legendre_mesh(self, kmin, kmax, NMesh, Llist):
        for i in range(len(Llist)):
            x, w = np.polynomial.legendre.leggauss(NMesh)
            self.kmesh[i*NMesh:(i+1)*NMesh] = 0.5 * (kmax-kmin) * x + 0.5 * (kmin + kmax)
            self.weights[i*NMesh:(i+1)*NMesh] = 0.5 * (kmax-kmin) * w
        return
    
    def get_index(self, i, l):
        return i + self.NMesh*self.Llist.index(l)

    def get_quantum_numbers(self, i):
        # return k, w, L, S, J, Tz, i.e. |k, w, L, S, J, Tz>
        return self.kmesh[i], self.weights[i], self.Llist[i//self.NMesh], self.S, self.J, self.Tz

class PartialWaveChannels:
    def __init__(self, Jmax, kmin=0, kmax=6, NMesh=100, Tz=None):
        self.Channels = []
        self.ChannelIndex = {}
        self.Jmax = Jmax
        self.NMesh = NMesh
        self.kmin = kmin
        self.kmax = kmax
        self.set_channels(Tz)
    
    def set_channels(self, Tz=None):
        if(Tz==None): Tz_list = [-1,0,1]
        else: Tz_list = [Tz]
        for J, Parity, S, tz in itertools.product(range(self.Jmax+1), [1,-1], [0,1], Tz_list):
            tmp = PartialWaveChannel(J, Parity, S, tz, kmin=self.kmin, kmax=self.kmax, NMesh=self.NMesh)
            if(tmp.n_dim == 0): continue
            self.Channels.append(PartialWaveChannel(J, Parity, S, tz, kmin=self.kmin, kmax=self.kmax, NMesh=self.NMesh))
            self.ChannelIndex[(J,Parity,S,tz)] = len(self.Channels)-1
        return
    
    def get_number_channels(self):
        return len(self.Channels)

    def get_channel_index(self, J, Parity, S, Tz):
        return self.ChannelIndex[(J,Parity,S,Tz)]

    def get_channel(self, J, Parity, S, Tz):
        return self.Channels[self.get_channel_index(J, Parity, S, Tz)]

    def save_npz(self, filename, compressed=True):
        # Save all channels into a single .npz file (arrays + metadata)
        kwargs = {}
        kwargs['n_channels'] = np.int64(len(self.Channels))
        kwargs['Jmax'] = np.int64(self.Jmax)
        kwargs['NMesh'] = np.int64(self.NMesh)
        for i, ch in enumerate(self.Channels):
            kwargs[f'J_{i}'] = np.int64(ch.J)
            kwargs[f'Parity_{i}'] = np.int64(ch.Parity)
            kwargs[f'S_{i}'] = np.int64(ch.S)
            kwargs[f'Tz_{i}'] = np.int64(ch.Tz)
            kwargs[f'kmin_{i}'] = np.float64(ch.kmin)
            kwargs[f'kmax_{i}'] = np.float64(ch.kmax)
            kwargs[f'NMesh_{i}'] = np.int64(ch.NMesh)
            kwargs[f'n_dim_{i}'] = np.int64(ch.n_dim)
            kwargs[f'Llist_{i}'] = np.array(ch.Llist, dtype=np.int64)
            kwargs[f'kmesh_{i}'] = np.asarray(ch.kmesh, dtype=np.float64)
            kwargs[f'weights_{i}'] = np.asarray(ch.weights, dtype=np.float64)
        if compressed:
            np.savez_compressed(filename, **kwargs)
        else:
            np.savez(filename, **kwargs)

    @staticmethod
    def load_npz(filename):
        data = np.load(filename, allow_pickle=False)
        n_channels = int(data['n_channels'])
        obj = object.__new__(PartialWaveChannels)
        obj.Channels = []
        obj.ChannelIndex = {}
        obj.Jmax = int(data['Jmax'])
        obj.NMesh = int(data['NMesh'])
        for i in range(n_channels):
            J = int(data[f'J_{i}'])
            Parity = int(data[f'Parity_{i}'])
            S = int(data[f'S_{i}'])
            Tz = int(data[f'Tz_{i}'])
            ch = object.__new__(PartialWaveChannel)
            ch.J = J
            ch.Parity = Parity
            ch.S = S
            ch.Tz = Tz
            ch.Llist = list(np.asarray(data[f'Llist_{i}'], dtype=np.int64).tolist())
            ch.n_blocks = len(ch.Llist)
            ch.kmin = float(data[f'kmin_{i}'])
            ch.kmax = float(data[f'kmax_{i}'])
            ch.NMesh = int(data[f'NMesh_{i}'])
            ch.kmesh = np.asarray(data[f'kmesh_{i}'], dtype=np.float64)
            ch.weights = np.asarray(data[f'weights_{i}'], dtype=np.float64)
            ch.n_dim = int(data[f'n_dim_{i}'])
            obj.Channels.append(ch)
            obj.ChannelIndex[(J,Parity,S,Tz)] = len(obj.Channels)-1
        return obj

    def get_jmax(self):
        return self.Jmax

    def get_nmesh(self):
        return self.NMesh

if(__name__=="__main__"):
    test = PartialWaveChannels(Jmax=2)
    J, Prty, S, Tz = 1, 1, 1, 0
    channel = test.get_channel(J, Prty, S, Tz)
    print("Partial-wave bases:")
    for i in range(channel.n_dim):
        k, w, L, S, J, Tz = channel.get_quantum_numbers(i)
        print(f"k = {k:7.4f}, w = {w:8.3e}, L = {L:3d}, S = {S:3d}, J = {J:3d}, Tz = {Tz:3d}")
