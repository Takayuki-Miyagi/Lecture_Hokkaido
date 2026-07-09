#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, itertools
import numpy as np
import pickle

class HOPartialWaveChannel:
    def __init__(self, Nmax, J, Parity, S, Tz):
        self.Nmax = Nmax
        self.J = J
        self.Parity = Parity
        self.S = S
        self.Tz = Tz
        self.nllist = []
        self.nl2idx = {}
        cnt = 0
        for L in range(abs(J-S), J+S+1):
            if((-1)**L != self.Parity): continue
            if(abs(self.Tz)==1 and (-1)**(L+S) == -1): continue
            for N in range(self.Nmax//2+1):
                if(2*N+L > self.Nmax): continue
                self.nllist.append((N,L))
                self.nl2idx[(N,L)] = cnt
                cnt += 1

        self.n_dim = len(self.nllist)
        return

    def get_index(self, n, l):
        return self.nl2idx[(n,l)]

    def get_quantum_numbers(self, idx):
        # return n, l, S, J, Tz, i.e. |n, l, S, J, Tz>
        return self.nllist[idx][0], self.nllist[idx][1], self.S, self.J, self.Tz

class HOPartialWaveChannels:
    def __init__(self, Nmax, Jmax, Tz=None):
        self.Channels = []
        self.ChannelIndex = {}
        self.Nmax = Nmax
        self.Jmax = Jmax
        self.set_channels(Tz)

    def set_channels(self, Tz=None):
        if(Tz==None): Tz_list = [-1,0,1]
        else: Tz_list = [Tz]
        for J, Parity, S, tz in itertools.product(range(self.Jmax+1), [1,-1], [0,1], Tz_list):
            tmp = HOPartialWaveChannel(self.Nmax, J, Parity, S, tz)
            if(tmp.n_dim == 0): continue
            self.Channels.append(HOPartialWaveChannel(self.Nmax, J, Parity, S, tz))
            self.ChannelIndex[(J,Parity,S,tz)] = len(self.Channels)-1
        return

    def get_number_channels(self):
        return len(self.Channels)

    def get_channel_index(self, J, Parity, S, Tz):
        return self.ChannelIndex[(J,Parity,S,Tz)]

    def get_channel(self, J, Parity, S, Tz):
        return self.Channels[self.get_channel_index(J, Parity, S, Tz)]

    def get_jmax(self):
        return self.Jmax

if(__name__=="__main__"):
    Nmax = 20
    test = HOPartialWaveChannels(Nmax,Jmax=2)
    J, Prty, S, Tz = 1, 1, 1, 0
    channel = test.get_channel(J, Prty, S, Tz)
    print("Partial-wave bases:")
    for i in range(channel.n_dim):
        N, L, S, J, Tz = channel.get_quantum_numbers(i)
        print(f"N = {N:3d}, L = {L:3d}, S = {S:3d}, J = {J:3d}, Tz = {Tz:3d}")
