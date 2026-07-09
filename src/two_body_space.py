#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, itertools
import numpy as np
if(__package__==None or __package__==""):
    from single_partile_orbit import Orbits
else:
    from .single_partile_orbit import Orbits

class TwoBodyChannel:
    def __init__(self, orbits, J, Parity, Tz):
        self.orbits = orbits
        self.J = J
        self.Parity = Parity
        self.Tz = Tz
        self.orbit1_index = []
        self.orbit2_index = []
        self.phase_from_indices = {}
        self.index_from_indices = {}
        self.number_states = 0
        self._set_two_body_channel()
        return

    def _set_two_body_channel(self):
        orbs = self.orbits
        for oa, ob in itertools.product(orbs.orbits, repeat=2):
            ia = orbs.get_orbit_index_from_orbit( oa )
            ib = orbs.get_orbit_index_from_orbit( ob )
            if( (oa.tz2 + ob.tz2) != 2*self.Tz ): continue
            if( (-1)**(oa.l + ob.l) != self.Parity ): continue
            if( self._triag( oa.j2, ob.j2, 2*self.J ) ): continue
            if(ia == ib and self.J%2==1): continue
            if(self.Tz==0 and oa.tz2==1 and ob.tz2==-1): continue
            if(abs(self.Tz)==1 and ia>ib): continue
            self.orbit1_index.append( ia )
            self.orbit2_index.append( ib )
            idx = len( self.orbit1_index )-1
            self.index_from_indices[(ia,ib)] = idx
            self.index_from_indices[(ib,ia)] = idx
            self.phase_from_indices[(ia,ib)] = 1
            self.phase_from_indices[(ib,ia)] = -(-1)**( (oa.j2+ob.j2)//2 - self.J )
        self.number_states = len( self.orbit1_index )

    def get_number_states(self):
        return self.number_states

    def get_indices(self,idx):
        return self.orbit1_index[idx], self.orbit2_index[idx]

    def get_orbits(self,idx):
        ia, ib = self.get_indices(idx)
        return self.orbits.get_orbit(ia), self.orbits.get_orbit(ib)

    def get_JPZ(self):
        return self.J, self.Parity, self.Tz

    def _triag(self,J1,J2,J3):
        b = True
        if(abs(J1-J2) <= J3 <= J1+J2): b = False
        return b

class TwoBodySpace:
    def __init__(self,orbits):
        self.orbits = orbits
        self.index_from_JPTz = {}
        self.channels = []
        self.number_channels = 0
        for J in range(2*self.orbits.emax+2):
            for Parity in [1,-1]:
                for Tz in [-1,0,1]:
                    channel = TwoBodyChannel(self.orbits, J, Parity, Tz)
                    if( channel.get_number_states() == 0): continue
                    self.channels.append( channel )
                    idx = len(self.channels) - 1
                    self.index_from_JPTz[(J,Parity,Tz)] = idx
        self.number_channels = len(self.channels)

    def get_number_channels(self):
        return self.number_channels

    def get_index(self,*JPZ):
        return self.index_from_JPTz[JPZ]

    def get_channel(self,idx):
        return self.channels[idx]

    def get_channel_from_JPTz(self,*JPZ):
        return self.get_channel( self.get_index(*JPZ) )

    def print_channels(self):
        print("  Two-body channels list ")
        print("  J,par,  Tz, # of states")
        for channel in self.channels:
            J,P,Z = channel.get_JPZ()
            print("{:3d},{:3d},{:4d},{:12d}".format(J,P,Z,channel.get_number_states()))

def main():
    orbs = Orbits()
    orbs.set_orbits(emax=0)
    two = TwoBodySpace(orbits=orbs)
    two.print_channels()
if(__name__=="__main__"):
    main()
