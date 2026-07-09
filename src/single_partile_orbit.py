#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, itertools
import numpy as np

class Orbit:
    def __init__(self):
        self.n, self.l, self.j2, self.tz2, self.e = -1, -1, -1, -1, -1
        return

    def set_orbit(self, *nljz):
        self.n, self.l, self.j2, self.tz2 = nljz
        self.e = 2*self.n+self.l
        return

    def get_nljz(self):
        return (self.n, self.l, self.j2, self.tz2)

class Orbits:
    def __init__(self):
        self.emax = -1
        self.orbits = []
        self.nljz_idx = {}
        self._labels_orbital_angular_momentum = ('s',\
                'p','d','f','g','h','i','k','l','m','n',\
                'o','q','r','t','u','v','w','x','y','z')
        pass

    def add_orbit(self,*nljz):
        if(nljz in self.nljz_idx):
            if(self.verbose): print("The orbit ({:3d},{:3d},{:3d},{:3d}) is already there.".format(*nljz) )
            return
        self.norbs = len(self.orbits)+1
        idx = self.norbs-1
        self.nljz_idx[nljz] = idx
        orb = Orbit()
        orb.set_orbit(*nljz)
        self.orbits.append( orb )

    def get_emax(self):
        return self.emax
    
    def get_number_orbits(self):
        return len(self.orbits)
    
    def get_orbit(self, idx):
        return self.orbits[idx]
    
    def get_orbit_label(self, idx):
        o = self.get_orbit(idx)
        n, l, j2, tz2 = o.get_nljz()
        l_label = self._labels_orbital_angular_momentum[l]
        j_label = f"{j2}/2"
        tz_label = "p" if tz2==-1 else "n"
        return f"{tz_label}{n}{l_label}{j_label}"

    def get_orbit_index(self,*nljz):
        return self.nljz_idx[nljz]

    def get_orbit_index_from_orbit(self,o):
        return self.get_orbit_index(o.n,o.l,o.j2,o.tz2)

    def get_orbit_index_from_tuple(self,nljz):
        return self.nljz_idx[nljz]

    def set_orbits(self, emax):
        self.emax = emax
        for N in range(emax+1):
            for l in range(N+1):
                if( (N-l)%2 == 1 ): continue
                n = (N-l)//2
                for j in [2*l-1, 2*l+1]:
                    if( j<0 ): continue
                    for z in [-1,1]:
                        self.add_orbit( n,l,j,z )

    def print_orbits(self):
        string = "Orbits list:\n"
        string += "idx,  n,  l, j2,tz2,  e\n"
        for o in self.orbits:
            nljz = o.get_nljz()
            n,l,j,z = o.get_nljz()
            idx = self.get_orbit_index(n,l,j,z)
            idx = self.get_orbit_index_from_tuple( nljz )
            idx = self.get_orbit_index_from_orbit( o )
            string += "{:3d},{:3d},{:3d},{:3d},{:3d},{:3d}\n".format(idx,*nljz,o.e)
        return string[:-1]
    def __str__(self):
        return self.print_orbits()

def main():
    orbs = Orbits()
    orbs.set_orbits(3)
    print(orbs)
if(__name__ == "__main__"):
    main()
