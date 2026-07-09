
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, itertools, re
import numpy as np
if(__package__==None or __package__==""):
    from single_partile_orbit import Orbits
    from two_body_space import TwoBodySpace
else:
    from .single_partile_orbit import Orbits
    from .two_body_space import TwoBodySpace

class ModelSpace:
    def __init__(self, rank=2):
        self.rank = rank
        self.orbits = None
        self.two_body_space = None
        self.hw = None
        self.periodic_table = [
            'NA',
            'H',  'He', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne',
            'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar', 'K',  'Ca',
            'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
            'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 'Rb', 'Sr', 'Y',  'Zr',
            'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn',
            'Sb', 'Te', 'I',  'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd',
            'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb',
            'Lu', 'Hf', 'Ta', 'W',  'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
            'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th',
            'Pa', 'U',  'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm',
            'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds',
            'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og' ]
        return
    
    def set_model_space_from_orbit(self, orbits):
        self.orbits = orbits
        self.emax = orbits.get_emax()
        self.two_body_space = TwoBodySpace(orbits)
        return

    def set_model_space_from_emax(self, emax):
        self.emax = emax
        self.orbits = Orbits()
        self.orbits.set_orbits(emax)
        self.two_body_space = TwoBodySpace(self.orbits)
        return
    
    def set_frequency(self, hw):
        self.hw = hw

    def get_frequency(self):
        return self.hw

    def get_orbits(self):
        return self.orbits

    def get_two_body_space(self):
        return self.two_body_space
    
    def set_reference_from_hole_occupancy(self, hole_occupancy):
        """
        hole_occupancy: list of tuples (n,l,j2,tz2,occ)
        """
        self.occupations = np.array([0.0]*len(self.orbits.orbits))
        Z, N = 0, 0
        for nljzocc in hole_occupancy:
            n, l, j2, tz2, occ = nljzocc
            if(tz2 ==-1): Z += occ * (j2+1)
            if(tz2 == 1): N += occ * (j2+1)
            idx = self.orbits.get_orbit_index(n, l, j2, tz2)
            self.occupations[idx] = occ
        A = Z + N
        self.Z = int(Z)
        self.N = int(N)
        self.A = int(A)

    def set_reference_from_string(self, ref):
        """
        ref: string of the form "He4", "O16", etc.
        A standard orbit ordering is assumed.
        """
        hole_occupancy = []
        self.Z, self.N, self.A = self._ZNA_from_str(ref)
        _Z, _N = 0, 0
        for e in range(self.emax+1):
            for g in range(2*e+1, -2*e-3, -4):
                j2 = abs(g)
                if(g<0): l = (j2+1)//2
                else: l = (j2-1)//2
                n = (e-l)//2

                _Z += (j2+1)
                _N += (j2+1)
                if(self.Z - _Z >=0): hole_occupancy.append((n,l,j2,-1,1.0))
                elif(self.Z -_Z < 0):
                    v = self.Z - _Z + j2 + 1
                    if(v > 0): hole_occupancy.append((n,l,j2,-1,v/(j2+1)))
                    else: hole_occupancy.append((n,l,j2,-1,0.0))

                if(self.N - _N >=0): hole_occupancy.append((n,l,j2,1,1.0))
                elif(self.N -_N < 0):
                    v = self.N - _N + j2 + 1
                    if(v > 0): hole_occupancy.append((n,l,j2,1,v/(j2+1)))
                    else: hole_occupancy.append((n,l,j2,1,0.0))
        
        if(self.Z - _Z > 0 or self.N - _N > 0):
            raise ValueError(f"The reference state is overfilled. Z={self.Z}, N={self.N}, but {_Z} protons and {_N} neutrons are filled.")
        self.set_reference_from_hole_occupancy(hole_occupancy)

        
    def _ZNA_from_str(self, Nucl):
        """  
        ex.) Nucl="O16" -> Z=8, N=8, A=16
        """
        isdigit = re.search(r'\d+', Nucl)
        A = int( isdigit.group() )
        asc = Nucl[:isdigit.start()] + Nucl[isdigit.end():]
        asc = asc.lower()
        asc = asc[0].upper() + asc[1:]
        Z = self.periodic_table.index(asc)
        N = A-Z
        return Z, N, A
    
    def assign_particle_hole(self):
        """
        particle_hole: list of 1 (particle) or 0 (hole) for each orbit
        """
        self.particle_hole = np.zeros(self.orbits.get_number_orbits(), dtype=np.int32)
        for idx, occ in enumerate(self.occupations):
            if(occ > 1.e-4): self.particle_hole[idx] = 0
            else: self.particle_hole[idx] = 1
        self.holes = np.where(self.particle_hole==0)[0]
        self.particles = np.where(self.particle_hole==1)[0]

    def is_particle(self, idx):
        if(self.particle_hole[idx] == 1): return True
        else: return False

    def is_hole(self, idx):
        if(self.particle_hole[idx] == 0): return True
        else: return False
    
    def __str__(self):
        string = f"Model space with emax={self.emax}, Z={self.Z}, N={self.N}, A={self.A}, Frequency={self.hw} MeV\n"
        string+= "Hole occupancies:\n"
        for idx, occ in enumerate(self.occupations):
            o = self.orbits.get_orbit(idx)
            n,l,j2,tz2 = o.get_nljz()
            if(occ < 1.e-4): continue
            string += f"  Orbit index {idx:3d}: label={self.orbits.get_orbit_label(idx):>10s}, n={n:2d}, l={l:2d}, 2*j={j2:4d}, 2*tz={tz2:4d}, occupation={occ:7.4f}\n"
        return string
    
def main():
    ms = ModelSpace()
    ms.set_model_space_from_emax(4)
    ms.set_frequency(16.0)
    ms.set_reference_from_string("O16")
    ms.assign_particle_hole()
    print(ms)

if(__name__=="__main__"):
    main()
    