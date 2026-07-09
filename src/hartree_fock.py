#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, itertools, functools
import math, copy
import numpy as np
import sympy as sp
if(__package__==None or __package__==""):
    import constants
    from single_partile_orbit import Orbits
    from two_body_space import TwoBodySpace
    from model_space import ModelSpace
    from partial_wave_basis import PartialWaveChannel, PartialWaveChannels
    from ho_partial_wave_basis import HOPartialWaveChannel, HOPartialWaveChannels
    from nn_pot import NNPotential
    from ho_nn_pot import HONNPotential
    from nuclear_operator import Operator
else:
    from . import constants
    from .single_partile_orbit import Orbits
    from .two_body_space import TwoBodySpace
    from .model_space import ModelSpace
    from .partial_wave_basis import PartialWaveChannel, PartialWaveChannels
    from .ho_partial_wave_basis import HOPartialWaveChannel, HOPartialWaveChannels
    from .nn_pot import NNPotential
    from .ho_nn_pot import HONNPotential
    from .nuclear_operator import Operator

np.set_printoptions(linewidth=200)

class HartreeFock:
    def __init__(self, hamiltonian):
        self.hamiltonian = hamiltonian
        self.model_space = hamiltonian.model_space
        orbits = self.model_space.orbits
        self.rho = np.zeros((orbits.get_number_orbits(), orbits.get_number_orbits()), dtype=np.float64) # < ho | rho | ho >
        self.coef = np.eye(orbits.get_number_orbits(), dtype=np.float64) # < ho | hf >
        self.T = self.hamiltonian.OneBody.copy()
        self.V = np.zeros_like(self.T)
        self.fock_matrix = np.zeros_like(self.T)
        self.set_monopole()

    def set_monopole(self):
        orbits = self.model_space.orbits
        self.monopole = {}
        for p in range(orbits.get_number_orbits()):
            for q in range(orbits.get_number_orbits()):
                for r in range(orbits.get_number_orbits()):
                    for s in range(orbits.get_number_orbits()):
                        if(r > p): continue
                        o_p = orbits.get_orbit(p)
                        o_q = orbits.get_orbit(q)
                        o_r = orbits.get_orbit(r)
                        o_s = orbits.get_orbit(s)
                        if((-1)**o_p.l != (-1)**o_r.l): continue
                        if((-1)**o_q.l != (-1)**o_s.l): continue
                        if(o_p.j2 != o_r.j2): continue
                        if(o_q.j2 != o_s.j2): continue
                        if(o_p.tz2 != o_r.tz2): continue
                        if(o_q.tz2 != o_s.tz2): continue
                        norm = 1.0
                        if(p==q): norm *= np.sqrt(2)
                        if(r==s): norm *= np.sqrt(2)
                        Jmin = abs(o_p.j2 - o_q.j2) // 2
                        Jmax = (o_p.j2 + o_q.j2) // 2
                        monopole = 0.0
                        for J in range(Jmin, Jmax+1):
                            if(p==q and J%2==1): continue
                            if(r==s and J%2==1): continue
                            monopole += (2*J+1) * self.hamiltonian.get_2bme_from_indices(p, q, r, s, J, J)
                        monopole *= norm / (o_p.j2 + 1)
                        self.monopole[(p, q, r, s)] = monopole

    def update_coefficients(self):
        for key, value in self.hamiltonian.one_body_channel.items():
            sub = self.fock_matrix[np.ix_(value, value)]
            eig, vec = np.linalg.eigh(sub)
            self.coef[np.ix_(value, value)] = vec

    def update_density_matrix(self):
        occ_mat = np.diag(self.model_space.occupations)
        self.rho = self.coef @ occ_mat @ self.coef.T

    def update_fock_matrix(self):
        self.V = np.zeros_like(self.V)
        for key, value in self.monopole.items():
            p, r, q, s = key
            self.V[p, q] += self.rho[r, s] * value
        self.V = self.V + self.V.T - np.diag(np.diag(self.V))
        self.fock_matrix = self.T + self.V

    def compute_energy(self):
        e1, e2 = 0.0, 0.0
        orbits = self.model_space.orbits
        for p in range(orbits.get_number_orbits()):
            o_p = orbits.get_orbit(p)
            for q in range(orbits.get_number_orbits()):
                e1 += self.rho[p, q] * self.T[p, q] * (o_p.j2 + 1)
                e2 += self.rho[p, q] * self.V[p, q] * (o_p.j2 + 1) * 0.5
        return self.hamiltonian.get_0bme(), e1, e2

    def transform_operator_ho_to_hf(self, op):
        op_out = op.transformation(self.coef)
        return op_out

    def solve(self, max_iter=1000, tol=1.e-8):
        self.update_density_matrix()
        self.update_fock_matrix()
        e0, e1, e2 = self.compute_energy()

        print(f"Initial energy: E0 = {e0:12.6f}, E1 = {e1:12.6f}, E2 = {e2:12.6f}, E = {e0+e1+e2:12.6f}")

        for i in range(max_iter):
            self.update_coefficients()
            rho_old = self.rho.copy()
            self.update_density_matrix()
            self.update_fock_matrix()
            e0, e1, e2 = self.compute_energy()
            print(f"Iteration {i+1:4d}: E0 = {e0:12.6f}, E1 = {e1:12.6f}, E2 = {e2:12.6f}, E = {e0+e1+e2:12.6f}")

            if(np.linalg.norm(rho_old - self.rho) < tol):
                print(f"Converged after {i+1} iterations.")
                break

    def second_order_correction(self, ham):
        e2 = 0.0
        orbits = self.model_space.orbits
        for p in self.model_space.holes:
            o_p = orbits.get_orbit(p)
            for q in self.model_space.particles:
                o_q = orbits.get_orbit(q)
                if(o_p.l != o_q.l): continue
                if(o_p.j2 != o_q.j2): continue
                if(o_p.tz2 != o_q.tz2): continue
                e2 += ham.get_1bme(p, q)**2 / (ham.get_1bme(p, p) - ham.get_1bme(q, q)) * (o_p.j2 + 1)
        
        for channels in ham.TwoBody.keys():
            ichbra, ichket = channels
            chbra = self.model_space.two_body_space.get_channel(ichbra)
            chket = self.model_space.two_body_space.get_channel(ichket)
            for ibra in range(chbra.get_number_states()):
                p, q = chbra.get_indices(ibra)
                o_p = orbits.get_orbit(p)
                o_q = orbits.get_orbit(q)
                if(self.model_space.is_particle(p) or self.model_space.is_particle(q)): continue
                for iket in range(chket.get_number_states()):
                    r, s = chket.get_indices(iket)
                    o_r = orbits.get_orbit(r)
                    o_s = orbits.get_orbit(s)
                    if(self.model_space.is_hole(r) or self.model_space.is_hole(s)): continue
                    e2 += ham.get_2bme_from_indices(p, q, r, s, chbra.J, chket.J)**2 / \
                        (ham.get_1bme(p, p) + ham.get_1bme(q, q) - \
                        ham.get_1bme(r, r) - ham.get_1bme(s, s)) * (2*chket.J+1)
        return e2



def main():

    import LECs
    emax = 4
    Nmax = 2*emax
    Jrel_max = 4
    hw = 20.0
    pw = PartialWaveChannels(Jmax=Jrel_max, NMesh=40, kmax=6)

    parameters = LECs.N2LO_opt
    filename = "N2LOopt.npz"
    vpot_mom = NNPotential(pw)
    if(os.path.exists(filename)):
        vpot_mom.load_npz_into(filename)
    else:
        vpot_mom.set_potential(parameters)
        vpot_mom.save_npz(filename)

    vpot_ho = HONNPotential(Nmax, Jrel_max, hw)
    vpot_ho.set_nn_potential(vpot_mom)

    ms = ModelSpace()
    ms.set_model_space_from_emax(emax)
    ms.set_frequency(hw)
    ms.set_reference_from_string("O16")
    ms.assign_particle_hole()
    tkin, vnn, ham = Operator(ms), Operator(ms), Operator(ms)
    tkin.set_kinetic_term()
    vnn.set_nn_interaction(vpot_ho)
    ham = tkin + vnn
    hf = HartreeFock(ham)
    hf.solve()
    ham = hf.transform_operator_ho_to_hf(ham)
    ham.take_normal_ordering()
    e2 = hf.second_order_correction(ham)
    print(f"HF: {ham.get_0bme():12.6f}, 2nd order correction: {e2:12.6f}, Total: {ham.get_0bme() + e2:12.6f}")

if(__name__=="__main__"):
    main()
