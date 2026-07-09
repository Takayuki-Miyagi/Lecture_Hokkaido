#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os, itertools, functools
import math
import numpy as np
import sympy as sp
from scipy.special import factorial, factorial2
from sympy.physics.wigner import wigner_3j, wigner_6j, wigner_9j, clebsch_gordan
if(__package__==None or __package__==""):
    import constants
    from single_partile_orbit import Orbits
    from two_body_space import TwoBodySpace
    from model_space import ModelSpace
    from partial_wave_basis import PartialWaveChannel, PartialWaveChannels
    from ho_partial_wave_basis import HOPartialWaveChannel, HOPartialWaveChannels
    from nn_pot import NNPotential
    from ho_nn_pot import HONNPotential
else:
    from . import constants
    from .single_partile_orbit import Orbits
    from .two_body_space import TwoBodySpace
    from .model_space import ModelSpace
    from .partial_wave_basis import PartialWaveChannel, PartialWaveChannels
    from .ho_partial_wave_basis import HOPartialWaveChannel, HOPartialWaveChannels
    from .nn_pot import NNPotential
    from .ho_nn_pot import HONNPotential

@functools.lru_cache(maxsize=None)
def _sixj(j1, j2, j3, j4, j5, j6):
    return float(wigner_6j(j1, j2, j3, j4, j5, j6))
@functools.lru_cache(maxsize=None)
def _ninej(j1, j2, j3, j4, j5, j6, j7, j8, j9):
    return float(wigner_9j(j1, j2, j3, j4, j5, j6, j7, j8, j9))
@functools.lru_cache(maxsize=None)
def _clebsch_gordan(j1, j2, j3, m1, m2, m3):
    return float(clebsch_gordan(j1, j2, j3, m1, m2, m3))
def _ls_coupling(la, ja, lb, jb, Lab, Sab, J):
    return np.sqrt( (2*ja+1)*(2*jb+1)*(2*Lab+1)*(2*Sab+1) ) * \
            np.float( wigner_9j( la, 0.5, ja, lb, 0.5, jb, Lab, Sab, J) )
class Operator:
    def __init__(self, ms, rankJ=0, rankP=1, rankTz=0, skew=False):
        self.model_space = ms
        self.rankJ = rankJ
        self.rankP = rankP
        self.rankTz = rankTz
        self.skew = skew
        self.ZeroBody = 0.0
        self.OneBody = np.zeros((ms.orbits.get_number_orbits(), ms.orbits.get_number_orbits()))
        self.TwoBody = {}
        self.normal_ordered = False
        for ichbra in range( ms.two_body_space.get_number_channels() ):
            chbra = ms.two_body_space.get_channel(ichbra)
            for ichket in range( ms.two_body_space.get_number_channels() ):
                chket = ms.two_body_space.get_channel(ichket)
                if(ichbra > ichket): continue
                if(abs(chbra.J-chket.J) > self.rankJ or chbra.J + chket.J < self.rankJ): continue
                if(chbra.Parity * chket.Parity * self.rankP != 1): continue
                if(abs(chbra.Tz - chket.Tz) != self.rankTz): continue
                self.TwoBody[(ichbra, ichket)] = np.zeros( (chbra.get_number_states(), chket.get_number_states()) )
        self.one_body_channel = {} 
        orbits = self.model_space.orbits
        for p in range(orbits.get_number_orbits()):
            o_p = orbits.get_orbit(p)
            key = (o_p.l, o_p.j2, o_p.tz2)
            if(key in self.one_body_channel): 
                self.one_body_channel[key].append(p)
            else:
                self.one_body_channel[key] = [p]
        return

    def __add__(self, other):
        op = Operator(self.model_space, self.rankJ, self.rankP, self.rankTz, self.skew)
        op.ZeroBody = self.ZeroBody + other.ZeroBody
        op.OneBody = self.OneBody + other.OneBody
        for channels in self.TwoBody.keys():
            ichbra, ichket = channels
            op.TwoBody[(ichbra, ichket)] = self.TwoBody[(ichbra, ichket)] + other.TwoBody[(ichbra, ichket)]
        return op

    def __mul__(self, scalar):
        op = Operator(self.model_space, self.rankJ, self.rankP, self.rankTz, self.skew)
        op.ZeroBody = self.ZeroBody * scalar
        op.OneBody = self.OneBody * scalar
        for channels in self.TwoBody.keys():
            ichbra, ichket = channels
            op.TwoBody[(ichbra, ichket)] = self.TwoBody[(ichbra, ichket)] * scalar
        return op

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __sub__(self, other):
        return self.__add__(other * (-1))

    def set_0bme(self, value):
        self.ZeroBody = value
        return

    def get_0bme(self):
        return self.ZeroBody

    def set_1bme(self, p, q, value):
        self.OneBody[p, q] = value
        return

    def get_1bme(self, p, q):
        return self.OneBody[p, q]

    def set_2bme_from_matrix_indices(self, idxbra, idxket, ibra, iket, value):
        self.TwoBody[(idxbra, idxket)][ibra, iket] = value
        if(idxbra==idxket):
            if(self.skew):
                self.TwoBody[(idxbra, idxket)][iket, ibra] = -value
            else:
                self.TwoBody[(idxbra, idxket)][iket, ibra] = value
        return

    def get_2bme_from_matrix_indices(self, idxbra, idxket, ibra, iket):
        return self.TwoBody[(idxbra, idxket)][ibra, iket]

    def set_2bme_from_indices(self, p, q, r, s, Jpq, Jrs, value):
        o_p = self.model_space.orbits.get_orbit(p)
        o_q = self.model_space.orbits.get_orbit(q)
        o_r = self.model_space.orbits.get_orbit(r)
        o_s = self.model_space.orbits.get_orbit(s)
        Pbra = (-1)**(o_p.l+o_q.l)
        Pket = (-1)**(o_r.l+o_s.l)
        Tzbra = (o_p.tz2 + o_q.tz2)//2
        Tzket = (o_r.tz2 + o_s.tz2)//2
        ichbra = self.model_space.two_body_space.get_index(Jpq, Pbra, Tzbra)
        ichket = self.model_space.two_body_space.get_index(Jrs, Pket, Tzket)
        if(ichbra > ichket):
            ichbra, ichket = ichket, ichbra
            p, q, r, s = r, s, p, q
            Jpq, Jrs = Jrs, Jpq
            Pbra, Pket = Pket, Pbra
            Tzbra, Tzket = Tzket, Tzbra
            if(self.skew): value *= -1
        chbra = self.model_space.two_body_space.get_channel(ichbra)
        chket = self.model_space.two_body_space.get_channel(ichket)
        idxbra = chbra.index_from_indices[(p,q)]
        idxket = chket.index_from_indices[(r,s)]
        value *= chbra.phase_from_indices[(p,q)] * chket.phase_from_indices[(r,s)]
        self.set_2bme_from_matrix_indices(ichbra, ichket, idxbra, idxket, value)

    def get_2bme_from_indices(self, p, q, r, s, Jpq, Jrs):
        o_p = self.model_space.orbits.get_orbit(p)
        o_q = self.model_space.orbits.get_orbit(q)
        o_r = self.model_space.orbits.get_orbit(r)
        o_s = self.model_space.orbits.get_orbit(s)
        Pbra = (-1)**(o_p.l+o_q.l)
        Pket = (-1)**(o_r.l+o_s.l)
        Tzbra = (o_p.tz2 + o_q.tz2)//2
        Tzket = (o_r.tz2 + o_s.tz2)//2
        ichbra = self.model_space.two_body_space.get_index(Jpq, Pbra, Tzbra)
        ichket = self.model_space.two_body_space.get_index(Jrs, Pket, Tzket)
        phase = 1
        if(ichbra > ichket):
            ichbra, ichket = ichket, ichbra
            p, q, r, s = r, s, p, q
            Jpq, Jrs = Jrs, Jpq
            Pbra, Pket = Pket, Pbra
            Tzbra, Tzket = Tzket, Tzbra
            if(self.skew): phase *= -1
        chbra = self.model_space.two_body_space.get_channel(ichbra)
        chket = self.model_space.two_body_space.get_channel(ichket)
        idxbra = chbra.index_from_indices[(p,q)]
        idxket = chket.index_from_indices[(r,s)]
        phase *= chbra.phase_from_indices[(p,q)] * chket.phase_from_indices[(r,s)]
        return self.get_2bme_from_matrix_indices(ichbra, ichket, idxbra, idxket) * phase

    def set_2bme_from_orbits(self, o_p, o_q, o_r, o_s, Jpq, Jrs, value):
        p = self.model_space.orbits.get_orbit_index_from_orbit(o_p)
        q = self.model_space.orbits.get_orbit_index_from_orbit(o_q)
        r = self.model_space.orbits.get_orbit_index_from_orbit(o_r)
        s = self.model_space.orbits.get_orbit_index_from_orbit(o_s)
        self.set_2bme_from_matrix_indices(p, q, r, s, Jpq, Jrs, value)

    def get_2bme_from_orbits(self, o_p, o_q, o_r, o_s, Jpq, Jrs):
        p = self.model_space.orbits.get_orbit_index_from_orbit(o_p)
        q = self.model_space.orbits.get_orbit_index_from_orbit(o_q)
        r = self.model_space.orbits.get_orbit_index_from_orbit(o_r)
        s = self.model_space.orbits.get_orbit_index_from_orbit(o_s)
        return self.get_2bme_from_indices(p, q, r, s, Jpq, Jrs)

    def set_kinetic_term(self):
        orbits = self.model_space.orbits
        def _red_nab_l(n1, l1, n2, l2):
            if(n1 == n2 and l1 == l2+1): me = -np.sqrt((l2 + 1)*(n2 + l2 + 1.5))
            elif(n1 == n2-1 and l1 == l2+1): me = -np.sqrt((l2 + 1)*n2)
            elif(n1 == n2 and l1 == l2-1): me = -np.sqrt(l2*(n2 + l2 + 0.5))
            elif(n1 == n2+1 and l1==l2-1): me = -np.sqrt(l2*(n2 + 1))
            else: me = 0
            return me

        def _red_nab_j(oa, ob):
            if(oa.tz2 != ob.tz2): return 0
            me = (-1.0)**((3+ob.j2)*0.5 + oa.l) * np.sqrt((oa.j2+1) * (ob.j2+1)) \
                    * _sixj(oa.j2*0.5, 1, ob.j2*0.5, ob.l, 0.5, oa.l) * _red_nab_l(oa.n, oa.l, ob.n, ob.l)
            return me

        def _me_NA(a, b, c, d, J):
            oa, ob, oc, od = orbits.get_orbit(a), orbits.get_orbit(b), orbits.get_orbit(c), orbits.get_orbit(d)
            me = -(-1.0)**((ob.j2+oc.j2)*0.5+J) * _sixj(oa.j2*0.5, ob.j2*0.5, J, od.j2*0.5, oc.j2*0.5, 1) * _red_nab_j(oa,oc) * _red_nab_j(ob,od)
            return me

        factor1 = self.model_space.get_frequency() * (1 - 1.0/self.model_space.A)
        for p in range(self.model_space.orbits.get_number_orbits()):
            for q in range(self.model_space.orbits.get_number_orbits()):
                o_p = orbits.get_orbit(p)
                o_q = orbits.get_orbit(q)
                if(o_p.l != o_q.l): continue
                if(o_p.j2 != o_q.j2): continue
                if(o_p.tz2 != o_q.tz2): continue
                if(abs(o_p.n - o_q.n) > 1): continue
                if(o_p.n == o_q.n): me = (2*o_p.n + o_p.l + 1.5) * 0.5
                elif(o_p.n == o_q.n-1): me = 0.5 * np.sqrt( (o_p.n+1) * (o_p.n+o_p.l+1.5) )
                elif(o_p.n == o_q.n+1): me = 0.5 * np.sqrt( (o_q.n+1) * (o_q.n+o_q.l+1.5) )
                self.set_1bme(p, q, me * factor1)

        factor2 = -self.model_space.get_frequency() / self.model_space.A
        for channels in self.TwoBody.keys():
             ichbra, ichket = channels
             chbra = self.model_space.two_body_space.get_channel(ichbra)
             chket = self.model_space.two_body_space.get_channel(ichket)
             for ibra in range(chbra.get_number_states()):
                 p, q = chbra.get_indices(ibra)
                 o_p = orbits.get_orbit(p)
                 o_q = orbits.get_orbit(q)
                 for iket in range(chket.get_number_states()):
                     r, s = chket.get_indices(iket)
                     o_r = orbits.get_orbit(r)
                     o_s = orbits.get_orbit(s)

                     norm = 1
                     if(p==q): norm /= np.sqrt(2)
                     if(r==s): norm /= np.sqrt(2)
                     me = _me_NA(p, q, r, s, chket.J) - (-1.0)**((o_r.j2+o_s.j2)*0.5 - chket.J) * _me_NA(p, q, s, r, chket.J)
                     me *= norm * factor2
                     self.set_2bme_from_indices(p, q, r, s, chbra.J, chket.J, me)
        return


    def set_nn_interaction(self, nnpot):

        def _tri(n1: int, na: int, nb: int) -> float:
            if n1 < 0 or na < 0 or nb < 0:
                return 0.0
            if n1 < na or n1 < nb:
                return 0.0
            if (n1 % 2) != (na % 2) or (n1 % 2) != (nb % 2):
                return 0.0
            denom = float(factorial2(na, exact=False) * factorial2(nb, exact=False))
            if denom == 0.0:
                return 0.0
            return float(factorial2(n1, exact=False) / denom)


        @functools.lru_cache(maxsize=None)
        def _cg_000(l1: int, l2: int, L: int) -> float:
            val = clebsch_gordan(sp.Integer(l1), sp.Integer(l2), sp.Integer(L),
                                sp.Integer(0), sp.Integer(0), sp.Integer(0))
            return float(sp.N(val))

        def _G(e1: int, l1: int, ea: int, la: int, eb: int, lb: int) -> float:
            """
            G(e1 l1; ea la, eb lb) in Eq.(27).  :contentReference[oaicite:2]{index=2}
            """
            if ea < 0 or eb < 0 or (ea + eb) != e1:
                return 0.0
            if la < 0 or lb < 0 or l1 < 0:
                return 0.0
            if la > ea or lb > eb or l1 > e1:
                return 0.0
            # parity constraints: e-l even
            if ((e1 - l1) % 2) or ((ea - la) % 2) or ((eb - lb) % 2):
                return 0.0

            cg = _cg_000(la, lb, l1)
            if cg == 0.0:
                return 0.0

            t1 = _tri(e1 - l1, ea - la, eb - lb)
            if t1 == 0.0:
                return 0.0
            t2 = _tri(e1 + l1 + 1, ea + la + 1, eb + lb + 1)
            if t2 == 0.0:
                return 0.0

            return math.sqrt((2 * la + 1) * (2 * lb + 1)) * cg * math.sqrt(t1 * t2)


        def _ho_bracket(Ncm, Lcm, nrel, lrel, N1, L1, N2, L2, L12, d=1.0) -> float:
            if d < 0:
                raise ValueError("d must be nonnegative")

            # map to paper variables
            E = 2 * Ncm + Lcm
            L = Lcm
            e = 2 * nrel + lrel
            l = lrel
            e1 = 2 * N1 + L1
            l1 = L1
            e2 = 2 * N2 + L2
            l2 = L2
            Lam = L12

            # selection: energy conservation Eq.(5)
            if (e1 + e2) != (E + e):
                return 0.0
            # parity Eq.(6)
            if ((l1 + l2 - L - l) % 2) != 0:
                return 0.0

            pow1 = (e1 - e) / 2
            pow2 = (e1 + e2) / 2
            if d == 0.0 and pow1 < 0:
                return 0.0
            pref = (d ** pow1) * ((1.0 + d) ** (-pow2))

            total = 0.0
            for ed in range(0, min(e, e2) + 1):
                eb = e - ed
                ec = e2 - ed
                ea = e1 - e + ed  # from ea+ec=E

                ded = ((-d) ** ed)

                for ld in range(ed, -1, -2):
                    for lb in range(eb, -1, -2):
                        if(abs(ld - lb) > l or ld + lb < l): continue

                        for lc in range(ec, -1, -2):
                            if(abs(ld - lc) > l2 or ld + lc < l2): continue
                            for la in range(ea, -1, -2):
                                if(abs(la - lc) > L or la + lc < L): continue
                                if(abs(la - lb) > l1 or la + lb < l1): continue

                                total += ded * _ninej(la, lb, l1, lc, ld, l2, L, l, Lam) * \
                                    _G(e1, l1, ea, la, eb, lb) * \
                                    _G(e2, l2, ec, lc, ed, ld) * \
                                    _G(E, L, ea, la, ec, lc) * \
                                    _G(e, l, eb, lb, ed, ld)
            return pref * total * (-1)**(Ncm+nrel+N1+N2)

        trans_coefs = {} # < Ncm, Lcm, nrel, lrel, Spq, Jrel, Jpq | p, q, Jpq >
        for channel in self.model_space.two_body_space.channels:
            Jpq = channel.J
            Ppq = channel.Parity
            Tzpq = channel.Tz
            trans_coefs[(Jpq, Ppq, Tzpq)] = {}
            for idx in range(channel.get_number_states()):
                p, q = channel.get_indices(idx)
                o_p = self.model_space.orbits.get_orbit(p)
                o_q = self.model_space.orbits.get_orbit(q)
                N = 2*o_p.n + o_p.l + 2*o_q.n + o_q.l
                trans_coefs[(Jpq, Ppq, Tzpq)][(p, q)] = {}

                for Spq in [0, 1]:
                    for lrel in range(N+1):
                        fact = 1.0
                        if(o_p.tz2 == o_q.tz2):
                            fact = (1 + (-1)**(lrel+Spq)) / np.sqrt(2)
                            if(p==q): fact /= np.sqrt(2)
                        if(abs(fact)<= 1.e-8): continue
                        for Jrel in range(abs(lrel-Spq), min(nnpot.HOPWChannels.Jmax, lrel+Spq)+1):
                            for Lcm in range(N+1-lrel):
                                for Ncm in range((N-Lcm-lrel)//2+1):
                                    nrel = (N - Lcm - 2*Ncm - lrel) // 2
                                    if((-1)**(lrel + Lcm) != Ppq): continue
                                    me = 0.0
                                    for Lpq in range(max(abs(Jpq-Spq), abs(Lcm-lrel), abs(o_p.l-o_q.l)), min(Jpq+Spq, Lcm+lrel, o_p.l+o_q.l)+1):
                                        me += (2*Lpq+1) * _ninej(o_p.l, o_q.l, Lpq, 0.5, 0.5, Spq, o_p.j2*0.5, o_q.j2*0.5, Jpq) * \
                                            _sixj(Lcm, lrel, Lpq, Spq, Jpq, Jrel) * _ho_bracket(Ncm, Lcm, nrel, lrel, o_p.n, o_p.l, o_q.n, o_q.l, Lpq)
                                    me *= (-1)**(Lcm + lrel + Spq + Jpq) * np.sqrt((o_p.j2+1)*(o_q.j2+1)*(2*Spq+1)*(2*Jrel+1)) * fact
                                    if(abs(me) < 1.e-8): continue
                                    trans_coefs[(Jpq, Ppq, Tzpq)][(p, q)][(Ncm, Lcm, nrel, lrel, Spq, Jrel)] = me

        for channels in self.TwoBody.keys():
             ichbra, ichket = channels
             chbra = self.model_space.two_body_space.get_channel(ichbra)
             chket = self.model_space.two_body_space.get_channel(ichket)
             for ibra in range(chbra.get_number_states()):
                 p, q = chbra.get_indices(ibra)
                 o_p = self.model_space.orbits.get_orbit(p)
                 o_q = self.model_space.orbits.get_orbit(q)
                 for iket in range(chket.get_number_states()):
                     r, s = chket.get_indices(iket)
                     o_r = self.model_space.orbits.get_orbit(r)
                     o_s = self.model_space.orbits.get_orbit(s)

                     coefs_bra = trans_coefs[(chbra.J, chbra.Parity, chbra.Tz)][(p, q)]
                     coefs_ket = trans_coefs[(chket.J, chket.Parity, chket.Tz)][(r, s)]

                     me = 0.0
                     for key_bra in coefs_bra.keys():
                         for key_ket in coefs_ket.keys():
                             Ncm_bra, Lcm_bra, nrel_bra, lrel_bra, Spq_bra, Jrel_bra = key_bra
                             Ncm_ket, Lcm_ket, nrel_ket, lrel_ket, Spq_ket, Jrel_ket = key_ket
                             if(Ncm_bra != Ncm_ket): continue
                             if(Lcm_bra != Lcm_ket): continue
                             if(Spq_bra != Spq_ket): continue
                             if(Jrel_bra != Jrel_ket): continue
                             if((lrel_bra + lrel_ket)%2 == 1): continue
                             if(abs(lrel_bra - lrel_ket) > 2): continue
                             me += coefs_bra[key_bra] * \
                                coefs_ket[key_ket] * \
                                nnpot.get_me(nrel_bra, lrel_bra, nrel_ket, lrel_ket, Spq_bra, Jrel_bra, chket.Tz)
                     self.set_2bme_from_indices(p, q, r, s, chbra.J, chket.J, me)

    def take_normal_ordering(self, undo=False):
        """
        Normal order the operator with respect to the reference state.
        only scalar operator
        """
        if(undo and not self.normal_ordered):
            print("Warning: the operator is not normal ordered. Nothing to undo.")
            return
        if(not undo and self.normal_ordered):
            print("Warning: the operator is already normal ordered. Nothing to do.")
            return
        tmp_2to1 = np.zeros_like(self.OneBody)
        for p in range(self.model_space.orbits.get_number_orbits()):
            o_p = self.model_space.orbits.get_orbit(p)
            for q in range(self.model_space.orbits.get_number_orbits()):
                o_q = self.model_space.orbits.get_orbit(q)
                if(o_p.l != o_q.l): continue
                if(o_p.j2 != o_q.j2): continue
                if(o_p.tz2 != o_q.tz2): continue
                me = 0.0
                for r in range(self.model_space.orbits.get_number_orbits()):
                    o_r = self.model_space.orbits.get_orbit(r)
                    occ = self.model_space.occupations[r]
                    if(occ < 1.e-4): continue
                    norm = 1.0
                    if(p==r): norm *= np.sqrt(2)
                    if(q==r): norm *= np.sqrt(2)
                    for J in range(abs(o_p.j2-o_r.j2)//2, (o_p.j2+o_r.j2)//2+1):
                        if(p==r and J%2==1): continue
                        if(q==r and J%2==1): continue
                        me += norm * occ * self.get_2bme_from_indices(p, r, q, r, J, J) * (2*J+1)
                tmp_2to1[p, q] = me / (o_p.j2 + 1)
        tmp_2to0, tmp_1to0 = 0.0, 0.0
        for p in range(self.model_space.orbits.get_number_orbits()):
            o_p = self.model_space.orbits.get_orbit(p)
            occ = self.model_space.occupations[p]
            tmp_1to0 += self.get_1bme(p, p) * occ * (o_p.j2+1)
            tmp_2to0 += tmp_2to1[p, p] * occ * (o_p.j2+1) * 0.5
        #print(self.ZeroBody, tmp_1to0, tmp_2to0)
        if(undo):
            self.ZeroBody -= tmp_1to0 - tmp_2to0
            self.OneBody -= tmp_2to1 
            self.normal_ordered = False
        else:
            self.ZeroBody += tmp_1to0 + tmp_2to0
            self.OneBody += tmp_2to1 
            self.normal_ordered = True

    def transformation(self, coefs):
        """
        Transform the operator with the given coefficients.
        only scalar operator
        coefs: < HO | p >
        """
        op = Operator(self.model_space, self.rankJ, self.rankP, self.rankTz, self.skew)
        if(self.normal_ordered): self.take_normal_ordering(undo=True)
        op.OneBody = coefs.T @ self.OneBody @ coefs
        for channels in self.TwoBody.keys():
            ichbra, ichket = channels
            chan = self.model_space.two_body_space.get_channel(ichket)
            mat = self.TwoBody[(ichbra, ichket)]
            U = np.zeros((chan.get_number_states(), chan.get_number_states()), dtype=np.float64)
            for ibra in range(chan.get_number_states()):
                p, q = chan.get_indices(ibra)
                o_p = self.model_space.orbits.get_orbit(p)
                o_q = self.model_space.orbits.get_orbit(q)
                for iket in range(chan.get_number_states()):
                    r, s = chan.get_indices(iket)
                    U[ibra, iket] = coefs[p, r] * coefs[q, s]
                    if(p!=q): U[ibra, iket] -= coefs[p, s] * coefs[q, r] * (-1)**((o_p.j2+o_q.j2)//2 - chan.J)
                    if(p==q): U[ibra, iket] *= np.sqrt(2)
                    if(r==s): U[ibra, iket] /= np.sqrt(2)
            op.TwoBody[(ichbra, ichket)] = U.T @ mat @ U
        return op
    
    def write_snt_file(self, filename):
        is_scalar = (self.rankJ == 0 and self.rankP == 1 and self.rankTz == 0)
        orbits = self.model_space.orbits
        p_norbs = 0; n_norbs = 0
        for o in orbits.orbits:
            if( o.tz2 ==-1 ): p_norbs += 1 
            if( o.tz2 == 1 ): n_norbs += 1 
        prt = "" 
        prt += "! J:{:3d} P:{:3d} Tz:{:3d}\n".format( self.rankJ, self.rankP, self.rankTz )
        prt += "! Zero body term: {:16.8e}\n".format(self.get_0bme())
        prt += "! model space \n"
        prt += " {0:3d} {1:3d} {2:3d} {3:3d} \n".format(p_norbs, n_norbs, 0, 0)
        for i in range(orbits.get_number_orbits()):
            o = orbits.get_orbit(i)
            prt += "{0:5d} {1:3d} {2:3d} {3:3d} {4:3d} \n".format( i+1, o.n, o.l, o.j2, o.tz2 )

        prt += "! one-body part\n"

        tmp, cnt = "", 0
        for p in range(orbits.get_number_orbits()):
            for q in range(orbits.get_number_orbits()):
                if(p>q): continue
                o_p = orbits.get_orbit(p)
                o_q = orbits.get_orbit(q)
                if(abs(self.get_1bme(p,q)) < 1.e-10): continue
                cnt += 1
                tmp += f"{p+1:3d} {q+1:3d} {self.get_1bme(p,q):16.8e}\n"
        prt += f"{cnt:5d} {0:3d}\n"
        prt += "! two-body part\n"

        tmp, cnt = "", 0
        for channels in self.TwoBody.keys():
            ichbra, ichket = channels
            chbra = self.model_space.two_body_space.get_channel(ichbra)
            chket = self.model_space.two_body_space.get_channel(ichket)
            mat = self.TwoBody[channels]
            for ibra in range(chbra.get_number_states()):
                for iket in range(chket.get_number_states()):
                    if(ichbra==ichket and iket<ibra): continue
                    if(abs(mat[ibra, iket]) < 1.e-10): continue
                    a, b = chbra.get_indices(ibra)
                    c, d = chket.get_indices(iket)
                    if(is_scalar):
                        tmp += f"{a+1:3d} {b+1:3d} {c+1:3d} {d+1:3d} {chket.J:3d} {mat[ibra, iket]:16.8e}\n"
                    else:
                        tmp += f"{a+1:3d} {b+1:3d} {c+1:3d} {d+1:3d} {chbra.J:3d} {chket.J:3d} {mat[ibra, iket]:16.8e}\n"
                    cnt += 1
        prt += f"{cnt:10d} {0:3d}\n"
        prt += tmp
        f = open(filename, "w") 
        f.write(prt)
        f.close()


    def __str__(self):
        orbits = self.model_space.get_orbits()
        string = self.model_space.__str__()
        string += f"Operator with rankJ={self.rankJ:2d}, rankP={self.rankP:2d}, rankTz={self.rankTz:2d}, skew={self.skew}\n"
        string += f"Zero-body matrix element: {self.ZeroBody:12.6f}\n"
        string += f"One-body matrix elements:\n"
        for p in range(self.model_space.orbits.get_number_orbits()):
            for q in range(self.model_space.orbits.get_number_orbits()):
                me = self.OneBody[p, q]
                if(abs(me) > 1.e-4):
                    string += f"  <{orbits.get_orbit_label(p):>8s}|O|{orbits.get_orbit_label(q):>8s}> = {me:12.6f}\n"
        string += f"Two-body matrix elements:\n"
        for channels in self.TwoBody.keys():
             ichbra, ichket = channels
             chbra = self.model_space.two_body_space.get_channel(ichbra)
             chket = self.model_space.two_body_space.get_channel(ichket)
             for ibra in range(chbra.get_number_states()):
                 p, q = chbra.get_indices(ibra)
                 o_p = self.model_space.orbits.get_orbit(p)
                 o_q = self.model_space.orbits.get_orbit(q)
                 for iket in range(chket.get_number_states()):
                     r, s = chket.get_indices(iket)
                     o_r = self.model_space.orbits.get_orbit(r)
                     o_s = self.model_space.orbits.get_orbit(s)

                     me = self.get_2bme_from_matrix_indices(ichbra, ichket, ibra, iket)
                     if(abs(me) > 1.e-4):
                         string += f"  <{self.model_space.orbits.get_orbit_label(p):>8s},{self.model_space.orbits.get_orbit_label(q):>8s};J={chbra.J:3d}|O|{self.model_space.orbits.get_orbit_label(r):>8s},{self.model_space.orbits.get_orbit_label(s):>8s};J={chket.J:3d}> = {me:12.6f}\n"
        return string

def main():

    import LECs
    emax = 2
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
    vnn.write_snt_file("vnn.snt")
    ham.take_normal_ordering()
    ham.take_normal_ordering(undo=True)

if(__name__=="__main__"):
    main()
