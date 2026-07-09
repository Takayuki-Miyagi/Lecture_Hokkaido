#!/usr/bin/env python3
# python3 phase-shift.py MOMENTUM_SPACE_FILE
import os, sys
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from matplotlib import colors
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pathlib import Path
current_dir = str(Path(__file__).parent.absolute())
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
import constants
from partial_wave_basis import PartialWaveChannel, PartialWaveChannels
from chiral_eft import ChEFTNNPotential
from nn_pot import NNPotential

parameters = {
    "Power":3,
    "Lambda":500,
    "ch_order": 2,
    "NMesh": 40,
    "Jmax": 6,
    "Ct_1S0_pp":-0.338223,
    "Ct_1S0_pn":-0.338320,
    "Ct_1S0_nn":-0.337303,
    "Ct_3S1":-0.221721,
    "C_1S0":2.488019,
    "C_3S1":0.675353,
    "C_1P1":-0.012651,
    "C_3P0":0.698454,
    "C_3P1":-0.937264,
    "C_3P2":-0.859526,
    "C_3S1_3D1":0.354479,
    "c1": -0.74,
    "c2": -0.49,
    "c3": -0.65,
    "c4":  0.96
     }

parameters = {
    "Power":4,
    "Lambda":394,
    "ch_order": 2,
    "NMesh": 40,
    "Jmax": 6,
    "Ct_1S0_pp":-0.338142,
    "Ct_1S0_pn":-0.339250,
    "Ct_1S0_nn":-0.338746,
    "Ct_3S1":-0.259839,
    "C_1S0":2.505389,
    "C_3S1":1.002189,
    "C_1P1":-0.387960,
    "C_3P0":0.700499,
    "C_3P1":-0.964856,
    "C_3P2":-0.883122,
    "C_3S1_3D1":0.452523,
    "c1": -0.74,
    "c2": -0.49,
    "c3": -0.65,
    "c4":  0.96
     }
filename = "deltago394.npz"
def VtoVdict(V, Mesh, Llist):
    NMesh = []
    for i, L in enumerate(Llist):
        NMesh.append(Mesh.size)
    i_start = []
    i_end = []
    for idx, n in enumerate(NMesh):
        i_start.append(sum(NMesh[:idx]))
        i_end.append(sum(NMesh[:idx+1]))
    v = {}
    for ib, Lb in enumerate(Llist):
        for ik, Lk in enumerate(Llist):
            v[(Lb,Lk)] = V[i_start[ib]:i_end[ib], i_start[ik]:i_end[ik]]
    return v

def VdicttoV(Vdict, Mesh, Llist):
    NMesh = []
    for i, L in enumerate(Llist):
        NMesh.append(Mesh.size)
    i_start = []
    i_end = []
    for idx, n in enumerate(NMesh):
        i_start.append(sum(NMesh[:idx]))
        i_end.append(sum(NMesh[:idx+1]))
    V = np.zeros((sum(NMesh), sum(NMesh)), dtype=complex)
    for ib, Lb in enumerate(Llist):
        for ik, Lk in enumerate(Llist):
            V[i_start[ib]:i_end[ib], i_start[ik]:i_end[ik]] = Vdict[(Lb,Lk)]
    return V

def interpolate(Mesh, Mesh_int, Llist, V):
    """
    in unit of MeV fm3
    """
    Vdict = VtoVdict(V, Mesh, Llist)
    Vdict_int = {}
    for key in Vdict.keys():
        Lb, Lk = key
        Vdict_int[(Lb, Lk)] = interpolate_block(Mesh, Mesh_int, Mesh_int, Vdict[key])
    Vint = VdicttoV(Vdict_int, Mesh_int, Llist)
    Vint = 0.5 * (Vint + Vint.T)
    return Vint


def interpolate_block(Mesh, Mesh_int_l, Mesh_int_r, V):
    Meshes = np.reshape(np.tile(Mesh, Mesh.size), (Mesh.size, Mesh.size))
    x = Meshes.reshape(Mesh.size * Mesh.size)
    y = np.transpose(Meshes).reshape(Mesh.size * Mesh.size)
    X, Y = Mesh_int_r, Mesh_int_l
    Z=griddata((x,y), V.reshape(Mesh.size * Mesh.size), (X[None,:], Y[:,None]), method='cubic')
    return Z

def propagator(Mesh, Weight, q, mu, Llist):
    """
    in unit of MeV^2
    """
    Gdict = {}
    for ib, Lb in enumerate(Llist):
        for ik, Lk in enumerate(Llist):
            Gdict[(Lb,Lk)] = np.zeros((Mesh.size,Mesh.size), dtype=complex)

    for key in Gdict.keys():
        Lb, Lk = key
        if(Lb != Lk): continue
        Gdict[(Lb,Lk)] = propagator_block(Mesh, Weight, q, mu)
    G0 = VdicttoV(Gdict, Mesh, Llist)
    return G0

def propagator_block(Mesh, Weight, q, mu):
    G0 = np.zeros((Mesh.size,Mesh.size), dtype=complex)
    if(q**2 >= 0):
        idx = np.argmin(np.abs(Mesh - q))
        for i in range(len(Mesh)):
            if(i==idx): continue
            G0[i,i] = 2 * mu * Weight[i] * constants.hc * Mesh[i]**2 / (q**2 - Mesh[i]**2)
            G0[idx,idx] -=  2 * mu * Weight[i] * constants.hc * q**2 / (q**2 - Mesh[i]**2)
        G0[idx,idx] -= 1.j * np.pi * mu * q * constants.hc
    else:
        for i in range(len(Mesh)):
            G0[i,i] = 2 * mu * Weight[i] * constants.hc * Mesh[i]**2 / (q**2 - Mesh[i]**2)
    return G0

def Tmatrix(V, G0):
    T = np.linalg.solve(np.eye(V.shape[0]) - V@G0, V)
    return T

def on_shell_Tmatrix(Tmat, Mesh, Mesh_int, idx, Llist):
    pos_extract = []
    qn_extract = []
    cnt, pos = 0, 0
    for i, L in enumerate(Llist):
        pos_extract.append(cnt + idx)
        qn_extract.append(L)
        cnt += Mesh_int.size
    T = Tmat[np.ix_(pos_extract, pos_extract)]
    return qn_extract, T

def on_shell_Smatrix(Tmat, qn, q, mu):
    T = np.zeros(Tmat.shape, dtype=complex)
    for i, ib in enumerate(qn):
        for j, ik in enumerate(qn):
            T[i,j] = Tmat[i,j]*q*constants.hc*mu
    S = np.eye(T.shape[0]) - 2 * np.pi * 1.j * T

    # make it unitary
    #invT = np.linalg.inv(T)
    #H = 0.5*(invT + invT.conj().T)
    #invT_unit = H + 1j*np.pi*np.eye(T.shape[0])
    #T_unit = np.linalg.inv(invT_unit)
    #S_unit = np.eye(T.shape[0]) - 2j*np.pi*T_unit
    #return S_unit
    return S

def compute_S_matrix_nn(V, q, mu, channel, phase=True):

    Mesh = channel.kmesh[:channel.NMesh]
    Weight = channel.weights[:channel.NMesh]
    Mesh_int = np.sort(np.concatenate((Mesh, np.array([q]))))
    idx = np.argmin(np.abs(Mesh_int - q))
    Weight_int = np.insert(Weight, idx, 0)

    G0 = propagator(Mesh_int, Weight_int, q, mu, channel.Llist)
    Vint = interpolate(Mesh, Mesh_int, channel.Llist, V)
    Tmat = Tmatrix(Vint, G0)

    qns, T = on_shell_Tmatrix(Tmat, Mesh, Mesh_int, idx, channel.Llist)
    Smat = on_shell_Smatrix(T, qns, q, mu)

    err = np.linalg.norm(Smat.conj().T @ Smat - np.eye(Smat.shape[0])) # unitarity check
    return Smat

def cross_section_from_S(Smat, q0, J):
    I = np.eye(Smat.shape[0])
    trace = np.trace((I - Smat) @ (I - Smat.conj().T))
    sigma = (np.pi / q0**2) * trace.real * (2*J+1) * 0.25
    sigma *= 1e-2 # fm2 -> barn
    return sigma

def main():
    pw = PartialWaveChannels(Jmax=4,NMesh=80,kmax=6)
    nnpot = NNPotential(pw)
    if(os.path.exists(filename)):
        nnpot.load_npz_into(filename)
    else:
        nnpot.set_potential(parameters)
        nnpot.save_npz(filename)
    Smatrices = {}
    Elabs = [1, 5, 10] + list(range(20,100,10)) + list(range(100, 350, 50))
    mu = [0.5 * constants.m_p, constants.m_p * constants.m_n / (constants.m_p + constants.m_n), 0.5 * constants.m_n]
    for channel in pw.Channels:
        J, Prty, S, Tz = channel.J, channel.Parity, channel.S, channel.Tz
        print(f"Doing for J={J:2d}, Parity={Prty:2d}, S={S:2d}, Tz={Tz:2d}")
        for Elab in Elabs:
            if(Tz==-1): q0 = np.sqrt(2.0 * mu[Tz+1]**2 / constants.m_p * Elab) / constants.hc
            if(Tz== 0): q0 = np.sqrt(2.0 * mu[Tz+1]**2 / constants.m_p * Elab) / constants.hc # proton incident beam
            if(Tz== 1): q0 = np.sqrt(2.0 * mu[Tz+1]**2 / constants.m_n * Elab) / constants.hc
            Smatrices[(J,Prty,S,Tz,Elab)] = compute_S_matrix_nn(nnpot.vpot[(J,Prty,S,Tz)], q0, mu[Tz+1], channel)
    
    # scattering length
    from scipy.stats import linregress
    for channel in pw.Channels:
        J, Prty, S, Tz = channel.J, channel.Parity, channel.S, channel.Tz
        label = ""
        if(J==0 and Prty==1 and S==0 and Tz==-1): label="pp a_s and r_s"
        if(J==0 and Prty==1 and S==0 and Tz== 0): label="pn a_s and r_s"
        if(J==0 and Prty==1 and S==0 and Tz== 1): label="nn a_s and r_s"
        if(J==1 and Prty==1 and S==1 and Tz== 0): label="pn a_t and r_t"
        if(label==""): continue
        k2, kcot = [], []
        for Elab in Elabs[:4]:
            if(Tz==-1): q0 = np.sqrt(2.0 * mu[Tz+1]**2 / constants.m_p * Elab) / constants.hc
            if(Tz== 0): q0 = np.sqrt(2.0 * mu[Tz+1]**2 / constants.m_p * Elab) / constants.hc # proton incident beam
            if(Tz== 1): q0 = np.sqrt(2.0 * mu[Tz+1]**2 / constants.m_n * Elab) / constants.hc
            Smat = Smatrices[(J,Prty,S,Tz,Elab)]
            delta = 0.5 * np.angle(Smat[0,0])
            k2.append(q0**2)
            kcot.append((q0/np.tan(delta)))
        k2, kcot = np.array(k2), np.array(kcot)
        res = linregress(k2, kcot)
        c0, c1 = res.intercept, res.slope
        a, r = -1/c0, 2*c1
        print(f"{label}: {a:7.4f} fm and {r:7.4f} fm")
    print()

    # phase shifts
    for channel in pw.Channels:
        J, Prty, S, Tz = channel.J, channel.Parity, channel.S, channel.Tz
        for Elab in Elabs:
            Smat = Smatrices[(J,Prty,S,Tz,Elab)]
            if(Smat.shape[0]==1):
                delta = 0.5 * np.angle(Smat[0,0])
                print(f"J={J:2d}, Prty={Prty:2d}, S={S:2d}, Tz={Tz:2d}, Elab={Elab:6.2f} MeV, delta={delta*180/np.pi:8.4f}")
            else:
                Sigma = 0.5 * np.angle(np.linalg.det(Smat))
                phase = np.exp(-1j * Sigma)
                sin2eps_complex = -1j * Smat[0,1] * phase
                sin2eps = np.real(sin2eps_complex)   
                eps = 0.5 * np.arcsin(sin2eps)
                delta_m = 0.5 * np.angle(Smat[0,0]/np.cos(2*eps))
                delta_p = 0.5 * np.angle(Smat[1,1]/np.cos(2*eps))
                print(f"J={J:2d}, Prty={Prty:2d}, S={S:2d}, Tz={Tz:2d}, Elab={Elab:6.2f} MeV, delta-={delta_m*180/np.pi:8.4f}, delta+={delta_p*180/np.pi:8.4f}, eps={eps*180/np.pi:8.4f}")
        print()

    # cross section
    for channel_Tz in [-1,0,1]:
        sigma = []
        for channel in pw.Channels:
            J, Prty, S, Tz = channel.J, channel.Parity, channel.S, channel.Tz
            if(channel_Tz != Tz): continue
            for Elab in Elabs:
                if(Tz==-1): q0 = np.sqrt(2.0 * mu[Tz+1]**2 / constants.m_p * Elab) / constants.hc
                if(Tz== 0): q0 = np.sqrt(2.0 * mu[Tz+1]**2 / constants.m_p * Elab) / constants.hc # proton incident beam
                if(Tz== 1): q0 = np.sqrt(2.0 * mu[Tz+1]**2 / constants.m_n * Elab) / constants.hc
                _sigma = cross_section_from_S(Smatrices[(J,Prty,S,Tz,Elab)], q0, J)
                sigma.append(_sigma)

        sigma = np.array(sigma)
        sigma = sigma.reshape((int(sigma.size)//len(Elabs), len(Elabs))).transpose()
        sigma_tot = np.sum(sigma, axis=1)
        if(sigma.size==0): continue

        print()
        if(channel_Tz==-1): print("pp channel:")
        if(channel_Tz== 0): print("pn channel:")
        if(channel_Tz== 1): print("nn channel:")
        print(f"{'Plab (MeV)':>12s}, {'sigma (b)':>10s}")
        for i in range(len(Elabs)):
            Elab = Elabs[i]
            print(f"{Elab:12.3f}, {sigma_tot[i]:10.6f}")

#def propagator_block_pv(Mesh, Weight, q, mu):
#    def I0_pv(q0, Lam):
#        return -Lam + 0.5*q0*np.log(abs((Lam + q0)/(Lam - q0)))
#    G0 = np.zeros((Mesh.size, Mesh.size), dtype=complex)
#    if(q**2 >= 0):
#        q0 = q.real
#        idx = np.argmin(np.abs(Mesh - q0))
#        Lam = Mesh.max()
#        for i in range(len(Mesh)):
#            if i == idx:
#                continue
#            G0[i,i] = 2 * mu * constants.hc * Weight[i] * Mesh[i]**2 / (q0**2 - Mesh[i]**2)
#        Ssum = 0.0
#        for i in range(len(Mesh)):
#            if i == idx: continue
#            Ssum += Weight[i] * Mesh[i]**2 / (q0**2 - Mesh[i]**2)
#        B = I0_pv(q0, Lam) - Ssum
#        G0[idx,idx] = 2 * mu * constants.hc * B
#        G0[idx,idx] -= 1j * np.pi * mu * q0 * constants.hc
#    else:
#        for i in range(len(Mesh)):
#            G0[i,i] = 2 * mu * constants.hc * Weight[i] * Mesh[i]**2 / (q**2 - Mesh[i]**2)
#    return G0

#def compute_S_matrix_from_K(V, q, mu, channel):
#    """Compute S by computing the K-matrix with principal-value G and
#    converting K -> T using the same on-shell scaling used for T.
#    """
#    Mesh = channel.kmesh[:channel.NMesh]
#    Weight = channel.weights[:channel.NMesh]
#    Mesh_int = np.sort(np.concatenate((Mesh, np.array([q]))))
#    idx = np.argmin(np.abs(Mesh_int - q))
#    Weight_int = np.insert(Weight, idx, 0)
#
#    # principal-value propagator (no on-shell imaginary part)
#    # Build a principal-value G0 (no on-shell imaginary part) by using
#    # the real-valued `propagator_block` for diagonal blocks.
#    Gdict = {}
#    for ib, Lb in enumerate(channel.Llist):
#        for ik, Lk in enumerate(channel.Llist):
#            Gdict[(Lb, Lk)] = np.zeros((Mesh_int.size, Mesh_int.size), dtype=complex)
#    for key in list(Gdict.keys()):
#        Lb, Lk = key
#        if Lb != Lk:
#            continue
#        # use the real principal-value block (no -1j*pi term)
#        Gdict[(Lb, Lk)] = propagator_block_for_K(Mesh_int, Weight_int, q, mu).astype(complex)
#    G0_pv = VdicttoV(Gdict, Mesh_int, channel.Llist)
#    Vint = interpolate(Mesh, Mesh_int, channel.Llist, V)
#    Kmat = Kmatrix(Vint, G0_pv)
#    qns, K_on = on_shell_Tmatrix(Kmat, Mesh, Mesh_int, idx, channel.Llist)
#    A = (-np.pi * q * constants.hc * mu)
#    K_scaled = K_on * A
#    I = np.eye(K_scaled.shape[0], dtype=complex)
#    S = (I + 1j * K_scaled) @ np.linalg.inv(I - 1j * K_scaled)
#    return S
#
#def Kmatrix(V, G0_pv):
#    """Compute K matrix using the principal-value propagator (real G)."""
#    K = np.linalg.solve(np.eye(V.shape[0]) - V@G0_pv, V)
#    return K
#
#def propagator_block_for_K(Mesh, Weight, q, mu):
#    G0 = np.zeros((Mesh.size,Mesh.size), dtype=np.float64)
#    idx = np.argmin(np.abs(Mesh - q))
#    for i in range(len(Mesh)):
#        if(i==idx): continue
#        G0[i,i] = 2 * mu * Weight[i] * constants.hc * Mesh[i]**2 / (q**2 - Mesh[i]**2)
#        G0[idx,idx] -=  2 * mu * Weight[i] * constants.hc * q**2 / (q**2 - Mesh[i]**2)
#    return G0

if(__name__=='__main__'):
    main()
