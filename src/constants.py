#!/usr/bin/env python3
from scipy.constants import physical_constants

hc = physical_constants['reduced Planck constant times c in MeV fm'][0]
m_n = physical_constants['neutron mass energy equivalent in MeV'][0]
m_p = physical_constants['proton mass energy equivalent in MeV'][0]
alpha = physical_constants['fine-structure constant'][0]
#m_p = 938.272046
#m_n = 939.565379
m_nucl = (m_p + m_n)*0.5
m_delta = 1232.0
hA = 1.40
m_pi0 = 134.9768
m_pic = 139.57039
m_pi = (m_pi0 + 2*m_pic) / 3
f_pi = 92.4
gA = 1.27