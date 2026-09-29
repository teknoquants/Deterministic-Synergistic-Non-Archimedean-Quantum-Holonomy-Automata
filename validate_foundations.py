# MIT License
#
# Copyright (c) September 2026 Roger Cooperman; A.E. Petrunina V.V.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this
# software and associated documentation files (the “Software”), to deal in the Software
# without restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
# OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.
#
"""Sections II-V: Haar identities, gauge action, geometry and hardware arithmetic."""
from fractions import Fraction
import numpy as np
from common import *


def run(c,rng,n):
    u=haar_su3(rng,n)
    c.close('F01','SU(3) samples: unitarity',u.conj().transpose(0,2,1)@u,np.eye(3))
    c.close('F02','SU(3) samples: determinant one',np.linalg.det(u),1)
    c.check('F03','Normalized Haar class integral',haar_class(ONE)==1,{'integral':str(haar_class(ONE))},kind='EXACT')
    c.mc('F04','Fundamental first moment is zero',u,0)
    # All 81 moments: U_ij conjugate(U_kl); paper's dagger indices are transposed.
    flat=u.reshape(n,9)
    second=flat[:,:,None]*flat[:,None,:].conj()
    c.mc('F05','All 81 fundamental second moments',second,np.eye(9)/3)
    # All 729 third moments, accumulate means/variances without N*729 storage.
    mean=np.einsum('ni,nj,nk->ijk',flat,flat,flat)/n
    a=np.abs(flat)**2
    var=np.maximum(np.einsum('ni,nj,nk->ijk',a,a,a)/n-np.abs(mean)**2,0)
    e=epsilon()
    target=np.einsum('abc,def->adbecf',e,e).reshape(9,9,9)/6
    se=np.sqrt(var/(n-1)); err=np.abs(mean-target)
    c.check('F06','All 729 SU(3) epsilon third moments',np.all(err<=7*se+1e-12),
            {'max_error':float(err.max()),'max_standard_errors':float(np.max(err/(se+1e-12))),'samples':n},kind='MONTE_CARLO')
    # Exact class counterparts independently validate the Weyl integration machinery.
    vals=[haar_class(CHI3),haar_class(mul(CHI3,CHIBAR)),haar_class(mul(mul(CHI3,CHI3),CHI3))]
    c.check('F07','Exact character moments: 0, 1, 1',vals==[0,1,1],{'moments':list(map(str,vals))},kind='EXACT')
    gs=haar_su3(rng,3); links=haar_su3(rng,2)
    v0=gs[0]@links[0]@gs[1].conj().T
    v1=gs[1]@links[1]@gs[2].conj().T
    c.close('F08','Path holonomy transforms only at endpoints',v0@v1,gs[0]@links[0]@links[1]@gs[2].conj().T)
    c.close('F09','Reverse edge is inverse, not an independent link',links[0]@links[0].conj().T,np.eye(3))
    c.mc('F10','Left/right gauge action preserves second Haar moment',
         np.abs(gs[0]@u@gs[1].conj().T)**2,np.ones((3,3))/3)
    c.check('F11','Q_p does not satisfy |x|_p <= 1 globally',padic_norm(Fraction(1,3),3)==3,
            {'x':'1/3','p':3,'p_adic_norm':3,'unit_ball':'Z_p only'},kind='EXACT',status='COUNTEREXAMPLE')
    c.check('F12','Compact link group does not bound electric energy',casimir(100,0)>1000,
            {'C2(100,0)':str(casimir(100,0)),'formula':'C2(a,0)=a(a+3)/3 -> infinity'},kind='EXACT',status='COUNTEREXAMPLE')
    c.note('F13','Boundary compactness is not bulk compactness or a UV cutoff',
           'P^1(Q_p) is compact in its projective/chordal topology. Q_p is not compact. The infinite tree with graph distance is unbounded. Product Haar normalization follows from compact SU(3), not the boundary geometry.')
    c.close('F14','780 nm corresponds approximately to 384 THz',299792458/(780e-9)/1e12,384,tol=0.5)
    c.check('F15','Optical and microwave single-photon energies differ',384e12/5.2e9>70000,
            {'frequency_energy_ratio':384e12/5.2e9,'interpretation':'Pumped frequency conversion must supply/remove the energy difference; the paper gives no dynamical transducer model.'},status='QUALIFICATION')
    c.note('F16','Hardware fidelity and SU(3) encoding are unspecified',
           'A two-level register alone is not a fundamental qutrit or L2(SU(3)). Multi-qubit encoding, controls, pump Hamiltonian, losses, coherence and experimental data are absent; no device-performance validation is possible.')
    pairs=[(Fraction(a,6),Fraction(b,10)) for a in range(-4,5) for b in range(-4,5)]
    c.check('F17','Exact rational p-adic norm is multiplicative and ultrametric',
            all(padic_norm(x*y,p)==padic_norm(x,p)*padic_norm(y,p) and
                padic_norm(x+y,p)<=max(padic_norm(x,p),padic_norm(y,p))
                for p in [2,3,5,7] for x,y in pairs),
            {'primes':[2,3,5,7],'rational_pairs_per_prime':len(pairs)},kind='EXACT')
    return u
