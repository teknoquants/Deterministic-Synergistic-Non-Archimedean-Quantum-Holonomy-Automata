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
"""Finite classical simulations of Section VII and checks of missing assumptions."""
import math
import numpy as np
from common import *


def qpe_zero(q,psi,m,phi):
    """Statevector QPE: all controlled powers followed by an inverse QFT.
    Returns unnormalized system state conditioned on ancilla outcome zero,
    and all ancilla measurement probabilities. No sampling is needed.
    """
    vals,vecs=np.linalg.eigh(q)
    coeff=vecs.conj().T@psi
    size=2**m
    times=np.arange(size)
    evolved=(np.exp(1j*phi*times[:,None]*vals)*coeff[None,:])@vecs.T
    # numpy FFT has the minus sign of the inverse QFT in our register convention.
    after=np.fft.fft(evolved,axis=0)/size
    return after[0],np.sum(np.abs(after)**2,axis=1)


def run(c,rng,u):
    q=constraint(('3','bar3')); p=singlet_projector(('3','bar3'))
    psi=rng.normal(size=9)+1j*rng.normal(size=9); psi/=np.linalg.norm(psi)
    post,prob=qpe_zero(q,psi,3,2*math.pi/8)
    c.close('A1.normalization','QPE ancilla probabilities sum to one',prob.sum(),1)
    c.close('A1.projection','Resolved non-aliased QPE zero outcome applies Gauss projector',post,p@psi)
    c.close('A1.success','Projection success probability equals invariant weight',prob[0],np.linalg.norm(p@psi)**2)
    aliased,pa=qpe_zero(q,psi,3,2*math.pi/3)
    c.check('A1.alias','An unsuitable phase aliases charged states to phase zero',
            np.linalg.norm(aliased-psi)<1e-10 and pa[0]>0.999,
            {'phi':'2*pi/3','nonzero_constraint_eigenvalue':3,'zero_outcome_probability':float(pa[0]),
             'actual_invariant_weight':float(np.linalg.norm(p@psi)**2)},status='COUNTEREXAMPLE')
    finite,pf=qpe_zero(q,psi,3,2*math.pi/7)
    leakage=float(np.linalg.norm((np.eye(9)-p)@finite)**2)
    c.check('A1.resolution','Finite QPE resolution can retain nonzero-charge amplitude',leakage>1e-8,
            {'m':3,'phi':'2*pi/7','unnormalized_charged_weight_after_zero_outcome':leakage},status='QUALIFICATION')
    c.note('A1.scope','QPE is a conditional projection, not automatic deterministic preparation',
           'Need initial singlet overlap, constraint-spectrum bounds, a non-aliased phase, precision analysis, and postselection or amplitude amplification. exp(i phi sum G_a^2) is the constraint oracle, not a generic SU(3) group element U(g_v).')

    # A2: valid unitary dilation uses a 3-dimensional color reference.
    bell=np.eye(3).reshape(-1)/math.sqrt(3)
    x=u[0]
    overlap=np.vdot(bell,np.kron(x,np.eye(3))@bell)
    c.close('A2.trace','Color-reference overlap is normalized trace Tr(U)/3',overlap,np.trace(x)/3)
    # Explicit Hadamard-test branches: (psi +/- U psi)/2.
    evolved=np.kron(x,np.eye(3))@bell
    probability=np.linalg.norm((bell+evolved)/2)**2
    c.close('A2.circuit','Hadamard circuit gives (1+Re Tr(U)/3)/2',probability,(1+np.trace(x).real/3)/2)
    c.mc('A2.vacuum','Haar-vacuum average of valid Hadamard-test probability is 1/2',
         (1+np.trace(u,axis1=1,axis2=2).real/3)/2,0.5)
    traced_control=np.diag([1,3]) # U=I gives Tr(U)=3
    residual=float(np.linalg.norm(traced_control.conj().T@traced_control-np.eye(2)))
    c.check('A2.nonunitary','The traced Wilson operator cannot directly be a controlled unitary',residual>1,
            {'U':'identity_3','Tr_U':3,'controlled_gate_unitarity_residual':residual,
             'repair':'Use unitary holonomy with a color/reference register, or a specified normalized block encoding.'},status='COUNTEREXAMPLE')

    # A3: exact finite Hermitian-contraction block encoding and a known QSP family.
    a=np.diag([-0.9,-0.3,0.0,0.4,0.85]); b=np.diag(np.sqrt(1-np.diag(a)**2))
    h=np.block([[a,b],[b,-a]])
    w=np.block([[a,1j*b],[1j*b,a]])
    c.close('A3.block','Hermitian block encoding is unitary and has top-left block A',h.conj().T@h,np.eye(10))
    c.close('A3.extract','Block-encoding normalization',h[:5,:5],a)
    for degree in [1,2,3,7,12]:
        result=np.linalg.matrix_power(w,degree)[:5,:5]
        expected=np.diag(np.cos(degree*np.arccos(np.diag(a))))
        c.close('A3.chebyshev'+str(degree),'Degree-d zero-phase QSP realizes T_d(A) using d signal calls',result,expected)
    printed=np.linalg.matrix_power(h,2)[:5,:5]
    target=2*a@a-np.eye(5)
    c.check('A3.sequence','A block encoding alone does not specify the claimed QSVT polynomial',
            np.linalg.norm(printed-target)>1,
            {'example':'With the printed zero-phase sequence and Hermitian U_H, d=2 gives I. A conventional signal rotation W^2 gives T2(A). Phases, signal conventions and target polynomial must be specified.',
             'max_difference_from_T2':float(np.max(np.abs(printed-target)))},status='QUALIFICATION')
    # Explicit bounded spectral filter; valid polynomial diagnostic, not phase synthesis.
    rows=[]
    for gap in [0.1,0.2,0.4,0.7]:
        for eps in [1e-3,1e-6]:
            k=math.ceil(math.log(eps)/math.log1p(-gap*gap))
            leakage=math.exp(k*math.log1p(-gap*gap))
            c.check(f'A3.filter{gap}_{eps}','Filter f(x)=(1-x^2)^k meets stopband tolerance',
                    leakage<=eps*(1+1e-12),{'normalized_gap':gap,'epsilon':eps,'degree':2*k,'stopband_sup':leakage})
            rows.append({'gap':gap,'epsilon':eps,'degree':2*k})
    c.note('A3.gap','Filtering complexity cannot be specified by epsilon alone',
           {'example_filter_degrees':rows,
            'bound':'This explicit (nonoptimal) bounded polynomial costs O(gap^-2 log(1/epsilon)); it demonstrates gap dependence, not optimal QSVT scaling. Markov: for |P|<=1 on [-1,1], P(0)=1 and |P(delta)|<=epsilon imply d>=sqrt((1-epsilon)/delta).',
            'missing':'Target filter, spectral promise/gap, block normalization, parity/boundedness, phase synthesis, oracle cost and precision. No full paper-specific QSVT algorithm is defined.'})
    c.check('A3.unbounded','H/||H|| needs irrep truncation or another bounded encoding',
            casimir(100,0)>casimir(10,0),
            {'reason':'Finite depth does not bound SU(3) electric Casimir. Even one full L2(SU(3)) link has ||H_E||=infinity.'},kind='EXACT',status='QUALIFICATION')
