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
"""Checks for all ten theorems. Qualifications are part of the result."""
import math
import numpy as np
from scipy.special import logsumexp
from common import *


def incidence(vertices,edges):
    b=np.zeros((vertices,len(edges)))
    for k,(i,j) in enumerate(edges): b[i,k]=-1; b[j,k]=1
    return b


def partition_link(t,cutoff):
    """Triangular truncation a+b<=cutoff. t=beta*g^2/2 must be >0.
    Omitted-shell upper bound: b_n=(n+2)^7 exp(-t*n^2/4)/64.
    For n>=K+1, shell_n <= b_n; ratios b_(n+1)/b_n decrease.
    If r<1, the entire omitted tail is <= b_(K+1)/(1-r).
    This analytic bound is evaluated in float64, not interval arithmetic.
    """
    if t<=0 or cutoff<0: raise ValueError('Require t>0 and cutoff>=0')
    terms=[2*math.log(dimension(a,b))-t*float(casimir(a,b))
           for a in range(cutoff+1) for b in range(cutoff+1-a)]
    logk=float(logsumexp(terms))
    n=cutoff+1
    logr=7*math.log1p(1/(n+2))-t*(2*n+1)/4
    if logr>=0: tail=math.inf
    else:
        logb=7*math.log(n+2)-math.log(64)-t*n*n/4
        tail=math.exp(logb)/(-math.expm1(logr))
    return math.exp(logk),tail


def run(c,rng,u):
    # T1: nontrivial gauge-invariant spaces, with independent analytic projectors.
    for reps in [('3',),('3','bar3'),('3','3','3')]:
        p=singlet_projector(reps)
        if len(reps)==1: expected=np.zeros((3,3))
        else:
            s=(np.eye(3).reshape(-1)/math.sqrt(3) if len(reps)==2
               else epsilon().reshape(-1)/math.sqrt(6))
            expected=np.outer(s,s)
        tag='T1.'+str(len(reps))
        c.close(tag+'a','Gauss nullspace equals analytic singlet projector '+str(reps),p,expected)
        c.close(tag+'b','Projector idempotence and Hermiticity',np.array([np.linalg.norm(p@p-p),np.linalg.norm(p-p.conj().T)]),0)
        c.close(tag+'c','All eight Gauss generators annihilate projector',total_generators(reps)@p,0)
    averaged=np.einsum('nij,nkl->nikjl',u,u.conj()).reshape(len(u),9,9)
    c.mc('T1.4','Haar-averaged 3 tensor bar3 equals singlet projector',averaged,singlet_projector(('3','bar3')))
    c.note('T1.scope','Finite representation sectors tested',
           'Tests cover 3, 3 tensor bar3 and 3 tensor 3 tensor 3. The infinite-dimensional Haar-projector theorem requires the compact-group averaging argument; finite matrices alone do not prove it.')

    # T2: verify Casimir from Lie generators, then exact arithmetic and boundary cases.
    t=generators()
    c.close('T2.1','Fundamental generator normalization',np.einsum('aij,bji->ab',t,t),np.eye(8)/2)
    c.close('T2.2','Fundamental electric Casimir is 4/3',sum(x@x for x in t),np.eye(3)*4/3)
    c.close('T2.3','3 tensor 3 splits into bar3 and 6 Casimirs',np.linalg.eigvalsh(constraint(('3','3'))),[4/3]*3+[10/3]*6)
    c.close('T2.4','3 tensor bar3 splits into 1 and 8 Casimirs',np.linalg.eigvalsh(constraint(('3','bar3'))),[0]+[3]*8)
    values=[casimir(a,b) for a in range(41) for b in range(41) if a+b]
    c.check('T2.5','Casimir floor and one-/two-link energies at g=1',
            min(values)==Fraction(4,3) and casimir(0,0)==0,
            {'min_nontrivial':str(min(values)),'one_link':'2/3','two_links':'4/3',
             'global_argument':'For a+b=n>=1, C2=(n^2-ab+3n)/3 >= n^2/4+n >=5/4; n=1 gives 4/3 and n>=2 gives >=3.'},kind='EXACT')
    # A single L2(SU3) link truncated to 1 + (3 x bar3): both endpoint constraints.
    qboth=np.zeros((10,10)); qboth[1:,1:]=np.eye(9)*8/3
    c.check('T2.6','Imposing Gauss law at BOTH free-tree leaves leaves only vacuum',
            np.count_nonzero(np.linalg.eigvalsh(qboth)<1e-10)==1,
            {'physical_dimension':1,'argument':'Leaf constraint forces trivial incident irrep; remove leaves inductively on every finite tree.'},status='QUALIFICATION')
    ranks=[round(np.trace(singlet_projector(r)).real) for r in [('3',),('3','bar3')]]
    c.check('T2.7','Sibling-leaf flux requires relaxed leaf constraints',ranks==[0,1],
            {'internal_vertex_singlet_ranks':ranks,'minimum_nonzero_path_length':2,
             'conditions':'Depth >=1 with sibling leaves; Gauss law imposed internally only. Depth-zero tree has no excitation. A one-edge interval with both ends free has a one-link excitation.'},status='QUALIFICATION')
    c.note('T2.sector','Static-source energy is not an excited level above a vacuum in the same sector',
           'The trivial vacuum violates Gauss law with external 3/bar3 charges. The adjacent-source flux state has energy 2g^2/3 relative to the source-free vacuum in a different sector; this is not by itself a within-sector spectral gap.')
    c.check('T2.infinite','Finite-link discreteness does not establish infinite-volume discrete spectrum',
            all(18*k>18*(k-1) for k in [2,4,8,16]),
            {'single_link_excitation_multiplicity_unconstrained':[18*k for k in [1,2,4,8,16]],
             'argument':'In the unconstrained infinite product reference-vacuum space, translate a fundamental excitation to infinitely many links. These orthogonal equal-energy states give infinite multiplicity and prevent compact resolvent. The fully constrained pure finite-tree space instead has only the vacuum; an infinite physical sector must be specified.'},status='QUALIFICATION')

    # T3: independent links along nonempty paths.
    w=np.tile(np.eye(3,dtype=complex),(len(u),1,1))
    for length in range(1,5):
        w=w@(u if length==1 else haar_su3(rng,len(u)))
        c.mc('T3.'+str(length),'Open Wilson trace vanishes, path length '+str(length),np.trace(w,axis1=1,axis2=2),0)
    c.check('T3.exact','Nontrivial character averages vanish exactly',
            all(haar_class(x)==0 for x in [CHI3,CHIBAR,CHI6,CHI8]),
            {'irreps':['3','bar3','6','8'],'nonempty_path_required':True},kind='EXACT')
    c.note('T3.scope','Open trace is gauge variant at independently transformed endpoints',
           'The paper omits a trace in the first factorization equation; the valid expression is Tr(product of matrix-valued Haar means). A zero vacuum expectation does not mean the line operator is zero.')

    # T4: finite core models of quotient cycles, not arithmetic Schottky construction.
    for name,v,edges in [('tree',4,[(0,1),(1,2),(2,3)]),
                         ('one_cycle',3,[(0,1),(1,2),(2,0)]),
                         ('two_cycles',4,[(0,1),(1,2),(2,0),(2,3),(3,0)])]:
        rank=int(np.linalg.matrix_rank(incidence(v,edges)))
        c.check('T4.'+name,'Cycle rank E-V+1 for connected '+name,
                len(edges)-rank==len(edges)-v+1,{'cycle_rank':len(edges)-rank},kind='FINITE_LINEAR_ALGEBRA')
    c.check('T4.integral','Printed bare cyclic Haar integral is zero, not nonzero',haar_class(CHI3)==0,
            {'integral_Tr_Uc':'0','reason':'Independent Haar links along a simple cycle; their product is Haar.'},kind='EXACT',status='COUNTEREXAMPLE')
    c.check('T4.operator','Cyclic trace operator has a nonzero vacuum-to-character matrix element',
            haar_class(mul(CHIBAR,CHI3))==1,{'integral_conj_chi3_times_chi3':'1'},kind='EXACT')
    c.close('T4.backtrack','Backtracking closed walk has trace 3, not zero',np.trace(u[0]@u[0].conj().T),3)
    c.note('T4.scope','No tree plaquettes means an empty magnetic sum',
           'It does not mean every closed-walk trace vanishes. A quotient graph still needs selected cycles and magnetic couplings. E-V+1 applies to a finite connected graph/core; a raw infinite quotient may have infinite E,V. No Schottky generators are supplied.')

    # T5: radial recurrence on the REGULAR infinite tree, not a boundary-truncated Laplacian.
    errors=[]; boundary_errors=[]; wrong_sign=[]
    for p in [2,3,5,7]:
        for delta in [0.3,0.7,1.0,1.6,2.3]:
            m2=p**delta+p**(1-delta)-(p+1)
            a=1/(p**delta-p**(-delta))
            g=lambda d:a*p**(-delta*d)
            errors.append((p+1+m2)*g(0)-(p+1)*g(1)-1)
            for d in range(1,8): errors.append((p+1+m2)*g(d)-g(d-1)-p*g(d+1))
            wrong_sign.append(-(p+1)*g(0)+(p+1)*g(1)-m2*g(0))
            for depth in [3,7,12]:
                for v in range(depth):
                    dist=2*(depth-v)
                    boundary_errors.append(g(dist)*p**(2*delta*depth)/(a*p**(2*delta*v))-1)
    c.close('T5.recurrence','Displayed positive kernel solves (L + m^2)G=delta for L=degree-adjacency',errors,0)
    c.close('T5.boundary','Renormalized boundary power law and d=2(N-v)',boundary_errors,0)
    distance_ok=True
    for p in [2,3,5,7]:
        depth=5
        for x,y in [(0,1),(1,1+p),(2,2+p*p),(3,3+p**4)]:
            xd=[(x//p**k)%p for k in range(depth)]
            yd=[(y//p**k)%p for k in range(depth)]
            prefix=next(k for k in range(depth) if xd[k]!=yd[k])
            graph_distance=2*(depth-prefix)
            norm=padic_norm(x-y,p)
            distance_ok &= norm==Fraction(p)**(-prefix)
            distance_ok &= abs(graph_distance-(2*depth+2*math.log(float(norm),p)))<1e-10
    c.check('T5.distance','Digit-path common ancestors reproduce p-adic geodesic distance',
            distance_ok,{'depth':5,'primes':[2,3,5,7],'boundary_pairs':16})
    c.check('T5.sign','For Box=adjacency-degree, printed (Box-m^2)G gives -delta',
            np.allclose(wrong_sign,-1),{'source_at_origin':wrong_sign[0],'required_change':'Use (L+m^2)G=delta, or negate the displayed G for (Box-m^2).'},status='COUNTEREXAMPLE')
    c.note('T5.mass','Mass-dimension relation and boundary prescription are required',
           'm^2=p^Delta+p^(1-Delta)-(p+1). The positive radial formula obeys the local recurrence for Delta>0, but an l2 resolvent column requires Delta>1/2: shell norms scale as p^((1-2Delta)d). At/below 1/2 extra boundary/Green-function interpretation is needed. This verifies scalar-kernel scaling, not an SU(3) holographic operator duality.')

    # T6: exact class integrals give the whole degenerate perturbation matrix.
    chars=[ONE,CHI3,CHIBAR]
    v=np.array([[float(haar_class(mul(conj(a),mul(add(CHI3,CHIBAR),b))))
                 for b in chars] for a in chars])*(-0.5) # g=1
    c.close('T6.diagonal','Vacuum and fundamental diagonal first-order shifts vanish',np.diag(v),0)
    c.close('T6.fusion','Fundamental fusion dimensions and multiplicities',
            [haar_class(mul(conj(CHIBAR),mul(CHI3,CHI3))),haar_class(mul(conj(CHI6),mul(CHI3,CHI3)))],[1,1])
    c.check('T6.degenerate','Vanishing diagonals do not imply vanishing first-order gap shift on a cycle',
            np.allclose(np.linalg.eigvalsh(v[1:,1:]),[-0.5,0.5]),
            {'basis':['chi_3','chi_bar3'],'perturbation_g1':v[1:,1:].tolist(),
             'first_order_shifts_g1':np.linalg.eigvalsh(v[1:,1:]).tolist(),
             'model':'One independent cycle holonomy; H_E chi_r=(L*g^2/2) C2(r) chi_r. Fundamental/conjugate are degenerate. H_B=-(chi3+chi_bar3)/(2g^2).'},kind='EXACT_INTEGRALS',status='COUNTEREXAMPLE')
    c.check('T6.open','Open-tree magnetic term and all perturbative shifts are identically zero',sum([])==0,{'plaquette_set':[]},kind='EXACT')
    c.close('T6.scaling','Relative magnetic/electric prefactor scales as g^-4',
            [(1/(2*g*g))/(g*g/2) for g in [1,2,4,8]],[g**-4 for g in [1,2,4,8]])

    # T7: finite cylinder functions; independence/product measure and inversion.
    fs=[mul(CHI3,CHIBAR),mul(mul(CHI3,CHI3),CHI3),add(ONE,CHI8)]
    original=math.prod(haar_class(f) for f in fs)
    transformed=math.prod(haar_class(conj(fs[i])) for i in [2,0,1])
    c.check('T7.exact','Product Haar cylinder moments invariant under relabeling/inversion',
            original==transformed==1,{'before':str(original),'after':str(transformed)},kind='EXACT')
    c.mc('T7.numeric','Orientation reversal preserves matrix second moments',np.abs(u.conj().transpose(0,2,1))**2,np.ones((3,3))/3)
    c.note('T7.scope','Infinite product invariance follows from finite cylinders',
           'An automorphism need not preserve each chosen finite edge subset. It sends its cylinder to the corresponding relabeled cylinder. Inversion must be included when a chosen independent-edge orientation reverses.')

    # T8: actual bit counts, correct tree geometry, and distinction from costs.
    rows=[]
    for p in [2,3,5]:
        for d in [0,1,2,4,8,16]:
            rooted=(p**(d+1)-1)//(p-1)
            ball=1+(p+1)*(p**d-1)//(p-1)
            bits=(ball-1).bit_length()
            c.check(f'T8.p{p}d{d}','Binary vertex-index capacity',2**bits>=ball and (bits==0 or 2**(bits-1)<ball),
                    {'rooted_p_ary_vertices':rooted,'regular_tree_ball_vertices':ball,'binary_bits':bits,
                     'one_hot_qubits':ball,'gauge_qubits_dim19':(ball-1)*5},kind='EXACT_INTEGER')
            rows.append((p,d,rooted,ball,bits))
    c.check('T8.onehot','One-hot encoding needs N qubits, not ceil(log2 N)',rows[-1][3]>rows[-1][4],
            {'distinction':'ceil(log2 N) describes binary indexing. The paper counts a rooted p-ary subtree; a ball of the degree-(p+1) tree has 1+(p+1)(p^D-1)/(p-1) vertices.'},kind='EXACT_INTEGER',status='COUNTEREXAMPLE')
    c.note('T8.oracle','Register size does not establish block-encoding oracle/gate complexity',
           'The paper supplies no oracle implementation or data-access model. Only the O(D log2 p) binary-index qubit count is checked. Gauge-field storage remains linear in links at fixed truncation.')

    # T9: exact multiplicities, convergence and a useful tail bound.
    convergence=[]
    for t in [0.05,0.2,0.7,2.0]:
        k=12
        while True:
            z,tail=partition_link(t,k)
            if tail<1e-9*max(z,1): break
            k+=4
        fine,_=partition_link(t,k+12)
        c.check('T9.t'+str(t),'Positive-temperature link sum converges within analytic tail bound',
                fine>=z-1e-10 and fine-z<=tail+1e-10*max(z,1),
                {'t':t,'cutoff_a_plus_b':k,'K_partial':z,'tail_bound':tail,'K_finer':fine})
        convergence.append(z)
    c.check('T9.multiplicity','Peter-Weyl truncated link dimension is sum dim(r)^2',
            sum(dimension(a,b)**2 for a,b in [(0,0),(1,0),(0,1)])==19,
            {'irreps':['1','3','bar3'],'link_dimension':19},kind='EXACT')
    c.note('T9.physical','Product heat trace is unprojected, not generally a Gauss-law subspace trace',
           'For finite pure trees constrained at all vertices, Z_phys=1, whereas K(t)^E>1 for E>0. A gauge-projected trace is bounded by the finite full trace, but does not factor into unconstrained link traces.')
    c.check('T9.beta0','The convergence statement requires beta>0 as well as g>0',
            sum(dimension(a,b)**2 for a in range(11) for b in range(11))>121,
            {'reason':'At beta=0 every irrep contributes dim(r)^2>=1, so the infinite trace diverges; beta<0 also diverges.'},kind='EXACT',status='COUNTEREXAMPLE')
    kval,_=partition_link(0.7,40)
    c.check('T9.volume','Finite-depth convergence does not give finite infinite-volume Z',kval>1,
            {'K':kval,'log_Z_at_edge_counts_1_10_100':[e*math.log(kval) for e in [1,10,100]],
             'limit':'K^E diverges for fixed positive temperature as E->infinity (unprojected).'},status='QUALIFICATION')

    # T10: construct actual finite pure states with a specified TTN bond rank.
    for d in [1,2,3,5]:
        a=rng.normal(size=(d+3,d))+1j*rng.normal(size=(d+3,d))
        b=rng.normal(size=(d+4,d))+1j*rng.normal(size=(d+4,d))
        psi=a@b.T; psi/=np.linalg.norm(psi)
        s=np.linalg.svd(psi,compute_uv=False); probs=s*s; probs=probs[probs>1e-14]
        entropy=float(-np.sum(probs*np.log(probs)))
        c.check('T10.d'+str(d),'TTN bond-rank entropy bound',len(probs)<=d and entropy<=math.log(d)+1e-10,
                {'bond_dimension':d,'schmidt_rank':len(probs),'entropy_nats':entropy,'bound_nats':math.log(d)})
        c.close('T10.saturate'+str(d),'Maximally entangled bond saturates log d',-np.sum(np.ones(d)/d*np.log(np.ones(d)/d)),math.log(d))
    c.check('T10.scope','Graph topology alone does not bound arbitrary-state Schmidt rank',math.log(4)>math.log(2),
            {'example':'Two qubits on each side in two Bell pairs: S=log4. Drawing a tree cut and assigning physical edge dimension 2 does not make this a bond-2 TTN.',
             'required_hypothesis':'A pure-state TTN with virtual bond dimension d_Lambda, or an independently proved Schmidt-rank bound. Mixed states need a different entropy statement.'},status='QUALIFICATION')
