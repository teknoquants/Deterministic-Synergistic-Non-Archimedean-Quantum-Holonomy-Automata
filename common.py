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
"""Shared numerical and exact-arithmetic tools; no quantum hardware required."""
from fractions import Fraction
from itertools import permutations
import math
import numpy as np


def padic_norm(x, p):
    """Exact rational p-adic norm for a prime p (validate caller's prime)."""
    if p < 2 or any(p % k == 0 for k in range(2, math.isqrt(p)+1)):
        raise ValueError('p must be prime')
    x=Fraction(x)
    if x == 0: return Fraction(0)
    a,b=abs(x.numerator),x.denominator
    valuation=0
    while a % p == 0: a//=p; valuation+=1
    while b % p == 0: b//=p; valuation-=1
    return Fraction(p)**(-valuation)


def casimir(a, b):
    return Fraction(a*a+b*b+a*b+3*a+3*b, 3)


def dimension(a, b):
    return (a+1)*(b+1)*(a+b+2)//2


def haar_su3(rng, n):
    """Complex Gaussian QR -> Haar U(3); determinant correction -> Haar SU(3)."""
    z = rng.normal(size=(n,3,3)) + 1j*rng.normal(size=(n,3,3))
    q, r = np.linalg.qr(z)
    d = np.diagonal(r, axis1=1, axis2=2)
    q = q * (d/np.abs(d))[:,None,:]
    q[:,:,0] /= np.linalg.det(q)[:,None]
    return q


def generators():
    """T_a=lambda_a/2, Tr(T_a T_b)=delta_ab/2."""
    return np.array([
        [[0,1,0],[1,0,0],[0,0,0]],
        [[0,-1j,0],[1j,0,0],[0,0,0]],
        [[1,0,0],[0,-1,0],[0,0,0]],
        [[0,0,1],[0,0,0],[1,0,0]],
        [[0,0,-1j],[0,0,0],[1j,0,0]],
        [[0,0,0],[0,0,1],[0,1,0]],
        [[0,0,0],[0,0,-1j],[0,1j,0]],
        np.diag([1,1,-2])/math.sqrt(3)], dtype=complex)/2


def total_generators(reps):
    t = generators()
    factors = [t if r == '3' else -t.conj() for r in reps]
    out = []
    for a in range(8):
        total = np.zeros((3**len(reps),)*2, complex)
        for i in range(len(reps)):
            term = np.array([[1.]])
            for j in range(len(reps)):
                term = np.kron(term, factors[j][a] if i == j else np.eye(3))
            total += term
        out.append(total)
    return np.array(out)


def constraint(reps):
    ts = total_generators(reps)
    return sum(t@t for t in ts)


def singlet_projector(reps):
    q = constraint(reps)
    vals, vecs = np.linalg.eigh(q)
    basis = vecs[:, np.abs(vals)<1e-10]
    return basis@basis.conj().T


def epsilon():
    e = np.zeros((3,3,3))
    for p in permutations(range(3)):
        inv = sum(p[i]>p[j] for i in range(3) for j in range(i+1,3))
        e[p] = (-1)**inv
    return e

# Exact Laurent polynomials in eigenvalues z1=x, z2=y, z3=(xy)^(-1).
def add(a,b):
    c = dict(a)
    for k,v in b.items(): c[k] = c.get(k,0)+v
    return {k:v for k,v in c.items() if v}


def scale(a,s): return {k:v*s for k,v in a.items() if v*s}


def mul(a,b):
    c = {}
    for (i,j),v in a.items():
        for (k,l),w in b.items(): c[i+k,j+l] = c.get((i+k,j+l),0)+v*w
    return {k:v for k,v in c.items() if v}


def conj(a): return {(-i,-j):v for (i,j),v in a.items()}


ONE = {(0,0):1}
CHI3 = {(1,0):1,(0,1):1,(-1,-1):1}
CHIBAR = conj(CHI3)
CHI6 = add(mul(CHI3,CHI3),scale(CHIBAR,-1))
CHI8 = add(mul(CHI3,CHIBAR),scale(ONE,-1))


def haar_class(poly):
    """Exact Weyl integral: constant term of f*|Vandermonde|^2 / 6.
    Only for class Laurent polynomials, not arbitrary matrix-entry functions.
    """
    zs = [{(1,0):1},{(0,1):1},{(-1,-1):1}]
    vand = ONE
    for i in range(3):
        for j in range(i+1,3): vand = mul(vand,add(zs[i],scale(zs[j],-1)))
    return Fraction(mul(poly,mul(vand,conj(vand))).get((0,0),0),6)


class Checks:
    def __init__(self): self.rows=[]
    def check(self, ident, claim, ok, evidence, kind='NUMERICAL', status='PASS'):
        row = dict(id=ident, claim=claim, status=status if bool(ok) else 'FAIL',
                   kind=kind, evidence=evidence)
        self.rows.append(row)
        print(f"{row['status']:14} {ident}: {claim}")
    def close(self, ident, claim, actual, expected, tol=1e-10):
        error = float(np.max(np.abs(np.asarray(actual)-np.asarray(expected))))
        self.check(ident,claim,error<=tol,dict(max_abs_error=error,tolerance=tol))
    def note(self,ident,claim,evidence):
        self.check(ident,claim,True,evidence,kind='SCOPE',status='QUALIFICATION')
    def mc(self,ident,claim,values,expected):
        """7 standard errors per complex coordinate; diagnostic, not certification."""
        values=np.asarray(values)
        mean=values.mean(axis=0)
        se=np.sqrt(np.mean(np.abs(values-mean)**2,axis=0)/(len(values)-1))
        err=np.abs(mean-expected)
        tol=7*se+1e-12
        self.check(ident,claim,np.all(err<=tol),
                   dict(samples=len(values),max_error=float(err.max()),
                        max_tolerance=float(tol.max()),
                        max_standard_errors=float(np.max(err/(se+1e-12)))),kind='MONTE_CARLO')
