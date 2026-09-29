#!/usr/bin/env python3
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
"""Run all checks and emit human-readable and machine-readable audit reports."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import sys
import numpy as np
import scipy
from common import Checks
import validate_foundations
import validate_theorems
import validate_subroutines


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=int,default=20000,help='Haar Monte Carlo samples (default 20000)')
    parser.add_argument('--seed',type=int,default=20260929)
    parser.add_argument('--output',type=Path,default=Path('results'))
    parser.add_argument('--section',choices=['all','foundations','theorems','subroutines'],default='all')
    args=parser.parse_args()
    if args.samples<100: parser.error('--samples must be >=100')
    rng=np.random.default_rng(args.seed); c=Checks()
    if args.section in ['all','foundations']:
        u=validate_foundations.run(c,rng,args.samples)
    else:
        from common import haar_su3
        u=haar_su3(rng,args.samples)
    if args.section in ['all','theorems']: validate_theorems.run(c,rng,u)
    if args.section in ['all','subroutines']: validate_subroutines.run(c,rng,u)
    counts=dict(Counter(r['status'] for r in c.rows))
    report=dict(paper='DS-NAQHA, September 2026; attached 1788940434090.pdf',
                generated_utc=datetime.now(timezone.utc).isoformat(),
                environment=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
                seed=args.seed,samples=args.samples,section=args.section,counts=counts,checks=c.rows,
                interpretation='PASS means the stated finite/exact check passed, not that the entire paper theorem is proved. COUNTEREXAMPLE is a successfully reproduced contradiction or failure of a stated formula/extension. QUALIFICATION identifies necessary scope/assumptions. FAIL means a check did not reproduce its expected result.')
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'validation_report.json').write_text(json.dumps(report,indent=2)+'\n')
    lines=['# DS-NAQHA validation results','',report['interpretation'],'',
           f"Seed: {args.seed}; samples: {args.samples}; environment: {report['environment']}",'',
           'Counts: '+json.dumps(counts),'']
    for row in c.rows:
        lines.extend([f"## {row['id']} — {row['status']}",'',row['claim'],
                      '',f"Method: {row['kind']}",'','```json',json.dumps(row['evidence'],indent=2),'```',''])
    (args.output/'validation_report.md').write_text('\n'.join(lines))
    print('\nSUMMARY:',counts)
    return 1 if counts.get('FAIL',0) else 0


if __name__=='__main__': sys.exit(main())
