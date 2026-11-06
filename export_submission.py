#!/usr/bin/env python3
"""Alias for the single maintained v2 submission converter; see --help."""
from prepare_submission import main
from decimal import InvalidOperation
import sys

if __name__=='__main__':
    try:raise SystemExit(main())
    except (ValueError,OSError,InvalidOperation,KeyError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr)
        raise SystemExit(2)
