"""Run existing EZR demos against a configurable MOOT checkout (Python 3.12+)."""
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ezr_eg as demos

root = Path(os.getenv('MOOT', '../moot')).expanduser().resolve()
# run() restores defaults after every demo, so both settings must be updated.
demos.the._defaults.update(File=str(root/'optimize/misc/auto93.csv'),
                          Klass=str(root/'classify/diabetes.csv'))
demos.the.update(demos.the._defaults)
demos.eg_all()
