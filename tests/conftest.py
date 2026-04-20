"""Import-path setup for the test suite.

Adding the repo root to ``sys.path`` lets tests do ``from minalphafold.X
import Y``; adding ``scripts/`` lets ``test_openproteinset_scripts`` import
the standalone entry points by filename (``from download_openproteinset
import ...``). pytest picks this up automatically via its ``conftest``
mechanism — no per-test boilerplate.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for entry in (REPO_ROOT, REPO_ROOT / "scripts"):
    path_str = str(entry)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
