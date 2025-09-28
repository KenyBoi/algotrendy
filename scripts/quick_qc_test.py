"""Quick smoke tests for quantconnect_integration.py

This script is intentionally standalone (not pytest) so it won't load project
`tests/conftest.py` which imports heavy deps. Run with the venv Python:

  python scripts/quick_qc_test.py

It performs small mocked checks (no network) to validate the module surface.
"""
import sys
import os

# Ensure the `src/` folder is on sys.path so this script can import the package
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

import importlib.util

# Load the quantconnect_integration module directly to avoid importing the
# full package (which pulls in heavy deps like pandas via __init__.py).
QC_PATH = os.path.join(SRC, 'algotrendy', 'quantconnect_integration.py')
import types

# Create a lightweight 'algotrendy' package in sys.modules so relative imports
# inside quantconnect_integration.py (e.g., from .config import CONFIG) work
# without importing algotrendy.__init__ which pulls heavy deps.
pkg_name = 'algotrendy'
pkg_mod = types.ModuleType(pkg_name)
pkg_mod.__path__ = [os.path.join(SRC, 'algotrendy')]
sys.modules[pkg_name] = pkg_mod

# Load algotrendy.config as a submodule
CONFIG_PATH = os.path.join(SRC, 'algotrendy', 'config.py')
spec_cfg = importlib.util.spec_from_file_location(f'{pkg_name}.config', CONFIG_PATH)
cfg_mod = importlib.util.module_from_spec(spec_cfg)
spec_cfg.loader.exec_module(cfg_mod)
sys.modules[f'{pkg_name}.config'] = cfg_mod

# Now load quantconnect_integration as a submodule of algotrendy
spec = importlib.util.spec_from_file_location(f'{pkg_name}.quantconnect_integration', QC_PATH)
qc_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(qc_mod)
sys.modules[f'{pkg_name}.quantconnect_integration'] = qc_mod

QuantConnectIntegration = qc_mod.QuantConnectIntegration
generate_qc_futures_algorithm = qc_mod.generate_qc_futures_algorithm


class _MockResp:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def main():
    print("Quick QC smoke test")

    # Initialize without credentials - should not raise
    qc = QuantConnectIntegration()
    print("- Created QuantConnectIntegration (no creds) ok")

    # Monkeypatch session.request to return a mock response
    qc.session.request = lambda method, url, headers=None, json=None, params=None, timeout=None: _MockResp({'ok': True, 'url': url, 'method': method})

    res = qc._make_request('projects')
    assert isinstance(res, dict) and res.get('ok') is True
    print(f"- _make_request mocked response OK: {res.get('ok')}")

    # Test algorithm generator
    code = generate_qc_futures_algorithm(['ES', 'NQ'], {'p': 1})
    assert isinstance(code, str) and 'class AlgoTrendyFuturesAlgorithm' in code
    print("- generate_qc_futures_algorithm produced algorithm string")

    print("All quick QC checks passed")


if __name__ == '__main__':
    main()
