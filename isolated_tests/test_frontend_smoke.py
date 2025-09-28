import pytest
from pathlib import Path
import sys
from fastapi.testclient import TestClient

# Ensure repository root is on sys.path so we can import frontend.app
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import the frontend app directly (this should avoid heavy ML imports)
from frontend.app import app

client = TestClient(app)


def test_spa_info():
    resp = client.get('/api/spa/info')
    assert resp.status_code == 200
    data = resp.json()
    assert 'frontend' in data
    assert 'proxy_prefix' in data


def test_proxy_close_reopen():
    resp = client.post('/api/proxy/positions/FAKE/close')
    assert resp.status_code == 200
    body = resp.json()
    assert body.get('symbol') == 'FAKE'

    resp2 = client.post('/api/proxy/positions/FAKE/reopen')
    assert resp2.status_code == 200
    body2 = resp2.json()
    assert body2.get('symbol') == 'FAKE'
