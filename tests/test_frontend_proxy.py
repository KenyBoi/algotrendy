from fastapi.testclient import TestClient
from frontend.app import app

client = TestClient(app)


def test_spa_info():
    resp = client.get('/api/spa/info')
    assert resp.status_code == 200
    data = resp.json()
    assert 'frontend' in data


def test_proxy_close_and_reopen():
    # Test the close proxy endpoint
    resp = client.post('/api/proxy/positions/FAKE/close')
    assert resp.status_code == 200
    data = resp.json()
    assert data.get('symbol') == 'FAKE'

    # Test the reopen proxy endpoint
    resp2 = client.post('/api/proxy/positions/FAKE/reopen')
    assert resp2.status_code == 200
    data2 = resp2.json()
    assert data2.get('symbol') == 'FAKE'