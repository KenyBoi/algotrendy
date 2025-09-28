import sys
from pathlib import Path

# Ensure repository root is on sys.path so frontend.app can be imported when running this script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient
from frontend.app import app

client = TestClient(app)

print('Checking /api/spa/info...')
resp = client.get('/api/spa/info')
print('status', resp.status_code, 'body:', resp.json())
if resp.status_code != 200:
    raise SystemExit(2)

print('Checking /api/proxy/positions/FAKE/close...')
resp2 = client.post('/api/proxy/positions/FAKE/close')
print('status', resp2.status_code, 'body:', resp2.json())
if resp2.status_code != 200:
    raise SystemExit(3)

print('Checking /api/proxy/positions/FAKE/reopen...')
resp3 = client.post('/api/proxy/positions/FAKE/reopen')
print('status', resp3.status_code, 'body:', resp3.json())
if resp3.status_code != 200:
    raise SystemExit(4)

print('All checks passed')
