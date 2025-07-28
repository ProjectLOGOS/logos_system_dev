import pytest
import requests
from grounding.perception_connectors.web import fetch_json
from grounding.perception_connectors.api import APIClient
from grounding.perception_connectors.files import read_csv, read_json

class DummyResponse:
    def __init__(self, json_data, status_code=200):
        self._json = json_data
        self.status_code = status_code
    def raise_for_status(self):
        if self.status_code != 200:
            raise requests.HTTPError()
    def json(self):
        return self._json

@pytest.fixture(autouse=True)
def mock_requests(monkeypatch):
    def dummy_get(url, params=None, headers=None):
        return DummyResponse({'key': 'value'})
    monkeypatch.setattr(requests, 'get', dummy_get)
    monkeypatch.setattr(requests, 'post', lambda url, headers=None, json=None: DummyResponse(json))

def test_fetch_json():
    data = fetch_json('http://example.com')
    assert data == {'key': 'value'}

def test_api_client():
    client = APIClient('http://api.test', api_key='abc')
    resp = client.get('/path')
    assert resp == {'key': 'value'}
    resp2 = client.post('/path', payload={'a':1})
    assert resp2 == {'a':1}

def test_read_csv(tmp_path):
    file = tmp_path / 'test.csv'
    file.write_text('a,b
1,2
3,4')
    rows = read_csv(str(file))
    assert rows == [{'a':'1','b':'2'},{'a':'3','b':'4'}]

def test_read_json(tmp_path):
    file = tmp_path / 'test.json'
    file.write_text('{"x":10}')
    obj = read_json(str(file))
    assert obj == {'x':10}
