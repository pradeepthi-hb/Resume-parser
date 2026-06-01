import unittest

from flask import Flask
from src.utils.api_routes import register_routes


class TestApiStability(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app = Flask(__name__)
        register_routes(app)
        app.config['TESTING'] = True
        cls.client = app.test_client()

    def test_smoke_learning_stats(self):
        learning_resp = self.client.get('/api/learning/stats')
        self.assertEqual(learning_resp.status_code, 200)

    def test_removed_runtime_map_endpoints_return_404(self):
        removed_endpoints = [
            ('GET', '/api/learning/runtime-map'),
            ('GET', '/api/learning/runtime-map/snapshots'),
            ('POST', '/api/learning/runtime-map/snapshot'),
            ('POST', '/api/learning/runtime-map/activate/test-snapshot'),
            ('POST', '/api/learning/runtime-map/rollback/test-snapshot'),
            ('GET', '/api/learning/runtime-map/snapshot/test-snapshot'),
            ('POST', '/api/learning/runtime-map/reset'),
        ]

        for method, path in removed_endpoints:
            with self.subTest(method=method, path=path):
                if method == 'GET':
                    response = self.client.get(path)
                elif method == 'POST':
                    response = self.client.post(path, json={})
                else:
                    self.fail(f'Unexpected method in test table: {method}')
                self.assertEqual(response.status_code, 404)


if __name__ == '__main__':
    unittest.main()
