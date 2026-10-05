"""
Locust load test for POST /api/v1/chat/stream. Each simulated user first calls /session/init to get a real session cookie (the app's actual auth — no bearer token), then streams chat requests using it, same as a real browser session would.

Usage: locust -f scripts/load_test.py --host http://localhost:8000
"""
from locust import HttpUser, task, between


class RAGUser(HttpUser):
    wait_time = between(1, 5)

    def on_start(self):
        """Mints a session cookie once per simulated user — requests.Session (which Locust's client wraps) persists it automatically for every later request from this user."""
        self.client.post("/api/v1/session/init")

    @task
    def chat_stream_task(self):
        """Sends one chat message and streams the NDJSON response, same as the real frontend does."""
        payload = {"message": "What is the warranty policy for the new X1 processor?"}

        with self.client.post(
            "/api/v1/chat/stream",
            json=payload,
            stream=True,
            name="/chat/stream",
        ) as response:
            if response.status_code != 200:
                response.failure(f"Unexpected status: {response.status_code}")
            else:
                for line in response.iter_lines():
                    pass  # drains the stream; a real test could validate each NDJSON line here
                response.success()