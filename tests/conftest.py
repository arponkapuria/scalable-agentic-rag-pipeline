# Pytest bootstrap: force-sets the env vars that config.py's Settings() needs at import time (DB, Redis, S3, Groq, OpenRouter). Uses direct assignment, not setdefault(), so a real .env or shell values can never leak in and tests stay hermetic.

import os

os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost:5432/test"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["S3_BUCKET_NAME"] = "test-bucket"
os.environ["GROQ_API_KEY"] = "test-groq-key"
os.environ["GROQ_MODELS"] = "model-a,model-b"
os.environ["OPENROUTER_API_KEY"] = "test-or-key"
os.environ["OPENROUTER_MODELS"] = "or-model-a,or-model-b"