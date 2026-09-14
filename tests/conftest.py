import os

# config.py's Settings() instantiates at import time and requires these,
# AND loads the repo's real .env file if present (env_file=".env" in
# Settings.Config). pydantic-settings' precedence is os.environ > .env
# file > field defaults — so setdefault() here is not enough on a machine
# with a real .env populated (every dev machine, from Phase 0 onward):
# the .env file's real values would still win over an unset os.environ
# var. Force-assign instead, so tests are hermetic regardless of what's
# in .env or the shell.
os.environ["DATABASE_URL"] = "postgresql+asyncpg://test:test@localhost:5432/test"
os.environ["REDIS_URL"] = "redis://localhost:6379/0"
os.environ["S3_BUCKET_NAME"] = "test-bucket"
os.environ["GROQ_API_KEY"] = "test-groq-key"
os.environ["GROQ_MODELS"] = "model-a,model-b"
os.environ["OPENROUTER_API_KEY"] = "test-or-key"
os.environ["OPENROUTER_MODELS"] = "or-model-a,or-model-b"