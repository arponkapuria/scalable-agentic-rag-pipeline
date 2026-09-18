# 1. Use minimal base image to reduce attack surface
FROM python:3.10-slim

# 2. Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 3. Create a non-root user (Security Best Practice)
RUN useradd -m -u 1000 sandbox_user

# 4. Install dependencies
WORKDIR /app
COPY runner.py .
COPY limits.yaml .
RUN pip install flask pyyaml

# 5. Switch to non-root user
USER sandbox_user

# 6. Network Policy
# Note: In K8s, we will apply a NetworkPolicy to deny all Egress traffic
# except returning the HTTP response to the API.

# 7. Healthcheck
# Without this, Kubernetes might route traffic to the sandbox before Flask has finished starting up,
# causing failed requests on cold starts.
# Uses Python's stdlib, not curl — python:3.10-slim doesn't include curl,
# which would have made this healthcheck always fail (found live while
# fixing this Dockerfile's other missing dependencies). Adding curl just
# to run curl would also work against this file's own "minimal base
# image to reduce attack surface" goal above.
HEALTHCHECK --interval=30s --timeout=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')" || exit 1

# 8. Entrypoint
EXPOSE 8080
CMD ["python", "runner.py"]