import httpx
from services.api.app.config import settings

async def run_python_code(code: str) -> str:
    """
    Tool: Python Code Interpreter.
    Executes Python code in a secure, isolated sandbox.
    Use this for math, data analysis, or complex logic.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                settings.SANDBOX_URL,
                json={"code": code, "timeout": 5},
                # Must exceed runner.py's own worst-case response time
                # (soft_timeout_seconds=8 in limits.yaml + up to 2s grace
                # period for SIGTERM before SIGKILL = ~10s), or the
                # CLIENT aborts first with a raw connection-timeout
                # exception instead of letting the server's own graceful
                # timeout handling return a clean 408 — found live while
                # fixing the sandbox's missing dependencies.
                timeout=12.0
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "success":
                    return f"Output:\n{data['output']}"
                else:
                    return f"Execution Error:\n{data['output']}"
            else:
                return f"Sandbox Error: Status {response.status_code}"
                
    except Exception as e:
        return f"Sandbox Connection Failed: {str(e)}"