"""
Code sandbox tool: sends Python code to the isolated sandbox service for execution and returns its output.
"""
import httpx
from services.api.app.config import settings


async def run_python_code(code: str) -> str:
    """Executes Python code in the isolated sandbox service.

    Args:
        code: The Python code to run.

    Returns:
        The formatted output, execution error, or a connection-failure message.
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                settings.SANDBOX_URL,
                json={"code": code, "timeout": 5},
                timeout=12.0  # exceeds the server's own worst-case response time, for a clean timeout instead of an abrupt one
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