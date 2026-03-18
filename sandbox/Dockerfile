FROM python:3.12-slim

WORKDIR /sandbox

# Install common testing dependencies
RUN pip install --no-cache-dir \
    pytest \
    pytest-asyncio \
    httpx \
    fastapi \
    pydantic \
    requests

# Security: run as non-root
RUN useradd -m sandboxuser
USER sandboxuser

# Tests are mounted at runtime via -v
CMD ["python", "-m", "pytest", "-v", "--tb=short", "/sandbox/tests"]
