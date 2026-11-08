FROM python:3.12-slim

# Non-root user.
RUN useradd --create-home --uid 10001 runner

WORKDIR /app

# Install dependencies first for layer caching.
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

COPY --chown=runner:runner . .

USER runner

# Default: print the benchmark summary without persisting anything.
ENTRYPOINT ["fsmckpt"]
CMD ["bench", "--no-write"]
