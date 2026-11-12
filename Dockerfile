FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE CHANGELOG.md ./
COPY src ./src
COPY configs ./configs

# kernel without torch; for writers and Hugging Face: pip install ".[torch]"
RUN pip install --no-cache-dir .

# The assembly is considered successful only with green smoke (safety-first).)
RUN atomicng smoke > /dev/null

ENTRYPOINT ["atomicng"]
CMD ["--help"]
