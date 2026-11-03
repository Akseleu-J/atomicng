FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE CHANGELOG.md ./
COPY src ./src
COPY configs ./configs

# Ядро без torch; для writer'ов/HF: pip install ".[torch]"
RUN pip install --no-cache-dir .

# Сборка считается успешной только при зелёном smoke (безопасность-first)
RUN atomicng smoke > /dev/null

ENTRYPOINT ["atomicng"]
CMD ["--help"]
