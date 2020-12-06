FROM python:3.8-slim
RUN pip install poetry
WORKDIR /tmp/myapp
COPY pyproject.toml poetry.lock .
RUN poetry config virtualenvs.create false
RUN poetry install --no-root
# Copy in everything else:
# COPY . .
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]