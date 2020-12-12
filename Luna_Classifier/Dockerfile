FROM python:3.8-slim

ENV PYTHONPATH="/tmp/myapp"

# setup poetry
RUN pip install poetry
WORKDIR /tmp/
COPY pyproject.toml poetry.lock .
RUN poetry config virtualenvs.create false
RUN poetry install --no-root

CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]