FROM python:3.8-slim

ENV PYTHONPATH="/tmp/myapp"

RUN apt update
RUN apt install -y curl
RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
RUN apt install -y nodejs


# setup poetry
RUN pip install poetry
WORKDIR /tmp/
COPY pyproject.toml .
RUN poetry config virtualenvs.create false
RUN poetry install --no-root


RUN jupyter labextension install jupyterlab-plotly@4.14.3

CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
