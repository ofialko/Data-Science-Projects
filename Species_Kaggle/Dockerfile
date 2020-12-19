FROM python:3.8-slim

RUN apt-get update
RUN apt -y install npm
RUN apt -y install nodejs


ENV PYTHONPATH="/tmp/myapp"
# setup poetry
RUN pip install poetry
WORKDIR /tmp/
COPY pyproject.toml .

RUN poetry config virtualenvs.create false
RUN poetry install --no-root
RUN jupyter labextension install @aquirdturtle/collapsible_headings

CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]