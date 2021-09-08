FROM python:3.8-slim

ENV PYTHONPATH="/tmp/myapp"

# RUN apt update
# RUN apt install -y curl
# RUN curl -sL https://deb.nodesource.com/setup_12.x | bash -
# RUN apt install -y nodejs

WORKDIR /tmp/
# setup python
COPY requirements.txt .
RUN pip install -r requirements.txt
# for Plotly
# RUN jupyter labextension install jupyterlab-plotly@4.14.3

CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
