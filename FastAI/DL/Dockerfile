FROM python:3.8-slim

ENV PYTHONPATH="/tmp/myapp"

# setup poetry
WORKDIR /tmp/
COPY requirements.txt .
RUN pip install -r requirements.txt

RUN jupyter serverextension enable --sys-prefix voila 

CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--allow-root"]
