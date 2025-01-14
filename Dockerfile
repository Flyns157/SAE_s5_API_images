FROM python:3.12.7

WORKDIR /app/

RUN python -m venv .venv

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./selfhost_stablediffusion_api /app/app

CMD "python3.12 -m app"
