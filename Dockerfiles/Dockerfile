FROM python:3.10
COPY . /app
WORKDIR /app
RUN python3 -m pip freeze > requirements.txt
CMD poetry run python API.py 