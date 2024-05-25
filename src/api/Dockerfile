FROM python:3.11.7-alpine


RUN apk add make cmake g++ gcc zlib-dev
RUN python3 -m pip install --upgrade pip
COPY requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY . . 
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]