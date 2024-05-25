FROM python:3.11.7-slim


#RUN apk add make cmake g++ gcc zlib-dev
RUN python3 -m pip install --upgrade pip
COPY requirements.txt requirements.txt
#RUN pip install setuptools==58
RUN pip install -r requirements.txt

COPY . . 
CMD ["faststream", "run", "--workers", "1", "main:app"]