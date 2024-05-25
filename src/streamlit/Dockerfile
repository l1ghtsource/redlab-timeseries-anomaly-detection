FROM python:3.11.7-alpine


RUN apk add make cmake g++ gcc zlib-dev
#RUN python3 -m pip install --upgrade pip

COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

COPY . . 

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]