FROM python:3.8.16
WORKDIR ./rllte
ADD . .
RUN pip install -r requirements.txt
CMD ["python", "test.py"]