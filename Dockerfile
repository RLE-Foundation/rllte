FROM python:3.8.16
WORKDIR ./rllte
ADD . .
RUN pip install -e .[envs]
CMD ["python", "test.py"]