FROM python:3
WORKDIR /usr/src/app

ADD requirements.txt /usr/src/app
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python", "./predictions.py"]
	