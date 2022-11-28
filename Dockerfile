FROM python:3.8

ARG VERSION

LABEL org.label-schema.version=$VERSION

COPY ./requirements.txt /webapp_flask/requirements.txt

WORKDIR /webapp_flask
RUN pip install -r requirements.txt
COPY ./webapp_flask/app.py /webapp_flask/app.py
COPY ./webapp_flask/saved_model/* /webapp_flask/saved_model/
COPY webapp_flask/templates/* /webapp_flask/templates/
COPY webapp_flask/uploads/* /webapp_flask/uploads/
EXPOSE 5000 5050
ENTRYPOINT [ "python" ]
CMD [ "app.py" ]