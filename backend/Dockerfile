FROM python:3.7
ENV PYTHONUNBUFFERED 1

# 메인테이너 정보 (옵션)
MAINTAINER WSROH <wonseok96@uos.ac.kr>

RUN pip install pip --upgrade

WORKDIR /home/app

COPY . /home/app

#RUN apk update \
#    && apk add --virtual build-deps gcc python3-dev musl-dev \
#    && apk add --no-cache mariadb-dev
#RUN apk update \
#    && apk add --upgrade --no-cache \
#    bash openssh curl ca-certificates openssl less htop \
#    g++ make wget rsync \
#    build-base libpng-dev freetype-dev libexecinfo-dev openblas-dev libgomp lapack-dev \
#    libgcc libquadmath musl  \
#    libgfortran \
#    lapack-dev
RUN pip install -r /home/app/requirements.txt
#RUN apk del build-deps
CMD python manage.py runserver 0.0.0.0:8000
