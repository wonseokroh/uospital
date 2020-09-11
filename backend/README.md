# uospital-backend

## Tech stack

* Python 3.7+
* Django Rest Framework

## How to run (Local)

```console
$ python --version
Python 3.7..
$ virtualenv venv
$ source venv/bin/activate
(venv) $ pip install -r requirements.txt
(venv) $ python manage.py runserver 0.0.0.0:8000
```

## How to run (Docker)

```console
$ docker build .
// 로컬에 빌드되어 만들어진 image의 id를 docker-compose.yml 파일의 backend -> image 값에 넣는다
$ docker-compose up
```
