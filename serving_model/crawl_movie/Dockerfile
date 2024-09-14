FROM python:3.10

RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime

ADD requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt

ADD movie_id2tmdb_id.json /root/movie_id2tmdb_id.json
ADD crawl_serving.py /root/crawl_serving.py

WORKDIR /root

ENTRYPOINT ["python3", "crawl_serving.py"]