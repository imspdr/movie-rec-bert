FROM python:3.10

RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime

ADD requirements.txt /root/requirements.txt
RUN pip install -r /root/requirements.txt

ADD bert4rec /root/bert4rec
ADD bert4rec_model /root/bert4rec_model
ADD index2item.json /root/index2item.json
ADD item2index.json /root/item2index.json
ADD bert_serving.py /root/bert_serving.py

WORKDIR /root

ENTRYPOINT ["python3", "bert_serving.py"]