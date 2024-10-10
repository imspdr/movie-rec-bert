# movie-rec-bert

## Structure

![Structure](/res/movieRecBert.png)

## Description

Kserve의 predictor를 이용한 serverless program. 

Nginx 및 React로 구성된 프론트엔드와 2개의 Kserve inference service crd로 구성된다.

첫 번째 inference service는 Bert4Rec 모델을 이용하여 입력된 영화 id sequence를 이용해 해당 sequence 다음에 올 가장 적합한 영화 id를 순서대로 k개 반환하는 기능을 제공한다. (/serving_models/bert)

두 번째 inference service는 tmdb의 영화 정보를 크롤링하여 영화 포스터, 제목, 개봉일, 간단한 설명, 태그 등을 제공한다. (/serving_models/crawl_movie)


## Install

```commandline
$ cd helm
$ kubectl create ns movie-rec-bert
$ helm upgrade --install movie-rec-bert movie-rec-bert
```
- kubernetes 환경에 helm chart를 이용하여 배포할 수 있다.
- 배포 전 kserve 및 관련 모듈을 설치해야한다. quick_install.sh script로 설치 가능하다. (kserve github 참고)
- 테스트 버전
  - kubernetes: v1.30.0
  - minikube: v1.33.1
  - kserve: 0.13.0
  - istio: 1.20.4
  - knative serving: v1.13.1
  - cert manager: v1.9.0
- kserve 0.7 이상의 버전에서 작동하지만 관련 모듈 간의 의존성은 확인이 필요하다.
- docker.io registry에 저장된 이미지를 이용한다. 
  - 이미지는 각 폴더의 build_push.sh 혹은 build_docker.sh script를 이용해 저장한다.
- istio, minikube 버전에 따라 helm chart values.yaml 수정이 필요하다
  - 현재 작성된 values는 테스트 버전 기준

# Example

![Sample](/res/movierecbert.gif)

# Bert

BERT4Rec은 양방향 sequence 추론 모델인 BERT를 아이템 추천에 이용하는 모델로 사용자가 아이템을 선택하는 순서를 보고 다음에 올 가장 적합한 아이템을 추론하는 모델이다.
기존 sequential recommendation이 단방향으로 학습하여 갖는 한계점을 극복한 모델이라고 한다.
단순히 sequence 다음에 올 아이템을 학습하는 것이 아닌 주어진 sequence에 랜덤하게 데이터를 마스크하여 중간 지점을 추론하도록 학습한다. 

학습 데이터는 movie lens (https://grouplens.org/datasets/movielens/) 중 ml-latest-small (2024년 9월 기준) 데이터를 이용한다. 
이중 rating 데이터를 이용해 사용자가 4점 이상 점수를 준 interaction 정보만을 이용해 학습을 진행했으며, 최소 3명의 사용자에게는 평가받은 영화만을 이용했다.
이 때 사용자별 평가한 영화의 평균 길이가 70을 조금 넘어서 BERT의 input sequece max length를 70으로 설정하여 학습시켰다.

모델을 구성하는 코드는 (https://github.com/zanussbaum/BERT4Rec-VAE-Pytorch) 를 이용했으며, 전처리 작업과 데이터를 다루는 부분을 간단하게 구현하여 사용했다.
학습과 관련된 코드는 /serving_models/bert/train_ml_small.py 에서 확인할 수 있다.


# Future Work

추천 모델을 학습시켜서 배포한 후 서비스에 이용하는 것까지는 만족스럽게 진행됐다.
다만 추천 모델의 성격 상 추론 결과에 대한 평가가 어렵고 모델에 대한 이해도도 충분치 않아 기능이 얼마나 잘 작동하는 지에 대한 평가가 어려운 것 같다.
모델에 대한 이해도를 높여 최적화를 진행하면 더 완성도를 높일 수 있을 것 같다.
content based / user based recommendation 등 다양한 형태의 추천 모델을 추가하여 추천된 영화를 비교해보는 것도 재밌는 작업이 될 것 같다. 
