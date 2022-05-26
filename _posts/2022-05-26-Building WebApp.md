# 머신러닝 모델을 사용하여 WebApp 만들기
이번에는 이 세상에 없었던 데이터셋에 대한 머신러닝 모델을 훈련할 예정이다.(NUFORC의 데이터베이스트를 참고) 그리고 훈련된 모델을 'pickle'하고 Flask앱에서 모델을 사용하게 된다. 이 때 Flask와 Pickle이 무엇인지를 알아보자.

✅ **Flask**: 'micro-framework'로 정의한 Flask는 파이썬으로 웹프레임워크의 기본적인 기능과 웹페이지를 만드는 템플릿 엔진을 제공한다.

✅ **Pickle**: Pickle은 파이썬 객체구조를 serialize와 de-serialize하는 파이썬 모듈이다. 모델을 pickle하게 되면, 웹에서 쓰기 위해 serialize 또는 flatten한다. pickle된 파일은 `.pkl` 확장자를 가지고 있다.

## 연습하기 - 데이터 정리
NUFORC에서 모아둔 8만개의 UFO 목격 데이터를 사용할 예정이다. 데이터에는 UFO 목격과 관련한 길고 짧은 다양한 설명이 있다.

*   "A man emerges from a beam of light that shines on a grassy field at night and he runs towards the Texas Instruments parking lot"
-> 한 남자가 밤에 풀밭을 비추는 한 줄기 빛으로부터 출현하고, 그는 텍사스 instrument 주차장으로 달려간다.

*   "the lights chased us"
-> 빛이 우리를 쫓아왔다

ufos.csv 스프레드시트에는 UFO가 목격된 `city`, `state`와 `country`, UFO의 `shape`, `latitude`, `longitude` 열이 포함되어 있다.

1. `pandas`, `matplotlib`,`numpy`를 import하고 ufos.csv도 import한다.

```import pandas as pd
 import numpy as np
 ufos = pd.read_csv('./data/ufos.csv')
 ufos.head()


