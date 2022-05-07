# 분류모델 만들기: 맛있는 아시아, 인도 요리
## 분류에 대한 소개: 정제, 전처리, 데이터 시각화하기

이번 네개의 레슨을 통해서, 고전적인 머신러닝의 기본적인 요점 - 분류에 대해 배울 것이다. </br>
아시아와 인도의 요리에 대한 데이터셋과 다양한 분류, 알고리즘을 사용할 예정이다.</br>

분류는 회귀기술과 많은 공통점이 있는 지도학습의 한 형태이다. 머신러닝이 데이터셋을 사용하여 값이나 이름을 예측하는 것이라면, 분류는 일반적으로 이진부류와 다중 클레스 분류, 이 두가지 종류로 나뉜다.

기억할 것:

- 선형회귀분석을 사용하면 변수 간의 관계를 예측하고 해당 선과 관련하여 세 데이터 점이 포함될 위치를 정확하게 예측할 수 있다. 그래서 예를 들어, 9월과 12월의 호박가격이 얼마인지를 예측할 수 있다.</br> 
- 로지스틱 회귀 분석을 통해 이진범주를 발견할 수 있다. 이 가격대에서 호박은 과연 주황색인가, 아니면 주황색이 아닌것인가?

분류는 다양한 알고리즘을 사용하여 데이터 포인트의 레이블이나 클래스를 결정하는 다른 방법을 결정한다. 이 요리 데이터를 사용하여 재료 그룹을 관찰함으로써 음식의 유래를 결정할 수 있는 지를 알아볼 것이다.

## 강의 전 퀴즈
### 인트로
분류는 머신러닝 연구자와 데이터 사이언티스트의 기본적인 활동 중 하나이다. 이진 값의 기본분류인("이 이메일은 스팸인가?")에서 컴퓨터 비전을 사용한 복잡한 이미지 분류 및 분할에 이르기까지 데이터를 클래스로 정렬하고 그에 대한 질문을 할 수 있으면 항상 유용하다.

보다 과학적인 방법으로 공정을 설명하기 위해, 분류방법에서는 입력변수와 출력변수 사이의 관계를 매핑할 수 있는 예측 모형을 만든다.

데이터를 정리하고 시각화하고 머신러닝 작업에 대비하는 프로세스를 시작하기 전에, 머신러닝을 활용하여 데이터를 분류하는 다양한 방법에 대해 조금 알아볼 예정이다.
통계에서 파생된 고전적인 머신러닝을 사용한 분류는 `흡연자`, `체중`, `나이`와 같은 특징을 사용하여 *X질환의 발병 가능성*을 결정한다. 앞서 수행한 회귀 연습과 유사한 지도학습 기법으로, 데이터에 레이블이 지정되고 ML알고리즘은 이러한 레이블을 사용하여 데이터 세트의 클래스를 분류하고 예측하여 그룹 또는 결과에 할당한다.

✅ 요리에 대한 데이터세트를 잠시 생각해보자. 멀티클래스 모델이 대답할 수 있는 것은? 2진수 모델은 무엇을 답할 수 있을까? 만약 당신이 주어진 요리가 페누그릭을 사용할 가능성이 있는지 여부를 결정하기 원한다면? 만약 여러분이 스타 아니스, 아티초크, 콜리플라워, 그리고 고추냉이가 가득한 식료품 봉지를 선물로 받고 싶다면, 당신은 전형적인 인도요리를 만들어낼 수 있는가?

## 분류기
우리가 이 요리 데이터셋에 대해 묻고 싶은 질문은 다양한 질문이다. 우리는 몇 가지 잠재적인 국가의 요리를 다룰 수 있기 때문이다. 성분 배치가 주어졌을 때, 이 많은 등급 중 어떤 데이터가 적합한가?
Tidymodel(정리모형)은 해결하려는 문제의 종류에 따라 데이터를 분류하는 데 사용할 수 있는 여러가지 알고리즘을 제공한다. 다음 두 가지 수업에서는 이러한 알고리즘 중 몇가지에 대해 배우게 될 것이다.

## 연습 - 데이터를 정제 및 균형조정
이 프로젝트를 시작하기 전에 가장 먼저 해야할 일은 데이터를 정리하고 균형을 맞춰 더 나은 결과를 얻는 것이다. 이 폴더의 root에 있는 빈 *notebook.ipynb*부터 시작하자.
가장 먼저 설치해야하는 것은 imblearn이다. 이것은 사이킷런의 패키지로 데이터의 균형조정을 더 낫게 해주는 역할을 한다.

1. `imblearn`을 설치하기 위해서, `pip install`을 실행하여라
```python
pip install imblearn
```

2. 데이터를 가져오는 데 필요한 패키지를 가져오고 시각화할 수 있으며, `imblearn`에서 `SMOTE`를 import 할 수 있다.
```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from imblearn.over_sampling import SMOTE
```
이제 데이터 가져오기를 읽도록 설정되었다.

3. 다음 작업은 데이터를 가져오는 것이다.
```python
df  = pd.read_csv('../data/cuisines.csv')
```
`read_csv()`를 사용하는 것은 csv 파일 cuisine.csv의 내용을 읽고 변수 `df`에 배치한다.

4. 데이터의 배열을 확인하여라.
```python
df.head()
```
첫 5개의 열은 이렇게 생겼다.
```
|     | Unnamed: 0 | cuisine | almond | angelica | anise | anise_seed | apple | apple_brandy | apricot | armagnac | ... | whiskey | white_bread | white_wine | whole_grain_wheat_flour | wine | wood | yam | yeast | yogurt | zucchini |
| --- | ---------- | ------- | ------ | -------- | ----- | ---------- | ----- | ------------ | ------- | -------- | --- | ------- | ----------- | ---------- | ----------------------- | ---- | ---- | --- | ----- | ------ | -------- |
| 0   | 65         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 1   | 66         | indian  | 1      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 2   | 67         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 3   | 68         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 0      | 0        |
| 4   | 69         | indian  | 0      | 0        | 0     | 0          | 0     | 0            | 0       | 0        | ... | 0       | 0           | 0          | 0                       | 0    | 0    | 0   | 0     | 1      | 0        |
```
5. `info()`를 불러옴으로써 데이터에 대한 정보를 얻어오기
```python
df.info()
```

```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2448 entries, 0 to 2447
Columns: 385 entries, Unnamed: 0 to zucchini
dtypes: int64(384), object(1)
memory usage: 7.2+ MB
```

## 연습 - 요리에 대해 배우기
이제 점점 작업이 흥미로워지고 있다. 이제 요리당 데이터의 분포를 살펴보자.

1. `barh()`를 불러와서 데이터를 도표로 그리자.
```python
df.cuisine.value_counts().plot.barh()
```
![cuisine-dist](https://user-images.githubusercontent.com/79850142/167266675-07fafe9d-58aa-4bff-8d22-f75df3cedb0e.png)
한정된 수의 요리가 있지만, 데이터의 분포가 고르지 않다. 하지만 이를 고칠 수 있다. 그렇게 하기 전에 한번 더 살펴보자.

2. 요리당 얼마나 많은 데이터를 사용할 수 있는지 알아보고 프린트하여라.
```python
thai_df = df[(df.cuisine == "thai")]
japanese_df = df[(df.cuisine == "japanese")]
chinese_df = df[(df.cuisine == "chinese")]
indian_df = df[(df.cuisine == "indian")]
korean_df = df[(df.cuisine == "korean")]

print(f'thai df: {thai_df.shape}')
print(f'japanese df: {japanese_df.shape}')
print(f'chinese df: {chinese_df.shape}')
print(f'indian df: {indian_df.shape}')
print(f'korean df: {korean_df.shape}')
```
결과는 이러하다:
```python
thai df: (289, 385)
japanese df: (320, 385)
chinese df: (442, 385)
indian df: (598, 385)
korean df: (799, 385)
```

## 재료를 찾기
이제 당신은 데이터를 더 깊이 파고들어 요리 당 전형적인 재료가 무엇인지 배울 수 있다. 음식 사이에 혼란을 일으키는 반복적인 데이터를 지워야 하는데, 이 문제에 대해서 알아보도록 하자.

1. 파이썬에서 `create_increment()`함수를 만들어 성분 데이터 프레임을 만든다. 이 기능은 도움이 되지 않는 열을 버리는 것부터 시작하여 성분을 개수에 따라 정렬한다.
```python
def create_ingredient_df(df):
    ingredient_df = df.T.drop(['cuisine','Unnamed: 0']).sum(axis=1).to_frame('value')
    ingredient_df = ingredient_df[(ingredient_df.T != 0).any()]
    ingredient_df = ingredient_df.sort_values(by='value', ascending=False,
    inplace=False)
    return ingredient_df
```
이제 그 기능을 사용하여 요리 별로 가장 인기있는 10대 식재료에 대한 아이디어를 얻을 수 있다.

2. `create_interent()`를 호출하고 `barh()`를 호출하여 표시한다
```python
thai_ingredient_df = create_ingredient_df(thai_df)
thai_ingredient_df.head(10).plot.barh()
```
![thai](https://user-images.githubusercontent.com/79850142/167266867-5741667f-ee2c-4c22-b191-fb3f3879a974.png)

3. 일본 데이터에 대해서도 똑같이 하여라
```python
japanese_ingredient_df = create_ingredient_df(japanese_df)
japanese_ingredient_df.head(10).plot.barh()
```
![japanese](https://user-images.githubusercontent.com/79850142/167266890-c99a50bd-0f79-45bc-bd93-e2d7b48664c3.png)

4. 이제 중국재료에 대해서도 똑같이 하여라
```python
chinese_ingredient_df = create_ingredient_df(chinese_df)
chinese_ingredient_df.head(10).plot.barh()
```
![chinese](https://user-images.githubusercontent.com/79850142/167266912-5fc05a81-8acd-4d26-9cdb-9a3f85748586.png)

5. 인도 재료에 대해서도 그대로 하여라
```python
indian_ingredient_df = create_ingredient_df(indian_df)
indian_ingredient_df.head(10).plot.barh()
```
![indian](https://user-images.githubusercontent.com/79850142/167266930-cb976761-4e92-4351-a6d4-6354c6711459.png)

6. 마지막으로 한국 재료에 대해서도 똑같이 하여라.
```python
korean_ingredient_df = create_ingredient_df(korean_df)
korean_ingredient_df.head(10).plot.barh()
```
![korean](https://user-images.githubusercontent.com/79850142/167266950-9af99386-74e7-428e-8fd2-55c1fd607987.png)

7. 이제, `drop()`을 호출함으로써 요리 사이에 혼란을 일으키는 가장 일반적인 재료를 삭제한다.
```python
feature_df= df.drop(['cuisine','Unnamed: 0','rice','garlic','ginger'], axis=1)
labels_df = df.cuisine #.unique()
feature_df.head()
```

## 데이터셋을 균형조정하기
이제 데이터를 정리했으므로, SMOTE-"합성 소수 과표본 기법"을 사용하여 균형을 잡아라.

1. `fit_resample()`을 호출하면 이 전략은 보간을 통해 새 샘플을 생성한다.
```python
oversample = SMOTE()
transformed_feature_df, transformed_label_df = oversample.fit_resample(feature_df, labels_df)
```
데이터의 균형을 유지함으로써 데이터를 분류할 때 더 나음 결과를 얻을 수 있다. 이진 분류에 대해 생각해보아라. 대부분의 데이터가 하나의 클래스인 경우 ML 모델은 해당 클래스를 더 자주 예측한다. 단지 더 많은 데이터가 있기 때문이다. 데이터의 균형을 맞추려면 왜곡된 데이터가 필요하며 이러한 불균형을 제거하는 데 도움이 된다.

2. 이제 성분 당 레이블 수를 확인할 수 있다.
```python
print(f'new label count: {transformed_label_df.value_counts()}')
print(f'old label count: {df.cuisine.value_counts()}')
```
결과:
```
new label count: korean      799
chinese     799
indian      799
japanese    799
thai        799
Name: cuisine, dtype: int64
old label count: korean      799
indian      598
chinese     442
japanese    320
thai        289
Name: cuisine, dtype: int64
```
이 데이터는 깔끔하고, 균형도 잘 잡혀있고 맛있어 보인다!

3. 마지막 단계는 레이블 및 기능을 포함한 균형 잡힌 데이터를 파일로 내보낼 수 있는 새 데이터 프레임에 저장하는 것이다.
```python
transformed_df = pd.concat([transformed_label_df,transformed_feature_df],axis=1, join='outer')
```

4. `transformed_df.head()`및 `transformed_df.info()`를 사용하여 데이터를 한번 더 살펴볼 수 있다. 다음 수업에서 사용할 수 있도록 이 데이터 복사본을 저장한다.
```python
transformed_df.head()
transformed_df.info()
transformed_df.to_csv("../data/cleaned_cuisines.csv")
```






