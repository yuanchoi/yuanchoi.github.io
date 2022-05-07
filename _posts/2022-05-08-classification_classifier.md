# 요리 분류기1

이과정에서는 지난 수업에 저장한 모든 음식에 대한 균형 잡힌 정제된 데이터로 가득 찬 데이터셋을 사용한다.
이 데이터셋을 다양한 분류기와 함께 사용하여 재료 그룹을 기반으로 특정 국가 음식을 예측할 수 있다.
이렇게 하는 동안 분류 작업에 알고리즘을 활용할 수 있는 몇가지 방법에 대해 자세히 살펴볼 수 있다.

# 준비
Intro를 끝냈다는 가정 하에 *cleaned_cuisine.csv* 파일이 root `/data`폴더에 꼭 저장되어있도록 하여라.

## 연습 - 국가 요리 예측하기

1. 이 과의 *notebook.ipynb* 폴더에서 작업하면서 해당 파일을 판다스 라이브러리와 함께 가져와라.
```python
import pandas as pd
cuisines_df = pd.read_csv("../data/cleaned_cuisines.csv")
cuisines_df.head()
```

2. 이제 더 많은 라이브러리를 임포트하라
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
from sklearn.svm import SVC
import numpy as np
```

3. 훈련을 위해 X, Y 좌표를 두 개의 데이터프레임으로 나눈다. 요리는 라벨 데이터 프레임이 될 수 있다.
```python
cuisines_label_df = cuisines_df['cuisine']
cuisines_label_df.head()
```
결과:
```
0    indian
1    indian
2    indian
3    indian
4    indian
Name: cuisine, dtype: object
```

4. `Unnamed:0`열과 `cuisine`열에서 `drop()`을 호출하라. 나머지 데이터를 교육가능한 기능으로 저장하여라.
```python
cuisines_feature_df = cuisines_df.drop(['Unnamed: 0', 'cuisine'], axis=1)
cuisines_feature_df.head()
```

이제 모델을 훈련시킬 준비가 되었다!

## 분류기 선택

이제 데이터가 정제되었고 훈련 준비가 되었으므로, 작업에 사용할 알고리즘을 결정해야 한다.
사이킷런 그룹은 "지도학습"에서 분류되며, 이 범주엑서 분류할 수 있는 여러가지 방법을 찾을 수 있다. 이 품종은 처음 보면 꽤 당황스럽다. 다음 방법에는 모든 분류기법이 포함된다.
-선형 모형
-서포트벡터머신
-확률적 경사 하강
-가장 가까운 이웃
-가우스 프로세스
-의사결정 트리
-앙상블 방법
-다중 클래스 및 다중 출력 알고리즘

### 어떤 분류기로 할까?
그렇다면 우리는 어떤 분류기를 선택해야 하는가? 종종, 여러 개를 훑어보고 좋은 결과를 찾는 것이 테스트를 하는 방법이다. 사이킷런은 생성된 데이터셋에 대해 KNeighbors, SVC two ways, 가우시안 프로세스 분류기, 의사결정트리분류기, 랜덤포레스트 분류기, MLPC 분류기, AdaBoost분류기, 가우시안NB를 나란히 비교하여 시각화된 결론을 보여준다.

![comparison](https://user-images.githubusercontent.com/79850142/167268403-c2c99a57-ed8d-43f6-b151-094627e20846.png)

### 좀 더 나은 접근
그러나 마구 추측하는 것보다 더 나음 방법은 다운로드 가능한 ML Cheat 시트의 아이디어를 따르는 것이다. 여기서, 우리는 다중 클래스 문제에 대해 몇 가지 선택사항이 있다는 것을 발견한다.

<img width="364" alt="cheatsheet" src="https://user-images.githubusercontent.com/79850142/167268456-eda70afd-af39-4180-9c1d-4c9b11c7600b.png">

## 추리
우리가 가지고 있는 제약 조건들을 고려할 때, 우리가 다른 접근 방식들을 통해 우리의 길을 추론할 수 있는지를 알아보자:

-신경망은 너무 무겁다. 깨끗하지만 최소한의 데이터셋과 노트북을 통해 로컬로 교육을 실행하고 있다는 사실을 감안할 때 신경망은 이 작업을 하기에 너무 무겁다.

-2등급 분류기가 없다. 우리는 2등급 분류기를 사용하지 않기 때문에, 그것은 일대다를 배제한다.

-의사결정트리 또는 로지스틱 회귀 분석이 작동할 수 있따. 의사결정 트리가 작동하거나 다중 클래스 데이터에 대해 로지스틱 회귀분석을 수행할 수 있다.

-멀티클래스 Boosted Decision Tree는 다른 문제를 해결한다. 멀티클래스 부스트 결정 트리는 예를 들어 순위를 구축하도록 설계된 작업과 같은 비모수 작업에 가장 적합하므로 우리에게 유용하지 않다.

## 사이킷런 사용하기
우리는 사이킷런을 사용하여 데이터를 분석할 것이다. 그러나 사이킷런에서는 로지스틱 회귀 분석을 사용하는 여러가지 방법이 있다. 전달할 매개변수를 살펴보자.

기본적으로 사이킷런에게 로지스틱 회귀분석을 수행하도록 요청할 떄 지정해야 하는 `multi_class`와 `solver`라는 두 가지 중요한 매개변수가 있다. `multi_class`값은 특정 동작을 적용한다. 해결사의 값은 사용할 알고리즘이다. 모든 해결사가 `multi_class` 값과 쌍으로 구성할 수 있는 것은 아니다.

문서에 따르면, 멀티클래스 사례에서 훈련 알고리즘은 다음과 같다:

-`multi_class` 옵션이 `overr`로 설정된 경우 일대다 체계를 사용한다

-`multi_class` 옵션이 `multinomial`로 설정되었을 경우 교차 엔트로피 손실을 사용한다.

사이킷런은 해결사가 다양한 종류의 데이터 구조에서 나타나는 다양한 문제를 처리하는 방법을 설명하는 이 표를 제공한다.

<img width="765" alt="solvers" src="https://user-images.githubusercontent.com/79850142/167268741-3ce6d74e-2784-4ece-8283-8475a4b0d42e.png">

## 연습 - 데이터 분리하기
최근 이전 수업에서 후자에 대해 알게 된 이후 첫번째 트레이닝을 위해 로지스틱 회귀 분석에 초점을 맞출 수 있다. `train_test_split()`를 호출하여 데이터를 트레이닝 및 테스트 그룹으로 나눈다.

```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## 연습 - 로지스틱 회귀 적용하기
멀티 클래스 케이스를 사용 중이므로 사용할 *scheme*과 설정할 해결사를 선택해야 한다. 다중 클래스 설정 및 **liblinear** 해결사와 함께 로지스틱 회귀 분석을 사용하여 훈련한다.

1. `multi_class`를 `overr`로 설정하고 해결사를 `liblinear`로 설정한 로지스틱 회귀 분석을 만든다

```python
lr = LogisticRegression(multi_class='ovr',solver='liblinear')
model = lr.fit(X_train, np.ravel(y_train))

accuracy = model.score(X_test, y_test)
print ("Accuracy is {}".format(accuracy))
```
✅ 기본값으로 주로 설정되어있는 `lbfgs`와 같은 해결사를 시도해보아라

결과값:
```
ingredients: Index(['cilantro', 'onion', 'pea', 'potato', 'tomato', 'vegetable_oil'], dtype='object')
cuisine: indian
```

✅ 다른 열의 수를 선택하여 결과를 확인해보아라

3. 더 자세히 살펴보면 다음과 같은 예측의 정확성을 확인할 수 있다.

```python
test= X_test.iloc[50].values.reshape(-1, 1).T
proba = model.predict_proba(test)
classes = model.classes_
resultdf = pd.DataFrame(data=proba, columns=classes)
topPrediction = resultdf.T.sort_values(by=[0], ascending = [False])
topPrediction.head()
```

결과 값이 나왔다 - 인도요리가 가장 좋은 확률을 가지고 있다.

Indian : 0.715851</br>
Chinese : 0.229475</br>
Japanese : 0.029763</br>
Korean : 0.017277</br>
Thai : 0.007634</br>

✅ 왜 이 모델이 인도요리인지 확신하는지 이유를 설명할 수 있는가?

4. 회귀 분석 수업에서 했던 것처럼 분류 보고서를 인쇄하여 더 자세히 살펴보아라.

```python
y_pred = model.predict(X_test)
print(classification_report(y_test,y_pred))
```


