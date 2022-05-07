# 요리 분류기 2
이 두번째 분류 과정에서는 숫자 데이터를 분류하는 더 많은 방법을 살펴볼 것이다. 또한 분류기를 다른 분류기로 선택하는 대 미치는 영향에 대해서도 배울 것이다.

## 분류 지도
이전에, 마이크로소프트의 Cheat Sheet를 사용하여 데이터를 분류할 떄 사용할 수 있는 다양한 옵션에 대해 배웠다. 사이킷런은 추정기를 더 좁히는 데 도움이 될 수 있는 유사하지만 보다 세분화된 Cheat Sheet를 제공한다.

![map](https://user-images.githubusercontent.com/79850142/167269253-621095eb-f1cf-4766-ab34-6eb44ddd0ec6.png)

### 계획
이 지도는 데이터를 명확하게 파악하면서 의사결정에 이르는 경로를 따라갈 수 있으므로 매우 유용하다

-우리는 50개 이상의 샘플을 가지고 있다</br>
-우리는 어떤 범주를 예측하고 싶다</br>
-레이블이 지정된 데이터가 있다</br>
-우리는 10만개 미만의 샘플을 가지고 있다</br>
-선형 SVC를 선택할 수 있다</br>
-만약 그게 안 된다면, 우리는 수치 데이터를 가지고 있기 때문에 KNeighbors 분류기를 사용해볼 수 있다</br>
-그래도 문제가 해결되지 않는다면 SVC 및 앙상블 분류기를 사용해 보아라</br>

## 연습 - 데이터를 분리하기

1. 우리는 몇개의 라이브러리를 임포트해야 한다.
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report, precision_recall_curve
import numpy as np
```

2. 트레이닝 데이터와 테스트 데이터로 나누어라
```python
X_train, X_test, y_train, y_test = train_test_split(cuisines_feature_df, cuisines_label_df, test_size=0.3)
```

## 회귀 SVC 분류기
SVC는 ML 기술인 Support-Vector Machine 제품군의 하위 제품이다. 이 방법에서는 'kernel'을 선택하여 레이블을 군집화하는 방법을 결정할 수 있다. 'C' 파라미터는 파라미터의 영향을 조절하는 '정규화'를 의미한다. 커널은 여러가지 중 하나일 수 있다. 여기서는 선형 SVC를 활용하도록 'linear'로 설정한다. 확률은 기본적으로 'false'로 설정되며, 여기서는 확률 추정치를 수집하기 위해 'true'로 설정한다. 우리는 확률값을 얻기 위한 데이터를 섞기 위해 무작위 상태를 '0'으로 설정했다.

### 연습 - 선형 SVC를 적용하라
분류기 배열을 만드는 것으로 시작하라. 테스트하는대로 이 어레이에 점진적으로 추가할 것이다.

1. 선형 SVC로 시작:
```python
C = 10
# Create different classifiers.
classifiers = {
    'Linear SVC': SVC(kernel='linear', C=C, probability=True,random_state=0)
}
```

2. 선형 SVC를 사용한 모델을 훈련시키고 레포트를 프린트해라
```python
n_classifiers = len(classifiers)

for index, (name, classifier) in enumerate(classifiers.items()):
    classifier.fit(X_train, np.ravel(y_train))

    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy (train) for %s: %0.1f%% " % (name, accuracy * 100))
    print(classification_report(y_test,y_pred))
```

결과가 꽤 좋다:
```python
Accuracy (train) for Linear SVC: 78.6% 
              precision    recall  f1-score   support

     chinese       0.71      0.67      0.69       242
      indian       0.88      0.86      0.87       234
    japanese       0.79      0.74      0.76       254
      korean       0.85      0.81      0.83       242
        thai       0.71      0.86      0.78       227

    accuracy                           0.79      1199
   macro avg       0.79      0.79      0.79      1199
weighted avg       0.79      0.79      0.79      1199
```

## KNeighbors 분류기
KNeighbors는 ML 방식의 "Neighbors"계열의 일부로, 지도학습과 비지도 학습 모두에 사용할 수 있다. 이 방법에서는 데이터에 대한 일반화된 레이블을 예측할 수 있도록 미리 정의된 수의 점이 생성되고 이러한 점 주위에 데이터가 수집된다.

### 연습 - KNeighbors 분류기 적용하기
이전의 분류기는 좋았고, 데이터도 잘 작동했지만, 아마도 우리는 더 나은 정확도를 얻을 수 있을 것이다. KNeighbors 분류기를 사용해보아라

1. 분류기 배열에 줄을 추가해라(Linear SVC 항목 뒤에 쉼표를 추가한다)
```python
'KNN classifier': KNeighborsClassifier(C),
```

결과는 더 악화가 되었다:
```python
Accuracy (train) for KNN classifier: 73.8% 
              precision    recall  f1-score   support

     chinese       0.64      0.67      0.66       242
      indian       0.86      0.78      0.82       234
    japanese       0.66      0.83      0.74       254
      korean       0.94      0.58      0.72       242
        thai       0.71      0.82      0.76       227

    accuracy                           0.74      1199
   macro avg       0.76      0.74      0.74      1199
weighted avg       0.76      0.74      0.74      1199
```

## 서포트 벡터 분류기
서포트 벡터 분류기는 분류 및 회귀 작업에 사용되는 ML 메서드의 Support Vector Machine 제품군의 일부이다. SVM은 두 범주 간의 거리를 최대화하기 위해 "공간 내 지점에 교육예제를 매핑"한다. 후속 데이터는 해당 범주를 예측할 수 있도록 이 공간에 매핑된다.

### 연습 - 서포트벡터 분류기 적용하기
이제 서포트벡터 분류기로 좀 더 나은 정확도를 노려보자

1. KNeighbors 항목 뒤에 쉼표를 추가하고 다음 줄을 추가하라
```python
'SVC': SVC(),
```
결과가 꽤 좋다
```
Accuracy (train) for SVC: 83.2% 
              precision    recall  f1-score   support

     chinese       0.79      0.74      0.76       242
      indian       0.88      0.90      0.89       234
    japanese       0.87      0.81      0.84       254
      korean       0.91      0.82      0.86       242
        thai       0.74      0.90      0.81       227

    accuracy                           0.83      1199
   macro avg       0.84      0.83      0.83      1199
weighted avg       0.84      0.83      0.83      1199
```

## 앙상블 분류기
지난 번 테스트는 꽤 괜찮았지만 끝까지 이 길을 가보도록 하자. 특히 랜덤 포레스트와 AdaBoost를 사용해보자
```python
  'RFST': RandomForestClassifier(n_estimators=100),
  'ADA': AdaBoostClassifier(n_estimators=100)
```
결과는 특히 랜덤포레스트가 더 좋다.
```
Accuracy (train) for RFST: 84.5% 
              precision    recall  f1-score   support

     chinese       0.80      0.77      0.78       242
      indian       0.89      0.92      0.90       234
    japanese       0.86      0.84      0.85       254
      korean       0.88      0.83      0.85       242
        thai       0.80      0.87      0.83       227

    accuracy                           0.84      1199
   macro avg       0.85      0.85      0.84      1199
weighted avg       0.85      0.84      0.84      1199

Accuracy (train) for ADA: 72.4% 
              precision    recall  f1-score   support

     chinese       0.64      0.49      0.56       242
      indian       0.91      0.83      0.87       234
    japanese       0.68      0.69      0.69       254
      korean       0.73      0.79      0.76       242
        thai       0.67      0.83      0.74       227

    accuracy                           0.72      1199
   macro avg       0.73      0.73      0.72      1199
weighted avg       0.73      0.72      0.72      1199
```

이 머신러닝은 모델의 품질을 향상시키기 위해 "여러 기본 추정기의 예측을 결합"한다. 이 예시에서는 랜덤트리와 AdaBoost를 사용하였다.

-평균과 방법인 랜덤 포레스트는 과적합을 피하기 위해 무작위성이 주입된 결정트리의 숲을 구축한다. n_estimators 매개변수는 트리 수로 설정된다

-AdaBoost는 분류기를 데이터셋에 적합시킨 다음 해당 분류기의 복사본을 동일한 데이터셋에 적합시킨다. 잘못 분류된 항목의 가중치에 초점을 맞추고 다음 분류자가 수정할 적합도를 조정한다.
















