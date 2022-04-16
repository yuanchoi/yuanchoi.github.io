# 타이타닉 머신러닝 과제

오늘은 캐글에서 입문자용으로 불리는 타이타닉 생존자 데이터를 가지고 모델훈련을 해보려고 한다

## 문제
타이타닉 데이터 세트를 해결하여라. 목표는 다른 열을 바탕으로 Survived열을 예상하는 것이다.

```import sys
assert sys.version_info >= (3, 7)
```

```
import sklearn
assert sklearn.__version__ >= "1.0.1"
```

```
from pathlib import Path
import pandas as pd
import tarfile
import urllib.request

def load_titanic_data():
    tarball_path = Path("datasets/titanic.tgz")
    if not tarball_path.is_file():
        Path("datasets").mkdir(parents=True, exist_ok=True)
        url = "https://github.com/ageron/data/raw/main/titanic.tgz"
        urllib.request.urlretrieve(url, tarball_path)
        with tarfile.open(tarball_path) as titanic_tarball:
            titanic_tarball.extractall(path="datasets")
    return [pd.read_csv(Path("datasets/titanic") / filename)
            for filename in ("train.csv", "test.csv")]
```

```
train_data, test_data = load_titanic_data()
```

데이터가 이미 훈련 세트와 테스트 세트로 나뉘어져있다. 하지만 데이터가 레이블을 포함하지 않으므로, 목표는 훈련 데이터를 가지고 최고의 훈련모델을 훈련시키고, 테스트 데이터를 예상하는 것이다.

```
train_data.head()
```


각각의 속성이 지닌 의미이다.

PassengerID: 각각의 승객을 구별하는 고유 식별자

Survived: 목표물이다. 0은 승객이 살아남지 못했다는 뜻이고, 1은 승객이 살아남았다는 의미이다.

Pclass: 승객의 객실 등급

Name, Sex, Age: 승객의 개인정보

SipSp: 타이타닉에 오른 승객의 형제자매&배우자 수

Parch: 타이타닉에 오른 승객의 자녀&부모 수

Ticket: 티켓 아이디

Fare: 티켓 요금

Cabin: 승객의 캐빈 숫자

Embarked: 승객이 타이타닉에 승선한 장소

목표는 승객의 나이, 성별, 객실 등급 등을 바탕으로 승객이 생존했는지를 예측하는 것이다.

```
train_data = train_data.set_index("PassengerId")
test_data = test_data.set_index("PassengerId")
```
```
train_data.info()
```
```
train_data[train_data["Sex"]=="female"]["Age"].median()
```

누락된 데이터 중 Cabin이 77%가 누락되었으므로, 이 속성은 무시하도록 한다. 누락된 데이터는 중간값으로 대체하도록 한다
```
train_data.describe()
```
```
train_data["Survived"].value_counts()
```
```
train_data["Pclass"].value_counts()
```
```
train_data["Sex"].value_counts()
```
```
train_data["Embarked"].value_counts()
```

Embark 속성은 승객이 어디서 승선했는지를 알려준다. C는 Cherbourg, Q는 Queenstown, S는 Southampton이다.

수와 관련된 속성을 위한 파이프라인부터 시작하여 전처리 파이프라인을 만들자

```
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
                         ("imputer", SimpleImputer(strategy="median")),
                         ("scaler", StandardScaler())
])
```
```
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
```
```
cat_pipeline = Pipeline([
                         ("ordinal_encoder", OrdinalEncoder()),
                         ("imputer", SimpleImputer(strategy="most_frequent")),
                         ("cat_encoder", OneHotEncoder(sparse=False)),
])
```
```
from sklearn.compose import ColumnTransformer

num_attribs = ["Age", "SibSp", "Parch", "Fare"]
cat_attribs = ["Pclass", "Sex", "Embarked"]

preprocess_pipeline = ColumnTransformer([
                                         ("num", num_pipeline, num_attribs),
                                         ("cat", cat_pipeline, cat_attribs),
])
```
```
X_train = preprocess_pipeline.fit_transform(train_data)
X_train
```
```
y_train = train_data["Survived"]
```

이제 분류기를 훈련시킬 준비가 되어있다. RandomForestClassifier부터 시작해보자

```
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_clf.fit(X_train, y_train)
```
모델이 훈련되었으니, 테스트 세트를 예상해보자. 이제 이 예상으로 CSV 파일을 만들면 된다.

더 나은 예상을 하고 싶다면 교차검증을 하면 된다.

```
X_test = preprocess_pipeline.transform(test_data)
y_pred = forest_clf.predict(X_test)
```
```
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
forest_scores.mean()
```
```
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
svm_scores.mean()
```
```
plt.figure(figsize=(8, 4))
plt.plot([1]*10, svm_scores, ".")
plt.plot([2]*10, forest_scores,".")
plt.boxplot([svm_scores, forest_scores], labels=("SVM", "Random Forest"))
plt.ylabel("Accuracy")
plt.show()
```
```
train_data["AgeBucket"] = train_data["Age"] // 15 * 15
train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean()
```
```
train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
train_data[["RelativesOnboard", "Survived"]].groupby(
    ['RelativesOnboard']).mean()
```






