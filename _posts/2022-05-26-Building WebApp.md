# 머신러닝 모델을 사용하여 WebApp 만들기
이번에는 이 세상에 없었던 데이터셋에 대한 머신러닝 모델을 훈련할 예정이다.(NUFORC의 데이터베이스트를 참고) 그리고 훈련된 모델을 'pickle'하고 Flask앱에서 모델을 사용하게 된다. 이 때 Flask와 Pickle이 무엇인지를 알아보자.

✅ **Flask**: 'micro-framework'로 정의한 Flask는 파이썬으로 웹프레임워크의 기본적인 기능과 웹페이지를 만드는 템플릿 엔진을 제공한다.

✅ **Pickle**: Pickle은 파이썬 객체구조를 serialize와 de-serialize하는 파이썬 모듈이다. 모델을 pickle하게 되면, 웹에서 쓰기 위해 serialize 또는 flatten한다. pickle된 파일은 `.pkl` 확장자를 가지고 있다.

## 연습하기 1 - 데이터 정리
NUFORC에서 모아둔 8만개의 UFO 목격 데이터를 사용할 예정이다. 데이터에는 UFO 목격과 관련한 길고 짧은 다양한 설명이 있다.

*   "A man emerges from a beam of light that shines on a grassy field at night and he runs towards the Texas Instruments parking lot"
-> 한 남자가 밤에 풀밭을 비추는 한 줄기 빛으로부터 출현하고, 그는 텍사스 instrument 주차장으로 달려간다.

*   "the lights chased us"
-> 빛이 우리를 쫓아왔다

ufos.csv 스프레드시트에는 UFO가 목격된 `city`, `state`와 `country`, UFO의 `shape`, `latitude`, `longitude` 열이 포함되어 있다.

1. `pandas`, `matplotlib`,`numpy`를 import하고 ufos.csv도 import한다.

```
import pandas as pd
import numpy as np
ufos = pd.read_csv('./data/ufos.csv')
ufos.head()
```

2. ufos 데이터를 새로운 제목의 작은 데이터프레임으로 변환한다. `Country` 필드가 유니크 값인지를 확인해야 한다.

```
ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})

ufos.Country.unique()
```

3. 이제 모든 null값을 드랍하고 1-60초 사이동안 목격한 케이스만 가져와 데이터의 수량을 줄인다

```
ufos.dropna(inplace=True)

ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]

ufos.info()
```

4. 사이킷런의 `LabelEncoder` 라이브러리를 import하여 국가의 텍스트 값을 숫자로 변환한다

✅ LabelEncoder는 데이터를 알파벳순으로 인코딩한다.

```
from sklearn.preprocessing import LabelEncoder

ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

ufos.head()
```

✅ 데이터는 이렇게 보인다

```
	Seconds	Country	Latitude	Longitude
2	20.0	3	    53.200000	-2.916667
3	20.0	4	    28.978333	-96.645833
14	30.0	4	    35.823889	-80.253611
23	60.0	4	    45.582778	-122.352222
24	3.0	    3	    51.783333	-0.783333
```


## 연습하기 2 - 모델 만들기

이제 데이터를 훈련하고 테스트셋과 훈련셋으로 나누어 모델을 훈련할 준비가 되었다.

1. X 벡터로 훈련시키고 싶은 세 개의 feature를 선택한다면, y 벡터는 `Country`가 될 것이다. 우리는 `Seconds`, `Latitude`, `Longitude`를 입력하면 국가 id가 결과로 나오길 바란다.

```
from sklearn.model_selection import train_test_split

Selected_features = ['Seconds','Latitude','Longitude']

X = ufos[Selected_features]
y = ufos['Country']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```

2. 로지스틱 회귀를 이용하여 모델을 훈련한다.

```
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print('Predicted labels: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))
```

`Country`, `Latitude/Longitude`가 상관관계가 있기 때문에, 정확도 95%는 나쁘지 않다.

만든 모델은 `Latitude`와 `Longitude`를 통해 `Country`를 알 수 있어야 하기 때문에 모델이 아주 혁신적이지는 않지만 원본 데이터베이스에서 훈련을 하고 웹앱에서 모델을 사용하기에는 좋다.


## 연습하기 3 - 모델 'pickle'하기

모델을 pickle하는 것은 크게 어렵지 않다. pickled되면, pickled된 모델을 불러와 초, 위도, 경도 값이 포함된 샘플 데이터 배열을 대상으로 테스트한다.

```
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

## 연습하기 4 - Flask 앱 만들기

1. ufo-model.pkl 파일과 notebook.ipynb 파일 옆에 web-app이라는 폴더를 만든다

2. web-app 폴더 내에 3가지 폴더를 만든다. **static** 내부에 **css**폴더가 있고, **template**도 있다. 이제 다음의 파일과 디렉토리들이 있어야 한다.

```
web-app/
  static/
    css/
    templates/
notebook.ipynb
ufo-model.pkl
```

3. web-app 폴더에서 만들 첫 파일은 **requirements.txt**파일이다. 자바스크립트앱의 *package.json*처럼, 앱에 필요한 모듈을 리스트한 파일이다. **requirements.txt** 에 해당라인을 추가한다.

```
scikit-learn
pandas
numpy
flask
```

4. web-app으로 이동하여 파일을 실행한다.

```
cd web-app
```

5. 터미널에서 `pip install`을 타이핑하여 **requirements.txt**에 나열된 라이브러리를 설치한다.

```
pip install -r requirements.txt
```

6. 이제 앱을 만들기 위해 3가지 파일이 더 필요하다

1) 최상단에 **app.py**를 만들기
2) *templates* 디렉토리에 **index.html**을 만든다
3) *static/css* 디렉토리에 **style.css**를 만든다.

7. 원하는 특정 스타일로  *style.css* 파일을 만들기:

```
body {
	width: 100%;
	height: 100%;
	font-family: 'Helvetica';
	background: black;
	color: #fff;
	text-align: center;
	letter-spacing: 1.4px;
	font-size: 30px;
}

input {
	min-width: 150px;
}

.grid {
	width: 300px;
	border: 1px solid #2d2d2d;
	display: grid;
	justify-content: center;
	margin: 20px auto;
}

.box {
	color: #fff;
	background: #2d2d2d;
	padding: 12px;
	display: inline-block;
}
```

8. 다음으로 *index.html* 파일을 만든다. 이 파일은 앱에서 어떻게 보일지를 설정해주는 역할을 한다.

```
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>🛸 UFO 출현을 예상해보자! 👽</title>
  <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}"> 
</head>

<body>
 <div class="grid">

  <div class="box">

  <p> UFO를 목격한 초 수, 위도, 경도에 따라 어떤 국가에서 발견되었는지를 예측해볼까요??</p>

    <form action="{{ url_for('predict')}}" method="post">
    	<input type="number" name="seconds" placeholder="초" required="required" min="0" max="60" />
      <input type="text" name="latitude" placeholder="위도" required="required" />
		  <input type="text" name="longitude" placeholder="경도" required="required" />
      <button type="submit" class="btn">UFO가 발견된 국가를 예측하기</button>
    </form>

  
   <p>{{ prediction_text }}</p>

 </div>
</div>

</body>
</html>
```

9. 아래 코드를 app.py에 추가한다.

```
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("./ufo-model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )


if __name__ == "__main__":
    app.run(debug=True)
```

## 결과물


![1](https://user-images.githubusercontent.com/79850142/170507694-6627f495-7234-4ad5-9bd7-253cc6735bb0.PNG)


![2](https://user-images.githubusercontent.com/79850142/170507748-b31f84ec-c1bd-41d3-bb06-32a9fdf67de8.PNG)

✅ 각각의 칸에 초, 위도, 경도를 입력해주면 

![3](https://user-images.githubusercontent.com/79850142/170507773-688c60fe-bd2f-4c37-bb3e-a6c44cf313f8.PNG)

✅ 예측국가명이 결과값으로 나온다.

















