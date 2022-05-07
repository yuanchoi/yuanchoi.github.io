# 요리 추천 웹어플리케이션 제작하기

이 수업에서, 당신은 이전 교육에서 배운 몇 가지 기술과 이 시리즈에서 사용된 맛있는 요리 데이터셋을 사용하여 분류 모델을 구축한다. 또한, 당신은 Onnx의 웹 런타임을 활용하여 저장된 모델을 사용할 수 있는 작은 웹어플리케이션을 만들 것이다.
머신러닝의 가장 유용한 적용 중 하나는 추천시스템은 구축하는 것이다.

## 모델 구축하기
응용 ML 시스템을 구축하는 것은 비즈니스 시스템에 이러한 기술을 활용하는 데 있어 중요한 부분이다. Onnx을 사용하여 웹 응용 프로그램 내에 모델을 사용할 수 있으므로 필요한 경우 오프라인 컨텍스트에서 모델을 사용할 수 있다.

이 과정에서 추론을 위한 기본 자바스크립트 기반 시스템을 구축할 수 있다. 그러나 먼저 모델을 교육하고 Onnx에서 사용하도록 변환해야 한다.

## 연습 - 분류모델을 훈련하기
우선, 우리가 사용한 청정 요리 데이터셋을 사용하여 분류모델을 훈련한다.

1. 유용한 라이브러리를 가져오기:

```python
!pip install skl2onnx
import pandas as pd 
```

2. 이후 `read_csv()`를 사용하여 CSV 파일을 읽어 이전 수업과 동일한 방식으로 데이터를 처리하라

```python
data = pd.read_csv('../data/cleaned_cuisines.csv')
data.head()
```

3. 처음 두 개의 불필요한 열을 제거하고 나머지 데이터를 X로 저장한다

```python
X = data.iloc[:,2:]
X.head()
```

4. 레이블을 y로 저장하라

```python
y = data[['cuisine']]
y.head()

```


### 훈련 루틴 시작하기
우리는 좋은 정확도를 가지고 있는 SVC 라이브러리를 이용할 것이다

1. 사이킷런으로부터 적절한 라이브러리를 임포트하기

```python
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,classification_report
```

2. 트레이닝셋과 테스트셋으로 분리하기

```python
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)
```

3. 이전 수업에서 했던 것처럼 SVC 분류를 구축하라

```python
model = SVC(kernel='linear', C=10, probability=True,random_state=0)
model.fit(X_train,y_train.values.ravel())
```

4. 이제, `predict()`를 호출하여 모델을 테스트하라

```python
y_pred = model.predict(X_test)
```

5. 모델의 질을 확인하기 위해 분류 레포트를 프린트하라

```python
print(classification_report(y_test,y_pred))
```

아까 봤듯이, 정확도는 꽤 좋다.

```
                precision    recall  f1-score   support

     chinese       0.72      0.69      0.70       257
      indian       0.91      0.87      0.89       243
    japanese       0.79      0.77      0.78       239
      korean       0.83      0.79      0.81       236
        thai       0.72      0.84      0.78       224

    accuracy                           0.79      1199
   macro avg       0.79      0.79      0.79      1199
weighted avg       0.79      0.79      0.79      1199
```

### 모델을 Onnx로 전환하기
적절한 텐서 수로 변환해야하는 것을 알아야한다. 이 데이터셋에는 380개의 성분이 나열되어 있으므로 FloatTensorType에 이 숫자를 기록해야 한다.

1. 380개의 텐서 수를 사용하여 변환하라

```python
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

initial_type = [('float_input', FloatTensorType([None, 380]))]
options = {id(model): {'nocl': True, 'zipmap': False}}
```

2. onx를 생성하고 파일을 model.onnx로 저장하라

```python
onx = convert_sklearn(model, initial_types=initial_type, options=options)
with open("./model.onnx", "wb") as f:
    f.write(onx.SerializeToString())
```

## 모델을 보기
Onnx 모델은 비주얼 스튜디오 코드에서 잘 보이지 않지만, 많은 연구자들이 모델을 시각화하기 위해 사용하는 매우 좋은 무료 소프트웨어가 있다. Netron을 다운로드하고 model.onnx 파일을 연다. 380개의 입력과 분류기가 나열된 단순 모델을 시각화할 수 있다.

<img width="196" alt="netron" src="https://user-images.githubusercontent.com/79850142/167270480-b11fee59-86b6-4f88-a884-fd06220b607c.png">

Netron은 모델을 보는 데 유용한 도구이다.

이제 웹 어플리케이션에서 이 깔끔한 모델을 사용할 준비가 되었다. 우리가 냉장고 안을 볼 때 유용하게 사용할 수 있는 앱을 만들고, 우리의 모델에 의해 결정되는 대로, 우리가 주어진 요리를 요리하기 위해 어떤 남은 재료의 조합을 사용할 수 있는지 알아보자.

1. 이 *index.html*에 다음 마크업을 추가하라

```
<!DOCTYPE html>
<html>
    <header>
        <title>Cuisine Matcher</title>
    </header>
    <body>
        ...
    </body>
</html>
```

2. 이제 `body` 태그 내에서 작업하면서 일부 성분을 반영하는 확인란 목록을 표시하기 위해 약간의 마크업을 추가한다.

```
<h1>Check your refrigerator. What can you create?</h1>
        <div id="wrapper">
            <div class="boxCont">
                <input type="checkbox" value="4" class="checkbox">
                <label>apple</label>
            </div>
        
            <div class="boxCont">
                <input type="checkbox" value="247" class="checkbox">
                <label>pear</label>
            </div>
        
            <div class="boxCont">
                <input type="checkbox" value="77" class="checkbox">
                <label>cherry</label>
            </div>

            <div class="boxCont">
                <input type="checkbox" value="126" class="checkbox">
                <label>fenugreek</label>
            </div>

            <div class="boxCont">
                <input type="checkbox" value="302" class="checkbox">
                <label>sake</label>
            </div>

            <div class="boxCont">
                <input type="checkbox" value="327" class="checkbox">
                <label>soy sauce</label>
            </div>

            <div class="boxCont">
                <input type="checkbox" value="112" class="checkbox">
                <label>cumin</label>
            </div>
        </div>
        <div style="padding-top:10px">
            <button onClick="startInference()">What kind of cuisine can you make?</button>
        </div> 
```

각 확인란에는 값이 지정된다는 것을 알아두어라. 이는 데이터셋에 따라 성분이 발견되는 지수를 반영한다. 예를 들어, 이 알파벳 목록에서 Apple은 다섯번째 열을 차지하기 때문에, 0에서 숫자를 세기 시작할 때 값는 '4'이다. 성분 스프레드시트를 참조하여 특정 성분의 색인을 찾을 수 있다.

index.html 파일에서 작업을 계속하고 최종종료 `</div>` 뒤에 모델이 호출되는 스크립트 블럭을 추가한다.

3. 첫번째로, Onnx Runtime을 임포트해라.

```
<script src="https://cdn.jsdelivr.net/npm/onnxruntime-web@1.9.0/dist/ort.min.js"></script> 
```

4. 런타임이 배치되면, 그것을 불러올 수가 있다.

```
<script>
    const ingredients = Array(380).fill(0);
    
    const checks = [...document.querySelectorAll('.checkbox')];
    
    checks.forEach(check => {
        check.addEventListener('change', function() {
            // toggle the state of the ingredient
            // based on the checkbox's value (1 or 0)
            ingredients[check.value] = check.checked ? 1 : 0;
        });
    });

    function testCheckboxes() {
        // validate if at least one checkbox is checked
        return checks.some(check => check.checked);
    }

    async function startInference() {

        let atLeastOneChecked = testCheckboxes()

        if (!atLeastOneChecked) {
            alert('Please select at least one ingredient.');
            return;
        }
        try {
            // create a new session and load the model.
            
            const session = await ort.InferenceSession.create('./model.onnx');

            const input = new ort.Tensor(new Float32Array(ingredients), [1, 380]);
            const feeds = { float_input: input };

            // feed inputs and run
            const results = await session.run(feeds);

            // read from results
            alert('You can enjoy ' + results.label.data[0] + ' cuisine today!')

        } catch (e) {
            console.log(`failed to inference ONNX model`);
            console.error(e);
        }
    }
           
</script>
```

이 코드에서는 현재 많은 일이 일어나고 있다.
1. 성분 확인란이 선택되었는지 여부에 따라 380개의 가능한 값(1또는0)을 설정하여 모든 모형으로 전송하여 추론했다.
2. 확인한 배열과 해당 확인란이 응용 프로그램이 시작될 때 호출되는 `init`함수로 확인되었는지 확인하는 방법을 만들었다. 확인란을 선택하면 선택한 성분을 반영하도록 성분 배열이 변경된다.
3. 확인란이 선택되었는지 여부를 확인하는 `testCheckboxes`함수를 만들었다. 버튼을 누르면 `start Inference`기능을 사용하고, 체크박스를 선택하면 추론을 시작한다.
4. 추론 루틴에는 다음이 포함된다
    1) 모델의 비동기 로드 설정
    2) 모델에 보낼 텐서 구조 생성
    3) 모델을 교육할 때 만든 `float_input`입력을 반영하는 'feeds'생성
    4) 이러한 feed를 모델에 보내고 응답을 기다린다


## 어플리케이션 테스트하기
index.html 파일이 있는 폴더에서 비주얼 스튜디오 코드에서 터미널 세션을 열어라. http-server가 전체적으로 설치되어 있는지 확인하고 프롬프트에 http-server를 입력한다. 로컬 호스트가 열리면 웹 어플리케이션을 볼 수 있다. 다양한 재료에 따라 어떤 요리를 추천하는지 확인한다.


<img width="816" alt="web-app" src="https://user-images.githubusercontent.com/79850142/167270808-bcc0ffb9-b06c-4b36-b871-54750025c76c.png">

















