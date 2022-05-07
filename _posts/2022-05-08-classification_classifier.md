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




