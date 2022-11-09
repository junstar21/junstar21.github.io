---
title:  "titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습"
excerpt: "2022-11-07 지난 시간의 titanic데이터의 하이퍼파라미터 튜닝 후 제출, Bike shareing damand 데이터 뎃 실습."

categories:
  - TIL
tags:
  - python
  - EDA
  - Learning Machine
  - Hyper Parameter
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
{% include spotifyplaylist.html id=page.spotifyplaylist %}


# [지난 포스팅에서 이어짐](https://junstar21.github.io/til/%ED%83%80%EC%9D%B4%ED%83%80%EB%8B%89-%EB%8D%B0%EC%9D%B4%ED%84%B0-%EC%85%8B%EC%9D%98-%ED%8C%8C%EC%83%9D%EB%B3%80%EC%88%98-%EB%A7%8C%EB%93%A4%EA%B8%B0,-One-Hot-Encoding,-%EA%B2%B0%EC%B8%A1%EC%B9%98-%EB%8C%80%EC%B2%B4,-%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-%EC%95%8C%EA%B3%A0%EB%A6%AC%EC%A6%98-%EC%A0%81%EC%9A%A9/)

## 하이퍼파라미터 튜닝 - RandomSearchCV

모델의 적합한 파라미터 튜닝값을 알아보기 위해 RandomSearchCV를 사용하였다.

```python
# RandomizedSearchCV 호출
from sklearn.model_selection import RandomizedSearchCV

# np.random.randint : 해당 범위 내 랜덤값을 정해줌
# np.random.uniform : 해당 범위 내 랜덤값을 중복되지 않는 수로 정해줌.

param_distributions = {"max_depth": np.random.randint(3, 100, 10), 
                       "max_features": np.random.uniform(0, 1, 10)}

# n_iter : 해당 작업을 얼마나 반복할지 결정
clf = RandomizedSearchCV(estimator=model, 
                         param_distributions=param_distributions,
                         n_iter=5,
                         n_jobs=-1,
                         random_state=42
                        )

clf.fit(X_train, y_train)
```

fit을 하여 최적의 파라미터 값을 알아본다.

```python
best_model = clf.best_estimator_
best_model

결과값
RandomForestClassifier(max_depth=9, max_features=0.4723162098197786, n_jobs=-1,
                       random_state=42)
```

추가적으로 점수와 어떤 결과들이 있는지를 확인해본다.

```python
# 최고의 점수값을 확인
clf.best_score_

결과값 : 0.826062394074446
```

```python
# 파라미터 조사 결과를 df형태로 나타내고, rank 순으로 정렬.
pd.DataFrame(clf.cv_results_).sort_values("rank_test_score").head()
```

## Best Estimator

```python
# 데이터를 머신러닝 모델로 학습(fit)합니다.
# 데이터를 머신러닝 모델로 예측(predict)합니다.
best_model.fit(X_train, y_train)
```

## 제출

```python
submit = pd.read_csv("data/titanic/gender_submission.csv")
file_name = f"{clf.best_score_}.csv"

submit["Survived"] = y_predict

submit.to_csv(file_name, index = False)
```

<aside>
🤔 **Cross Validation과 Hold-out Validation의 차이**

7:3 이나 8:2 로 나누는 과정은 hold-out-validation 입니다. hold-out-validation 은 중요한 데이터가 train:valid 가 7:3이라면 중요한 데이터가 3에만 있어서 제대로 학습되지 못하거나 모든 데이터가 학습에 사용되지도 않습니다. 그래서 모든 데이터가 학습과 검증에 사용하기 위해 cross validation을 합니다.
hold-out-validation 은 한번만 나눠서 학습하고 검증하기 때문에 빠르다는 장점이 있습니다. 하지만 신뢰가 떨어지는 단점이 있습니다.  hold-out-validation 은 당장 비즈니스에 적용해야 하는 문제에 빠르게 검증해보고 적용해 보기에 좋습니다.
cross validation 이 너무 오래 걸린다면 조각의 수를 줄이면 좀 더 빠르게 결과를 볼 수 있고 신뢰가 중요하다면 조각의 수를 좀 더 여러 개 만들어 보면 됩니다.

</aside>

<aside>
💡 **점수에 대한 강사님의 소견**

타이타닉 데이터는 이미 답이 공개가 되어있기 때문에 치팅(답을 베껴서 제출)이 많습니다. 피처엔지니어링을 많이 하면 많이 할 수록 점수가 올라갈 것 같지만 내려갈 때가 더 많을 수도 있습니다. 점수를 올리고 내리는데 너무 집중하기 보다는 일단은 다양한 방법을 시도해 보는 것을 추천합니다. 다양한 사례를 탐색해 보는 것을 추천합니다. 팀을 꾸릴 때는 도메인 전문가, 프로그래머, 데이터 사이언티스트, 데이터 엔지니어 등으로 팀을 꾸립니다.

점수를 올리기 위해서는 EDA를 꼼꼼하게 하고 우리가 예측하고자 하는 정답이 어떤 피처에서 어떻게 다른 점이 있는지 특이한 점은 없는지 탐색해 보는게 중요합니다.

</aside>

# [Bike Shareing Demand](https://www.kaggle.com/competitions/bike-sharing-demand) 실습

## 경진대회의 성격 파악하기

어떤 문제 종류? ⇒ 회귀

무엇을 예측? ⇒ 매 시간 빌려진 자전거의 수의 예측

- Demand가 들어간 경진대회는 대부분 수요에 대한 예측문제

### 데이터 확인하기

```
datetime - hourly date + timestamp
season -  1 = spring, 2 = summer, 3 = fall, 4 = winter
holiday - whether the day is considered a holiday
workingday - whether the day is neither a weekend nor holiday
weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
temp - temperature in Celsius
atemp - "feels like" temperature in Celsius
humidity - relative humidity
windspeed - wind speed
casual - number of non-registered user rentals initiated
registered - number of registered user rentals initiated
count - number of total rentals
```

## 0601

### 라이브러리 및 데이터 로드와 데이터 확인

```python
# 라이브러리 로드

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
```

```python
# 데이터 로드 및 확인

train = pd.read_csv("data/bike/train.csv")
test = pd.read_csv("data/bike/test.csv")

print(train.shape, test.shape)
결과값 : (10886, 12) (6493, 9)

set(train.columns) - set(test.columns)
결과값 : {'casual', 'count', 'registered'}
```

확인 결과, 우리가 예측해야 하는 값은 `count` 인 것을 확인하였다. 하지만, `casual`과 `registered` 도 예측해야 하는 항목에 선정되어있다. 이 이후는 차후에 알아보도록 하겠다.

**결측치 확인**

```
train.info()

결과값 : 
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 10886 entries, 0 to 10885
Data columns (total 12 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   datetime    10886 non-null  object 
 1   season      10886 non-null  int64  
 2   holiday     10886 non-null  int64  
 3   workingday  10886 non-null  int64  
 4   weather     10886 non-null  int64  
 5   temp        10886 non-null  float64
 6   atemp       10886 non-null  float64
 7   humidity    10886 non-null  int64  
 8   windspeed   10886 non-null  float64
 9   casual      10886 non-null  int64  
 10  registered  10886 non-null  int64  
 11  count       10886 non-null  int64  
dtypes: float64(3), int64(8), object(1)
memory usage: 1020.7+ KB
```

```
test.info()

결과값 : 
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 6493 entries, 0 to 6492
Data columns (total 9 columns):
 #   Column      Non-Null Count  Dtype  
---  ------      --------------  -----  
 0   datetime    6493 non-null   object 
 1   season      6493 non-null   int64  
 2   holiday     6493 non-null   int64  
 3   workingday  6493 non-null   int64  
 4   weather     6493 non-null   int64  
 5   temp        6493 non-null   float64
 6   atemp       6493 non-null   float64
 7   humidity    6493 non-null   int64  
 8   windspeed   6493 non-null   float64
dtypes: float64(3), int64(5), object(1)
memory usage: 456.7+ KB
```

```
train.isnull().sum()

결과값 : 
datetime      0
season        0
holiday       0
workingday    0
weather       0
temp          0
atemp         0
humidity      0
windspeed     0
casual        0
registered    0
count         0
dtype: int64
```

```
test.isnull().sum()

결과값 :
datetime      0
season        0
holiday       0
workingday    0
weather       0
temp          0
atemp         0
humidity      0
windspeed     0
dtype: int64
```

```
train.describe()
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습/Untitled.png)

확인 결과

- casual,registered,count 평균값에 비해 max값이 크다
- datetime이 object 형식
- 풍속과 습도가 0인 날이 포함

### 전처리

날짜를 연, 월, 일, 분, 초로 나누는 파생변수를 만든다.

```python
# "datetime" column의 type을 datetime으로 변환한다.
train["datetime"] = pd.to_datetime(train["datetime"])

train["year"] = train["datetime"].dt.year
train["month"] = train["datetime"].dt.month
train["day"] = train["datetime"].dt.day
train["hour"] = train["datetime"].dt.hour
train["minute"] = train["datetime"].dt.minute
train["second"] = train["datetime"].dt.second

train.head(2)
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습/Untitled 1.png)

### EDA

히스토그램으로 전반적인 분포를 파악한다.

```python
# train의 histogram

train.hist(figsize = (12,10), bins = 50);
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습/Untitled 2.png)

- windspeed에 0이 많으며, 습도에도 0이 존재.
- 날씨의 경우, 맑은 날(1)이 제일 많은 것으로 파악.
- minute과 second는 0으로 존재.
- 우리가 예측하려는 count 값은 0이 대부분.

```python
# test의 histogram
test.hist(figsize = (12,10), bins = 50);
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습/Untitled 3.png)

- year의 분포가 train과 다른 형태를 띄고 있으며, 20의 값이 존재하지 않음.
- windspeed에서 0의 값이 굉장히 높은 분포를 띔.

**데이터들의 시각화를 통한 분석**

```python
train[train["windspeed"] == 0].shape

결과값 :
(1313, 18)

# 풍속과 대여량의 시각화
sns.scatterplot(data = train, x = "windspeed", y = "count")
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습/Untitled 4.png)

- 풍속의 값이 연속적으로 이어지는 것이 아닌, 범주형처럼 나뉘어지는 구간이 있어보인다.

```python
# 풍속과 대여량의 시각화
sns.scatterplot(data = train, x = "humidity", y = "count")
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습/Untitled 5.png)

- 여기에서는 0으로 된 값이 많아 보이지는 않으며, 습도와 자전거 대여량은 상관이 없어 보인다.

```python
# 온도와 체감온도의 시각화
sns.scatterplot(data = train, x = "temp", y = "atemp")
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습/Untitled 6.png)

- 온도와 체감온도는 강력한 양의 상관관계
- 오류 데이터가 존재하는 것으로 판단됨.

```python
# 이상치 찾기

train[(train["temp"] > 20) & (train["temp"] < 40) & (train["atemp"] < 15)]
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습/Untitled 7.png)

12년 8월 17일에 체감온도가 12.12도로 고정된 날짜들이 존재한다. 센서 고장 의심.

```python
# 날씨에 따른 평균 자전거 대여수
# ci = 에러바 표시유무. 버전에 따라 해당 명령어는 다르게 표기되니 확인할 필요가 있다.

sns.barplot(data = train, x = "weather", y = "count", ci = None)
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습/Untitled 8.png)

- 폭우 폭설이 내리는 날(4)이 비가 오는 날(3)보다 대여량이 많게 측정되었다.

날씨 4의 데이터를 확인해보기로 한다.

```python
train[train["weather"] == 4]
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습/Untitled 9.png)

확인한 결과 폭우와 폭설이 내리는 경우의 데이터는 단 하나만 존재하는 것을 확인하였다.

### 학습, 예측 데이터 만들기

```python
# label_name : 정답값
label_name = "count"

# feature_names : 학습, 예측에 사용할 컬럼명(변수)
# train columns 중 count, datetime, casual, registered 항목이 test에 없기 제외한다.
feature_names = train.columns.tolist()
feature_names.remove(label_name)
feature_names.remove("datetime")
feature_names.remove("casual")
feature_names.remove("registered")

# 학습(훈련)에 사용할 데이터셋 예) 시험의 기출문제
X_train = train[feature_names]

# 예측 데이터셋, 예) 실전 시험 문제
X_test = test[feature_names]

# 학습(훈련)에 사용할 정답값 예) 기출문제의 정답
y_train = train[label_name]
```

### 머신러닝 알고리즘

회귀 유형이므로 `RandomForestRegressor`를 사용한다.

```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(random_state= 42, n_jobs = -1)
```

<aside>
💡 **criterion**
분류형과 회귀형일때 설정 값이 다르다. regreesion에서는 squared_error가 default로 설정되어있다.

</aside>

### 교차검증

```python
# 모의고사를 풀어서 답을 구하는 과정과 유사합니다.
# cross_val_predict는 예측한 predict값을 반호나하여 직접 계산해 볼 수 있습니다.
# 다른 cross_val_score, cross_validate는 스코어를 조각마다 직접 계산해서 반환해줍니다.

from sklearn.model_selection import cross_val_predict

y_valid_pred = cross_val_predict(model, X_train, y_train, cv = 5, n_jobs = -1, verbose=2)
y_valid_pred

결과값 :
array([ 74.45,  65.47,  44.94, ..., 165.29, 152.17,  84.65])
```

### 평가

각종 평가수식으로 평가를 진행하였다. MAE, MSE, RMSE에 대한 자세한 사항은 10/31일자 내용을 확인하도록 하자.

****MAE(Mean Absolute Error)****

```python
mae = abs(y_train - y_valid_pred).mean()
결과값 : 50.40957652030154

# sklearn에서도 똑같이 mad를 구할 수 있다.
from sklearn.metrics import mean_absolute_error
mean_absolute_error(y_train, y_valid_pred)
결과값 : 50.40957652030131
```

**MSE(Mean Squared Error)**

```python
# MSE(Mean Squared Error)
mse = np.square(y_train - y_valid_pred).mean()
결과값 : 5757.8679269795975

from sklearn.metrics import mean_squared_error
mean_squared_error(y_train, y_valid_pred)
결과값 : 5757.867926979607
```

****RMSE(Root Mean Squared Error)****

```python
# RMSE(Root Mean Squared Error)
RMSE = np.sqrt(mse)
결과값 : 75.88061627965074
```

<aside>
💡 **멘토님의 remind**

MAE

- 모델의 예측값과 실제 값 차이의 절대값 평균
- 절대값을 취하기 때문에 가장 직관적임

MSE

- 모델의 예측값과 실제값 차이의 면적의(제곱)합
- 제곱을 하기 때문에 특이치에 민감하다.

RMSE

- MSE에 루트를 씌운 값
- RMSE를 사용하면 지표를 실제 값과 유사한 단위로 다시 변환하는 것이기 때문에 MSE보다 해석이 더 쉽다.
- MAE보다 특이치에 Robust(강하다)
</aside>

****RMSLE(Root Mean Squared Logarithm****

- $\sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 }$
- 각 log마다 1을 더하는 이유 : 정답에 +1을 해서 1보다 작은 값이 있을 때 마이너스 무한대로 수렴하는 것을 방지
- 로그를 취하면 skewed 값이 덜 skewed(찌그러지게) 하게 된다. 또한, 스케일 범위값이 줄어드는 효과를 볼 수 있다.
    
    ```python
    sns.kdeplot(y_train)
    sns.kdeplot(y_valid_pred)
    ```
    
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습/Untitled 10.png)
    
    ```python
    sns.kdeplot(np.log(train["count"]+1))
    ```
    
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습/Untitled 112.png)
    
- 또한, 분포가 좀 더 정규분포에 가까워지기도 한다
- RMSLE는 RMSE 와 거의 비슷하지만 오차를 구하기 전에 예측값과 실제값에 로그를 취해주는 것만 다르다.

```python
# RMSLE 계산

(((np.log1p(y_train) - np.log1p(y_valid_pred)) **2).mean()) ** (1/2)
결과값 : 0.5200652012443514

from sklearn.metrics import mean_squared_log_error
(mean_squared_log_error(y_train, y_valid_pred)) **(1/2)
결과값 : 0.5200652012443514
```

<aside>
💡 **멘토님의 예시**

RMSLE는 예측과 실제값의 "상대적" 에러를 측정해줍니다.

예를 들어서
실제값: 90, 예측값: 100 일 때
RMSE = 10
RMSLE = 0.1042610...

실제값: 9,000, 예측값: 10,000 일 때
RMSE = 1,000
RMSLE = 0.1053494...

RMSLE의 한계는 상대적 에러를 측정하기 때문에
예를 들자면 1억원 vs 100억원의 에러가 0원 vs 99원의 에러와 같다 라고 나올 수 있습니다.

그리고 RMSLE는 실제값보다 예측값이 클떄보다, 실제값보다 예측값이 더 작을 때 (Under Estimation) 더 큰 패널티를 부여합니다.

배달 시간을 예측할때 예측 시간이 20분이었는데 실제로는 30분이 걸렸다면 고객이 화를 낼 수도 있을겁니다. 이런 조건과 같은 상황일 때 RMSLE를 적용할 수 있을 것입니다.

</aside>

<aside>
💡 **강사님 예시**

- 부동산 가격으로 예시를 들면 1) 2억원짜리 집을 4억으로 예측 2) 100억원짜리 집을 110억원으로 예측 
Absolute Error 절대값의 차이로 보면  1) 2억 차이 2) 10억 차이
Squared Error 제곱의 차이로 보면 1) 4억차이 2) 100억차이
Squared Error 에 root 를 취하면 absolute error 하고 비슷해 집니다.
비율 오류로 봤을 때 1)은 2배 잘못 예측, 2)10% 잘못 예측
- 자전거 대여수는 대부분 작은 값에 몰려있습니다. 그래서 log를 취하고 계산하게 되면 오차가 큰 값보다 작은값에 더 패널티가 들어가게 됩니다.
</aside>

<aside>
💡 **RMSE와 RMSLE의 차이**
RMSE: 오차가 클수록 가중치를 주게 됨(오차 제곱의 효과)
RMSLE: 오차가 작을수록 가중치를 주게 됨(로그의 효과). 최소값과 최대값의 차이가 큰 값에 주로 사용. ex) 부동산 가격

⚠️ 측정 공식은 이 분야에는 이 공식이 딱 맞다라기 보다는 보통 해당 도메인에서 적절하다고 판단되는 공식을 선택해서 사용

</aside>

### 학습 및 제출

```python
y_predict = model.fit(X_train, y_train).predict(X_test)
```

제출할 파일명에는 계산한 RMSLE의 값이 들어간 파일을 제출하여 구분하기 쉽도록 하였다.

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습/Untitled 12.png)

점수를 더 올려보기 위해서 피처를 조정하기로 한다.

```python
feature_names = train.columns.tolist()
feature_names.remove(label_name)
feature_names.remove("datetime")
feature_names.remove("casual")
feature_names.remove("registered")
feature_names.remove('month')
feature_names.remove('day')
feature_names.remove('second')
feature_names.remove('minute')
feature_names

결과값 :
['season',
 'holiday',
 'workingday',
 'weather',
 'temp',
 'atemp',
 'humidity',
 'windspeed',
 'year',
 'hour']
```

피처를 조정(day, month, second, minute 제외) 후 동일한 방법을 진행 후 케글에 제출하고 점수를 확인하였다.

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습/Untitled 13.png)

점수가 상향한 모습을 볼 수 있다. second와 minute는 값이 0이기에 제외하고, day 는 train 에는 1~19일 test 에는 20~말일까지 있기 때문에 학습한 것이 예측에 도움지 않기 때문에 제외를 한다. (위가 train set, 아래가 test set)

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습/Untitled 14.png)

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습/Untitled 15.png)

month의 경우, 달에 따라 count 값이 영향을 받는 거 같지만 2011년과 2012년의 동일 달을 비교 했을때 차이가 크기 때문에 삭제

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-titanic 하이퍼파라미터 튜닝, Bike Shareing Demand 실습/Untitled 16.png)