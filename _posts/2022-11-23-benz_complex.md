---
title: "GBM 3대장 실습, Complex Matrix 이론"
excerpt: "2022-11-23 XBG, LightGBM, Catboost, Complex Matrix"

# layout: post
categories:
  - TIL
tags:
  - python
  - EDA
  - Learning Machine
  - Feature Engineering
  - Linear Regression
  - Gradient Boosting
  - XBG
  - LightGBM
  - Catboost
  - Complex Matrix
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
### **⚠️ 해당 내용은 멋쟁이사자처럼 AI School 오늘코드 박조은 강사의 자료를 토대로 정리한 내용입니다.**
{% include spotifyplaylist.html id=page.spotifyplaylist %}

11/16에서 이어짐

**🤔 배깅과 부스팅을 사용하는 경우?**

- 배깅 : 오버피팅
- 부스팅 : 개별 트리의 성능이 중요할때

# Benz boosting model input

⚠️ **해당 과정은 구글의 colab에서 실행하는 것을 추천한다!** 버전의 호환성과 더불어 boost 모델들은 다른 언어로 제작되있기 때문에, 로컬환경에서 모델을 설치할 경우 오류 발생으로 인해 로컬환경에 문제가 생길 수 있다.

## category type 변경

범주형 피처들을 먼저 살펴보기로 한다.

```python
# object columns만 따로 꺼내서 변수형 피처들 다루기
cat_col = train.select_dtypes(include="object").columns
cat_col

결과값:
Index(['X0', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X8'], dtype='object')
```

```python
# lightGBM, CatBoost에서는 범주형 피처를 인코딩없이 사용할 수 있다.
# 따로 범주형 피처를 지정해서 사용할 수 있다.

train[cat_col] = train[cat_col].astype("category")
test[cat_col] = test[cat_col].astype("category")
```

## Feature Engineering

### One-Hot-Encoding

`pd.get_dummies()` 기능을 사용해도 좋지만, 현재 예제에서는 sklearn의 성능이 더 좋게 나오기 때문에 sklearn을 사용한다.

```python
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(handle_unknown = "ignore")

train_ohe = ohe.fit_transform(train.drop(columns = 'y'))
test_ohe = transform(test)

train_ohe.shape, test_ohe.shape

결과값 :
((4209, 919), (376, 4209))
```

```python
# hold-out validation 으로 train값 나누기

X = train_ohe
y = train.y

X.shape, y.shape

결과값 :
((4209, 919), (4209,))
```

## 학습, 검증세트 나누기

```python
# train_test_split을 이용해 X, y 값을 X_train, X_valid, y_train, y_valid 으로 나눠줍니다.
# Hold-out-validation을 위해 train, valid세트로 나눠준다.

from sklearn.model_selection import train_test_split

# test_size = 0.1 : valid 사이즈를 지정한다. 
	# train의 비중을 90%, valid의 비중을 10%로 나눠준다.
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size = 0.1, random_state = 42
)

X_train.shape, X_valid.shape, y_train.shape, y_valid.shape

결과값:
((3788, 919), (421, 919), (3788,), (421,))

X_test = test_ohe
```

## **XGBoost 모델**

### XGBoost Parameter

- 부스팅 파라미터
    - Learning_rate[기본값 : 0.3] : Learning rate가 높을수록 과적합되기 쉬움
    - n_estimators [기본값 : 100] : 생성할 weaker learner 수. learning_rate가 낮을 땐 n_estimators를 높여야 과적합이 방지됨. value가 너무 낮으면 underfitting이 되고 이는 낮은 정확성의 prediction이 되는 반면, value가 너무 높으면 overfitting이 되고 training data 에는 정확한 prediction을 보이지만 test data에서는 정확성이 낮은 prediction을 가짐
    - max_depth [기본값 : 3] : 트리의 maximum depth. 적절한 값이 제시되어야 하고 보통 3-10 사이 값이 적용됨, max_depth가 높을수록 모델의 복잡도가 커져 과적합되기 쉬움
    - min_child_weight [기본값 : 1] : 관측치에 대한 가중치 합의 최소를 말함. 값이 높을수록 과적합이 방지됨
    - gamma [기본값 : 0] : 리프노드의 추가분할을 결정할 최소손실 감소값. 해당값보다 손실이 크게 감소할 때 분리, 값이 높을수록 과적합이 방지됨
    - subsample [기본값 : 1] : weak learner가 학습에 사용하는 데이터 샘플링 비율, 보통 0.5 ~ 1 사용됨, 값이 낮을수록 과적합이 방지됨
    - colsample_bytree [ 기본값 : 1 ] : 각 tree 별 사용된 feature의 퍼센테이지, 보통 0.5 ~ 1 사용됨, 값이 낮을수록 과적합이 방지됨
- 일반 파라미터
    - booster [기본값 = gbtree] : 어떤 부스터 구조를 쓸지 결정, 의사결정기반모형(gbtree), 선형모형(gblinear), dart
    - n_jobs : XGBoost를 실행하는 데 사용되는 병렬 스레드 수
    - verbosity [기본값 = 1] : 로그출력여부 0 (무음), 1 (경고), 2 (정보), 3 (디버그)
    - early_stopping_rounds : 손실함수 값이 n번정도 개선이 없으면 학습을 중단
- 학습과정 파라미터
    - eval_metric:
        - rmse: root mean square error
        - mae: mean absolute error
        - logloss: negative log-likelihood
        - error: Binary classification error rate (0.5 threshold)
        - merror: Multiclass classification error rate
        - mlogloss: Multiclass logloss
        - auc: Area under the curve
        - map (mean average precision)

```python
# xgboost 모듈 호출
import xgboost as xgb

model_xgb = xgb.XGBRegressor(random_state= 42, n_jobs=-1)

model_xgb.fit(X_train, y_train)

model_xgb.feature_importances_[:5]

결과값 : 
array([0.        , 0.00473197, 0.00115691, 0.        , 0.        ],
      dtype=float32)

xgb.plot_importance(model_xgb)
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-23-benz_complex/Untitled.png)

```python
xgb.plot_tree(model_xgb, num_trees=1)
fig = plt.gcf()
fig.set_size_inches(30, 20)
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-23-benz_complex/Untitled 1.png)

**🤔 배깅 모델은 시각화가 어려워 3rd party 도구를 따로 설치해야 시각화 가능하다. 그것도 개별 트리를 시각화 하는 것은 어렵다. 그런데 부스팅 모델은 왜 시각화가 가능할까?**

배깅모델은 병렬적으로 트리를 여러 개 생성하지만, 부스팅은 순차적으로 생성하기 때문에 가능하다.

valid 예측 점수를 확인해보도록 하자.

```python
# valid score
score_xgb = model_xgb.score(X_valid, y_valid)
score_xgb

결과값 : 0.6128264118065729
```

이후, 학습된 내용으로 예측을 시키고 제출까지 진행하도록 한다. 진행 방법은 지난 포스팅을 참고하도록 한다.

```python
# predict
y_pred_xgb = model_xgb.predict(X_test)

submission['y'] = y_pred_xgb

file_name = f'{base_path}/sub_xgb_{score_xgb}.csv'
submission.to_csv(file_name)
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-23-benz_complex/Untitled 2.png)

## [LightGBM](https://lightgbm.readthedocs.io/en/latest/Parameters.html) 모델

- XGBoost에 비해 성능은 비슷하지만 학습 시간을 단축시킨 모델이다.
- XGBoost에 비해 더 적은 시간, 더 적은 메모리를 사용한다.

### LightGBM Parameters

- max_depth : 나무의 깊이. 단일 결정나무에서는 충분히 데이터를 고려하기 위해 depth를 적당한 깊이로 만들지만, 보정되기 때문에 부스팅에서는 깊이 하나짜리도 만드는 등, 깊이가 짧은것이 크리티컬하지 않음
- min_data_in_leaf : 잎이 가질 수 있는 최소 레코드 수, 기본값은 20, 과적합을 다루기 위해 사용
- feature_fraction : 부스팅 대상 모델이 랜덤포레스트일때, 랜덤포레스트는 feature의 일부만을 선택하여 훈련하는데, 이를 통제하기 위한 파라미터, 0.8이라면 LightGBM이 각 반복에서 80%의 파라미터를 무작위로 선택하여 트리를 생성
- bagging_fraction : 데이터의 일부만을 사용하는 bagging의 비율. 예를들어 오버피팅을 방지하기 위해 데이터의 일부만을 가져와서 훈련시키는데, 이는 오버피팅을 방지하며 약한예측기를 모두 합칠경우는 오히려 예측성능이 좋아질 수 있음 훈련 속도를 높이고 과적합을 방지하는 데 사용
- early_stopping_round : 더이상 validation데이터에서 정확도가 좋아지지 않으면 멈춰버림 훈련데이터는 거의 에러율이 0에 가깝게 좋아지기 마련인데, validation데이터는 훈련에 사용되지 않기때문에 일정이상 좋아지지 않기 때문
- lambda : 정규화에 사용되는 파라미터, 일반적인 값의 범위는 0 ~ 1
- min_gain_to_split : 분기가 되는 최소 정보이득, 트리에서 유용한 분할 수를 제어하는 데 사용
- max_cat_group : 범주형 변수가 많으면, 하나로 퉁쳐서 처리하게끔 만드는 최소단위
- objective : lightgbm은 regression, binary, multiclass 모두 가능
- boosting: gbdt(gradient boosting decision tree), rf(random forest), dart(dropouts meet multiple additive regression trees), goss(Gradient-based One-Side Sampling)
- num_leaves: 결정나무에 있을 수 있는 최대 잎사귀 수. 기본값은 0.31
- learning_rate : 각 예측기마다의 학습률 learning_rate은 아래의 num_boost_round와도 맞춰주어야 함
- num_boost_round : boosting을 얼마나 돌릴지 지정한다. 보통 100정도면 너무 빠르게 끝나며, 시험용이 아니면 1000정도 설정하며, early_stopping_round가 지정되어있으면 더이상 진전이 없을 경우 알아서 멈춤
- device : gpu, cpu
- metric: loss를 측정하기 위한 기준. mae (mean absolute error), mse (mean squared error), 등
- max_bin : 최대 bin
- categorical_feature : 범주형 변수 지정
- ignore_column : 컬럼을 무시한다. 무시하지 않을경우 모두 training에 넣는데, 뭔가 남겨놓아야할 컬럼이 있으면 설정
- save_binary: True 메모리 절약

전체적인 과정은 XGBoost와 동일하나, 사용하는 모델만 LightGBM으로 다르다. 모델만 바꿔준 뒤 진행하도록 한다.

```python
import lightgbm as lgbm
# model_lgbm
model_lgbm = lgbm.LGBMRegressor(random_state = 42, n_jobs = -1)

model_lgbm.fit(X_train, y_train)

lgbm.plot_importance(model_lgbm)
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-23-benz_complex/Untitled 3.png)

```python
lgbm.plot_tree(model_lgbm, figsize=(20, 20), tree_index=0, 
               show_info=['split_gain', 'internal_value', 'internal_count', 'leaf_count'])
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-23-benz_complex/Untitled 4.png)

```python
# valid score
score_lgbm = model_lgbm.score(X_valid, y_valid)

score_lgbm 
결과값 : 0.5720514617008872

# predict
y_pred_lgbm = model_lgbm.predict(X_test)

y_pred_lgbm[:5]
결과값 :
array([ 76.99591393,  92.18224812,  77.30829539,  75.78294519,
       111.97681237])

# submit
submission['y'] = y_pred_lgbm
file_name = f'{base_path}/sub_lgbm_{score_lgbm}.csv'
submission.to_csv(file_name)
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-23-benz_complex/Untitled 5.png)

## [CatBoost](https://catboost.ai/en/docs/references/training-parameters/common) 모델

- catboost는 기존 GBT의 느린 학습 속도와 과대적합 문제를 개선한 모델입니다.
- 과대적합이란 모델이 지나친 학습으로 인해 경향이 학습용 세트에 쏠려 있는 현상을 말합니다.
- 학습용 세트에서는 예측을 잘 하지만(특수한 상황), 일반적인 상황에서 예측 능력이 떨어지는 것입니다.

### 주요 파라미터

- cat_features
    - 범주형 변수 인덱스 값
- loss_function
    - 손실 함수를 지정합니다.
- eval_metric
    - 평가 메트릭을 지정합니다.
- iterations
    - 머신러닝 중 만들어질 수 있는 트리의 최대 갯수를 지정합니다.
- learning_rate
    - 부스팅 과정 중 학습률을 지정합니다.
- subsample
    - 배깅을 위한 서브샘플 비율을 지정합니다.
- max_leaves
    - 최종 트리의 최대 리프 개수를 지정합니다.

Catboost 역시 다른 GBM모델처럼 진행을 해준다.

```python
# catboost
import catboost

# model_cat

# catboost 의 회귀 알고리즘의 기본 metric 은 RMSE
# eval_metric = "R2" : 알고리즘을 R2 Score로 설정.
model_cat = catboost.CatBoostRegressor(eval_metric="R2", verbose = False)

from scipy.stats import randint
from sklearn.utils.fixes import loguniform

# catboost는 자체적으로 searchCV기능이 있다.
# grow_policy : 트리를 어떤식으로 성장시킬 것인지를 결정. Defalut로는 SymmetricTree(대칭트리).
                # Lossguide - 리프별, Depthwise : 깊이별
param_grid = {
    'n_estimators': randint(100, 300),
    'depth': randint(1, 5),
    'learning_rate': loguniform(1e-3, 0.1),
    'min_child_samples': randint(10, 40),
    'grow_policy': ['SymmetricTree', 'Lossguide', 'Depthwise']
}

# randomized_search
result = model_cat.randomized_search(param_grid, X_train, y_train, cv=3, n_iter=10)

df_result = pd.DataFrame(result)
df_result = df_result.loc[["train-R2-mean", "test-R2-mean"], "cv_results"]
df_result

결과값 :
train-R2-mean    [-55.46630687885133, -49.38772650148917, -43.9...
test-R2-mean     [-55.529821058885176, -49.444705355730996, -44...
Name: cv_results, dtype: object

pd.DataFrame({"train-R2-mean": df_result.loc["train-R2-mean"], 
              "test-R2-mean" :  df_result.loc["test-R2-mean"] }).tail(3)
```

|  | train-R2-mean | test-R2-mean |
| --- | --- | --- |
| 215 | 0.616273 | 0.554339 |
| 216 | 0.616432 | 0.554440 |
| 217 | 0.616722 | 0.554476 |

```python
# R2 Score의 마지막 50개 값을 plot

pd.DataFrame({"train-R2-mean": df_result.loc["train-R2-mean"], 
              "test-R2-mean" :  df_result.loc["test-R2-mean"] }).tail(50).plot()
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-23-benz_complex/Untitled 6.png)

```python
# R2 Score의 전체적인 모양 plot
pd.DataFrame({"train-R2-mean": df_result.loc["train-R2-mean"], 
              "test-R2-mean" :  df_result.loc["test-R2-mean"] }).plot()
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-23-benz_complex/Untitled 7.png)

```python
# fit
model_cat.fit(X_train, y_train)

score_cat = model_cat.score(X_valid, y_valid)
score_cat
결과값 : 0.6172756661736007

# Predict
y_cat_pred = model_cat.predict(X_test)

# submit 
submission['y'] = y_cat_pred
file_name = f'{base_path}/sub_cat_{score_cat}.csv'
submission.to_csv(file_name)
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-23-benz_complex/Untitled 8.png)

## 범주형 데이터 다루기

### **category type 변경**

```python
# object => category
# category 타입으로 되어있으면 lightGBM, CatBoost에서 인코딩없이 사용할 수 있다.

cat_col = train.select_dtypes(include="object").columns
train[cat_col] = train[cat_col].astype("category")
test[cat_col] = test[cat_col].astype("category")
```

### LightGBM

```python
# lgbm.LGBMRegressor
model_lgbmr = lgbm.LGBMRegressor(random_state = 42)

# 데이터를 전처리하지 않고 category 형태로 넣어주면 알아서 학습한다.
# category형태로 되어 있다면 인코딩 과정이 필요 없어진다.

from sklearn.model_selection import cross_val_score

cv_score_lgbmr = cross_val_score(model_lgbmr, train.drop(columns="y"), train["y"], cv=3)

# fit & predict
model_lgbmr.fit(train.drop(columns = 'y'), train['y'])
```

### Catboost

```python
model_cat = catboost.CatBoostRegressor(eval_metric='R2', verbose=False, cat_features=cat_col.tolist())

from sklearn.model_selection import cross_val_predict

y_valid_cat = cross_val_predict(model_cat, train.drop(columns = "y"), train["y"], cv = 3)

from sklearn.metrics import r2_score

r2_score(train["y"], y_valid_cat)

# fit & predict 
y_pred_cat = model_cat.fit(train.drop(columns="y"), train["y"]).predict(test)
```

이 과정을 진행한 이유?

 ⇒ 전처리, 인코딩 없이 쉽게 학습을 진행해보기 위함.

# 불균형 데이터: SMOTE 와 분류 측정지표

## Confusion Matrix(혼동 행렬)

- Confusion Matrix의 사용 이유
    - 암환자 진단의 경우, 높은 정확도와 상관없이 참/거짓이 굉장히 중요한 데이터 종류 중 하나
    - 희소한 데이터를 정확하게 예측하는 것이, 전체 데이터에 대한 정확도보다 중요할 경우가 있음
    - 또한, 현실에서 마주하는 데이터들의 대부분은 ‘불균형 데이터’임.
    - 정확도 외에 다른 측정 지표가 필요

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-23-benz_complex/Untitled 9.png)

위 그림은, 출저마다 다름.

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-23-benz_complex/Untitled 10.png)

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-23-benz_complex/Untitled 11.png)

- TRUE : 모델이 맞췄을 때
- FALSE : 모델이 틀림
- Positive : 모델이 예측 값이 TRUE
- Negative : 모델이 예측 값이 FALSE
- FP(False Positive, Negative Positive) - 1종 오류
    - 실제는 임신이 아닌데(0), 임신(1)로 예측
- FN(False Negative, Positive Negative) - 2종 오류
    - 실제는 임신인데(1), 임신이 아닌 것(0)으로 예측
- $Precision = tp/(tp + fp)$
    - 정밀도 측정 - 스팸메일 확인 여부
- $Recall = tp/(tp + fn)$
    - 재현율 - 암환자 여부

😵‍💫 **개념이 햇갈린다?** 

모르는걸 아는거로 거짓말 했다 : 모를 수 있으니깐 거짓말 하지 말라고 1단계 정도로 혼남(1종 오류)

아는데 모르는거로 거짓말 했다 : 의도적으로 거짓말을 했기에 2단계 호되게 혼남(2종 오류)