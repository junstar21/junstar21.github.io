---
title:  "Feature Engineering(Scaling,  Transformation, 로그/지수함수, 이산화, 인코딩, 파생변수생성)"
excerpt: "2022-11-09 주택 가격 예측 경진대회 데이터를 활용한 Scaling,  Transformation, 로그/지수함수, 이산화, 인코딩, 파생변수생성"

categories:
  - TIL
tags:
  - python
  - EDA
  - Learning Machine
  - Feature Scaling
  - Transformation
  - log function
  - exponential function
  - Discretisation
  - Encoding
  - Feature Generation
  - Feature Engineering

# layout: post
# title: Your Title Here
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
{% include spotifyplaylist.html id=page.spotifyplaylist %}


# [지난 포스팅에서 이어짐](https://junstar21.github.io/til/np.log1p%EC%99%80-np.expm1,-Feature-Engineering,-%ED%9D%AC%EC%86%8C%EA%B0%92-%ED%83%90%EC%83%89/)

## Feature Scaling

### 📖 변수 스케일링의 개념

- 범위가 다르면 Feature끼리 비교하기 어려우며, 일부 머신러닝 모델에서는 제대로 작동하지 않는다.
- Feature Scaling이 잘 되어있으면 다른 변수끼리 비교하는 것이 편리하다.
- Feature Scaling이 잘 되어있으면 알고리즘 속도와과 머신러닝의 성능 향상을 기대할 수 있다.
- 일부 Feature Scaling은 이상치에 강한 경향을 보일 수 있다.
- 변수 스케일링 기법
    
    
    | 이름 | 정의 | 장점 | 단점 |
    | --- | --- | --- | --- |
    | Normalization - Standardization (Z-score scaling) | 평균을 제거하고 데이터를 단위 분산에 맞게 조정 | 표준 편차가 1이고 0을 중심으로 하는 표준 정규 분포를 갖도록 조정 | 변수가 왜곡되거나 이상치가 있으면 좁은 범위의 관측치를 압축하여 예측력을 손상시킴 |
    | Min-Max scaling | Feature를 지정된 범위로 확장하여 기능을 변환한다. 기본값은 [0,1] |  | 변수가 왜곡되거나 이상치가 있으면 좁은 범위의 관측치를 압축하여 예측력을 손상시킴 |
    | Robust scaling | 중앙값을 제거하고 분위수 범위(기본값은 IQR)에 따라 데이터 크기를 조정한다. | 편향된 변수에 대한 변환 후 변수의 분산을 더 잘 보존하며, 이상치 제거에 효과적이다. |  |
- 변수 스케일링 기법에 따른 정규분포도 비교
    
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled.png)
    
    - 왼쪽부터 Normalization - Standardization(std), Min-Max, Robust
    - x 축 값을 보면 min-max x값이 0~1 사이에 있고, std => 평균을 빼주고 표준편차로 나눠주고, roubust => 중간값으로 빼고 IQR로 나눠준 결과

### 변수 스케일링과 트랜스포메이션

`SalePrise`의 분포도를 확인해보자.

```python
train["SalePrice"].hist(bins = 50, figsize = (8,3))
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 1.png)

분포도를 확인하면 약간 왼쪽으로 치우친 형태를 띄고 있는 것을 확인할 수 있다.

각 스케일링 기법들은 sklearn에 내장되어있으므로 각각 호출을 하도록 한다.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
```

### ⌨️ **Scaling 기법 적용하기**

- StandardScaler의 fit에는 matrix를 넣어주어야 하기 때문에 Series가 아닌 DataFrame으로 넣어야해서 대괄호를 두번 감싸서 DF형태로 넣어준다. 또한, 반환값도 matrix 형태이기 때문에 새로운 파생변수를 만들고자 한다면 DF형태로 파생변수를 만들어준다.
- 사이킷 런의 다른 기능에서는 fit ⇒ predict를 했었지만, 전처리에서는 fit ⇒ transform을 사용한다.
- 스케일링을 예시로 fit 은 계산하기 위한 평균, 중앙값, 표준편차가 필요하다면 해당 데이터를 기준으로 기술통계값(`describe()`)을 구하고 해당 값을 기준으로 transform에서 계산을 적용해서 값을 변환해준다.
- fit 은 train에만 사용하고 transform은 train, test 에 사용한다. fit 은 test 에 사용하지 않는다. 그 이유는 기준을 train으로 정하기 위해서이다. test에는 train을 기준으로 학습한 것을 바탕으로 transform 만 진행하도록 한다.

```python
ss = StandardScaler()

train[["SalePrice_ss"]] = ss.fit(train[["SalePrice"]]).transform(train[["SalePrice"]])
train[["SalePrice", "SalePrice_ss"]].head(2)
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 2.png)

Min Max, Robust 모두 위와 같은 코드로 진행된다. 

```python
# Min-Max 할당
mm = MinMaxScaler()
train[["SalePrice_mm"]] = mm.fit(train[["SalePrice"]]).transform(train[["SalePrice"]])

# Robuse 할당
rs = RobustScaler()
train[["SalePrice_rs"]] = rs.fit(train[["SalePrice"]]).transform(train[["SalePrice"]])

# 적용이 되었는지 확인해보자.
train[["SalePrice", "SalePrice_mm", "SalePrice_rs"]].head(2)
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 3.png)

```python
# 위 코드들은 다음과 같이도 작성 할 수 있다.
# 주의할 사항이라면 train의 feature에만 fit_transform기능을 사용해야한다.

train['SalePrice_ss'] = ss.fit_transform(train[['SalePrice']])
train['SalePrice_mm'] = mm.fit_transform(train[['SalePrice']])
train['SalePrice_rb'] = rb.fit_transform(train[['SalePrice']])
```

Scaling처리한 colums의 기술통계값과 히스토그램을 살펴보자.

```python
train[["SalePrice", "SalePrice_ss", "SalePrice_mm", "SalePrice_rs"]].describe()
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 4.png)

* StandardScaling 의 특징 : 평균값이 0이며, 표준편차가 1이다.
* Min-Max의 특징 : 최솟값이 0 최댓값이 1
* Robust Scaling의 특징 :  중간값(중앙값, 50%, 2사분위수)가 0

각 scaling한 값을 histogram으로 plot하여 시각화를 해보도록 한다.
```python
train[["SalePrice", "SalePrice_ss", "SalePrice_mm", "SalePrice_rs"]].hist(bins = 50, figsize = (10,5));
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 5.png)

## Transformation

### 📖 **Transformation의 개념**

- Robust Scaling을 제외한 Feature Scaling은 일반적으로 편향된 분포나 이상치에 취약하며, Feature Scaling을 해줘도 표준정규분표형태를 띄지 않음.
- 그러기 위해선 log Transformation이 필요함
    - log Transformation을 적용하는 이유는 log 함수가 x값에 대해 상대적으로 작은 스케일에서는 키우고, 큰 스케일에서는 줄여주는 효과가 있기 때문
- 편향된 Feature의 경우 log가 적용된 값은 원래 값에 비해서 더 고르게 분포되며, 이는 y예측값에 유용하다.

### 📒 **정규분포와 Transformation**
    
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 6.png)

* 원래 값과 표준정규분포를 띄게 된 값의 displot을 절대평가 기준으로 4분위로 나누어 비교(Equal width binning)
  * 원래 값은 1분위에 값이 몰려 있기 때문에 이 구간에서 예측 성능이 뛰어날 수 있으나, 일반적인 예측 성능은 낮아지게 된다.
  * 표준정규분포를 띄게 된 값은 2,3분위에 값이 집중되어 있어 일반적인 예측 성능이 올라가게 된다.
- (1, 4)구간보다 (2, 3)구간이 상대적으로 더 중요하다.
    - 예측하려는 값이 비교적 매우 작거나 매우 큰 값보단 중간값에 가까운 값일 확률이 높기 때문이다.
    - 따라서, 중간값을 잘 예측하는 모델이, 일반적인 예측 성능이 높은 모델이다.
- 정규분포로 고르게 분포된 값이 예측에 더 유리한 자료이다.
    - 정규분포로 고르게 분포시키는 것이 다양한 예측값에 대해서 대응할 수 있게 해준다.
    - scale이 작은 값과 scale이 큰 값에 대해서 비슷하게 대응할 수 있게 해줄 수 있다.
- log transformation만 적용해도 정규분포 형태가 되며, Standard Scaler를 적용하면 표준 편차가 1이고 0을 중심으로 하는 표준정규분포를 갖도록 조정할 수 있다.

### ⌨️ **Transformation 실습**

```python
# SalePrice에 log함수를 취한다.
train["SalePrice_log1p"] = np.log1p(train["SalePrice"])

# 위에 SalePrice에 log함수를 취한 값에 StandardScale 처리를 한다.
train[["SalePrice_log1p_ss"]] = ss.fit_transform(train[["SalePrice_log1p"]])

# 반대로 StandardScale 처리가 된 값에 log함수를 취한다.
train["SalePrice_ss_log1p"] = np.log1p(train["SalePrice_ss"])
```

`SalePrice_ss_log1p`과정에서 Error가 발생하는 이유는, `SalsPrice_ss` 중에 (-1)보다 높은 값이 있기 때문에 (+1) 을 해도 음수가 발생한다. 따라서 해당 값은 `NaN` 값으로 변환된다. 이는 `dsecribe()`의 count항목에서 확인할 수 있다.

```python
train[["SalePrice_ss", "SalePrice_ss_log1p", "SalePrice_log1p","SalePrice_log1p_ss"]].describe()
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 7.png)

Scaling한 Feature와 Scaling + Transformation한 Feature의 histogram을 살펴보도록 한다.

```python
train[["SalePrice_ss", "SalePrice_ss_log1p", "SalePrice_log1p","SalePrice_log1p_ss"]].hist(bins = 50, figsize = (10,5));
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 8.png)

`SalePrice_log1p_ss`이 가장 표준정규분포에 가깝우며, `SalePrice_log1p`도 정규분포에 가까운 모습이다.

**🤔 표준정규분표와 그냥 정규분포 두 개 중에는 모델에서 사용할 때 성능차이가 많이 발생하는가?**

- 트리계열 모델을 사용한다면 일반 정규분포를 사용해도 무관하나, 스케일값이 영향을 미치는 모델에서는 표준정규분포로 만들어 주면 더 나은 성능을 낼 수도 있다.
- 하지만, 표준정규분포로 만들 때 값이 왜곡될 수도 있기 때문에 주의가 필요하며, 상황에 맞는 변환방법을 사용하는 것을 추천한다.

**🤔 log처리와 Scaling처리의 순서는 어떻게 하는 것이 좋은가?**

- 데이터에 따라 다르기 때문에 적절한 해석과 상황에 따라서 적용해주는 것이 중요하다.
- 만약 log를 적용해줘야겠다고 한다면 log적용 후 Scaling해주는 것이 정규분포에 가깝기 때문에 log → Scaling 순서로 적용해준다.

## 지수함수, 로그함수 이해

### **로그함수(log)**

```python
# np.arange(1,10,0.5) : 1부터 10까지 0.5 단위로 리스트를 만들어줌.
# array([1. , 1.5, 2. , 2.5, 3. , 3.5, 4. , 4.5, 5. , 5.5, 6. , 6.5, 7. ,
#       7.5, 8. , 8.5, 9. , 9.5])

x = np.arange(1,10,0.5)
sns.lineplot(x=x, y = x)
sns.lineplot(x=x, y=np.log(x))
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 9.png)

로그는 x가 음수값이면 존재할 수 없다.

```python
np.log(-1)

에러값 출력

결과값 : nan
```

### **지수함수(e)**

범위에 음수값이 있으면 해당 값은 0으로 출력해준다.

```python
x = np.arange(-10,10,0.5)
sns.lineplot(x=x, y=np.exp(x))
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 10.png)

지수함수와 로그함수의 그래프를 보면 x=y 기울기를 기준으로 서로 반대형상을 띄는 것을 확인할 수 있다.

## 이산화

### 📖 **이산화의 개념**

- 이산화(Discretisation)는 Numerical Feature를 일정 기준으로 나누어 그룹화

❓ **이산화를 사용하는 이유**

- 우리의 사고방식과 부합하는 측면이 있어 직관적이기 때문
- ex) 인구 구성원을 분석할 때, 해당 나이를 다 측정하는 것보단 20대, 30대, 40대 이러한 식으로 분석하면 경향이 뚜렷해지고 이해하기가 쉬워짐.
- 데이터 분석과 머신러닝 모델에 유리.
    - 유사한 예측 강도를 가진 유사한 속성을 그룹화하여 모델 성능을 개선하는 데 도움
    - Numerical Feature로 인한 과대적합을 방지

📂 **이산화의 종류**

- Equal width binning : 범위를 기준으로 나누는 것
- Equal frequency binning :  빈도를 기준으로 나누는 것
    
    
    | 방법 | 정의 | 장점 | 단점 |
    | --- | --- | --- | --- |
    | Equal width binning
    예) 절대평가, 히스토그램, pd.cut(), 고객을 구매금액에 따라 나눌 때 | 가능한 값의 범위를 동일한 너비의 N개의 bins로 나눈다. |  | 편향된 분포에 민감 |
    | Equal frequency binning
    예) 상대평가, pd.qcut(), 고객을 구매금액 상위 %에 따라 등급을 나눌 때 | 변수의 가능한 값 범위를 N개의 bins로 나눈다. 여기서 각 bins은 동일한 양의 관측값을 전달한다. | 알고리즘의 성능 향상에 도움 | 이 임의의 비닝은 대상과의 관계를 방해가능성 있음. |

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 11.png)

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 12.png)

### ⌨️ **이산화 실습**

```python
# SalePrice - cut
# bins : 몇개로 구간을 나눌 것이지 결정.
# labels : 나눈 구간의 이름을 어떻게 붙힐지 결정
train["Saleprice_cut"] = pd.cut(train["SalePrice"], bins = 4, labels=[1,2,3,4])

# SalePrice - qcut
# q : Quantile. 4개의 Quantile로 구간을 나눔
train["Saleprice_qcut"] = pd.qcut(train["SalePrice"], q = 4, labels=[1,2,3,4])
```

이산화를 진행한 각각의 columns의 `value_counts()`를 살펴보자.

```python
# value_counts(1) : value_counts의 비율을 나타내주는 옵션이다.

display(train["SalePrice_cut"].value_counts())
train["SalePrice_cut"].value_counts(1)

결과값 : 
1    1100
2     330
3      25
4       5
Name: SalePrice_cut, dtype: int64

1   0.75
2   0.23
3   0.02
4   0.00
Name: SalePrice_cut, dtype: float64
```

```python
display(train["SalePrice_qcut"].value_counts().sort_index())
display(train["SalePrice_qcut"].value_counts(1).sort_index())

결과값 : 
1    365
2    367
3    366
4    362
Name: SalePrice_qcut, dtype: int64

1   0.25
2   0.25
3   0.25
4   0.25
Name: SalePrice_qcut, dtype: float64
```

`pd.cut()`은 특정구간에 따라 나누었기 때문에 구역별 비율이 다르지만, `pd.qcut()`은 `SalePrice`의 비율에 따라 4개의 구간으로 나누었기 때문에 비율이 동일한 것을 확인할 수 있다. 

`value_counts()`의 내용을 시각화해보도록 한다.

```python
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (10,4))
sns.countplot(data = train, x = "SalePrice_cut", ax = ax[0]).set(title="cut")
sns.countplot(data = train, x = "SalePrice_qcut", ax = ax[1]).set(title="qcut")
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 13.png)

- `pd.cut`은 절대평가와 유사한 개념이며, histogram의 `bins`와 같은 개념이다.
- `pd.qcut`은 상대평가와 유사한 개념이기 때문에 `pd.qcut`으로 데이터를 분할하게 되면 비슷한 비율로 나눠주게 된다.
- 머신러닝에서 데이터를 분할해서 연속된 수치데이터를 이산화 해주는 이유는 머신러닝 알고리즘에 힌트를 줄 수도 있고, 너무 세분화된 조건으로 오버피팅(과대적합)되지 않도록 도움을 줄 수 있기 때문이다.

## 인코딩

### 📖 인코딩 개념

- 인코딩(Encoding)은 Categorical Feature를 Numerical Feature로 변환하는 과정

❓ **인코딩을 사용하는 이유**

- 데이터 시각화와 머신러닝 모델에 유리하기 때문이다.
- 최근 부스팅 3대장(Xgboost, LightGBM, catBoost) 알고리즘 중에는 범주형 데이터를 알아서 처리해 주는 알고리즘도 있지만 사이킷런에서는 범주형 데이터를 피처로 사용하기 위해서는 별도의 변환작업이 필요하다.

**Ordinal-Encoding**

- Ordinal-Encoding은 Categorical Feature의 고유값들을 임의의 숫자로 바꿉니다.
- 지정하지 않으면 0 부터 1씩 증가하는 정수로 지정.
- 장점 : 직관적이며 개념적으로 복잡하지 않고 간단하다.
- 단점 : 데이터에 추가적인 가치를 더해주지 않는다.
    - 값이 크고 작은게 의미가 있을 때는 상관 없지만, 순서가 없는 데이터에 적용해 주게 되면 잘못된 해석을 할 수 있으니 주의가 필요하다.
- 순서가 있는 명목형 데이터에 사용한다. ex)기간의 1분기, 2분기, 3분기, 4분기
    
    
    | 인코딩 전 | 인코딩 후 |
    | --- | --- |
    | favorite_drink | favorite_drink |
    | coffee | 0 |
    | coke | 1 |
    | water | 2 |

**One-Hot-Encoding**

- One-Hot-Encoding은 Categorical Feature를 다른 bool 변수(0 또는 1)로 대체하여 해당 관찰에 대해 특정 레이블이 참인지 여부를 나타낸다.
- `pd.get_dummies()` 로 사용이 가능하다.
- 장점 : 해당 Feature의 모든 정보를 유지한다.
- 단점 : 해당 Feature에 너무 많은 고유값이 있는 경우, Feature을 지나치게 많이 사용한다.
- 순서가 없는 명목형 데이터에 사용한다. ex) 좋아하는 음료, 주택의 종류, 수업의 종류
    
    
    | 인코딩 전 | 인코딩 후 |  |  |
    | --- | --- | --- | --- |
    | favorite_drink | favorite_drink_coffee | favorite_drink_coke | favorite_drink_water |
    | coffee | 1 | 0 | 0 |
    | coke | 0 | 1 | 0 |
    | water | 0 | 0 | 1 |

### ⌨️ 인코딩 실습

`MSZonig` 변수로 **Ordinal-Encoding**과 **One-Hot-Encoding**실습을 진행하도록 한다.

```
MSZoning: Identifies the general zoning classification of the sale.

   A    Agriculture
   C    Commercial
   FV    Floating Village Residential
   I    Industrial
   RH    Residential High Density
   RL    Residential Low Density
   RP    Residential Low Density Park 
   RM    Residential Medium Density
```

```python
train["MSZoning"].value_counts()

결과값 : 
RL         1151
RM          218
FV           65
RH           16
C (all)      10
Name: MSZoning, dtype: int64
```

Ordinal-Encoding

```python
# .astype("category").cat.codes 을 통해서 Ordinal-Encoding을 진행한다.
display(train["MSZoning"].astype("category").cat.codes)
train["MSZoning"].astype("category").cat.codes.value_counts()

결과값 : 
Id
1       3
2       3
3       3
4       3
5       3
       ..
1456    3
1457    3
1458    3
1459    3
1460    3
Length: 1460, dtype: int8
3    1151
4     218
1      65
2      16
0      10
dtype: int64
```

확인 결과, RL → 3, RM → 2, FV → 1, RH → 2, C → 0으로 변환된 것을 확인할 수 있으며, vector(1차원) 형태로 나오는 것을 볼 수 있다.

One-Hot-Encoding

```python
pd.get_dummies(train["MSZoning"])
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 14.png)

One-Hot-Encoding은 matrix(2차원)형태로 나오는 것을 확인할 수 있다.

**sklearn을 이용한 인코딩**

사이킷런을 이용해서 Ordinal-Encoding과 One-Hot-Encoding을 사용할 수 있다. 작동 개념은 아래와 같다.

```python
from sklearn.preprocessing import OrdinalEncoder
oe = OrdinalEncoder()
X = [['male', 'from US', 'uses Safari'], 
     ['female', 'from Europe', 'uses Firefox']]

# X 변수를 fit함으로써 숫자로 변환시켜준다.
oe.fit(X)

# DF안에 있는 내용들을 인코딩 된 값으로 변환시켜준다.
print(oe.transform([['female', 'from US', 'uses Safari']]))
결과값 : [[0. 1. 1.]]

print(oe.categories_)
결과값 : [array(['female', 'male'], dtype=object), array(['from Europe', 'from US'], dtype=object), array(['uses Firefox', 'uses Safari'], dtype=object)]

```

```python
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder()
X = [['male', 'from US', 'uses Safari'],
     ['female', 'from Europe', 'uses Firefox']]

# X 변수를 fit함으로써 숫자로 변환시켜준다.
enc.fit(X)

# DF안에 있는 내용들을 인코딩 된 값으로 변환시켜주고 matrix 형태(.toarray())로 변환시켜준다
enc_out = enc.transform([['female', 'from US', 'uses Safari'],
               ['male', 'from Europe', 'uses Safari']]).toarray()

print(enc_out)
결과값 : [[1. 0. 0. 1. 0. 1.]
         [0. 1. 1. 0. 0. 1.]]

print(enc.get_feature_names_out())
결과값 : ['x0_female' 'x0_male' 'x1_from Europe' 'x1_from US' 
          'x2_uses Firefox' 'x2_uses Safari']

pd.DataFrame(enc_out, columns=enc.get_feature_names_out())
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 15.png)

이제 사이킷런을 활용해서 MSZoning을 인코딩해보자.

```python
MSZoning_oe = oe.fit_transform(train[["MSZoning"]])
print(MSZoning_oe)train["MSZoning_oe"] = oe.fit_transform(train[["MSZoning"]])
train[["MSZoning", "MSZoning_oe"]].sample(3)
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 16.png)

```python
# MSZoning_enc
MSZoning_enc = enc.fit_transform(train[["MSZoning"]]).toarray()

print(MSZoning_enc)
결과값 :
[[0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 0.]
 ...
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 0.]
 [0. 0. 0. 1. 0.]]

print(enc.get_feature_names_out())
결과값 : ['MSZoning_C (all)' 'MSZoning_FV' 'MSZoning_RH' 'MSZoning_RL' 'MSZoning_RM']

pd.DataFrame(MSZoning_enc, columns = enc.get_feature_names_out())
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 17.png)

🤔 **사이킷런으로 인코딩을 하는 이유?**

- `pandas.get_dummies` 는 인코딩한 데이터를 저장하지 않는다. 즉, train set과 test set 모두 각각 적용을 해줘야 인코딩을 진행 할 수 있다. 하지만, 현실세계 혹은 경진대회에서는 test set에 있는 내용은 어떤 내용인지 알 수가 없다(실제 일부 경진대회에서는 test set의 인코딩을 금하고 있다). 다시 말해서, train set의 인코딩 내용과 test set의 인코딩 내용이 다르게 나올 가능성이 있다는 것이다.
- 반면, 사이킷런의 경우, 특정 인코딩 내용을 저장할 수 있다. train set에 있는 내용을 fit을 해주게 되면, fit에는 train set의 내용을 기준으로 인코딩해주는 기능을 저장하게 된다. fit에 저장된 내용을 토대로 train set과 test set을 transform을 하면 fit에 저장되있는 내용으로 train과 test를 인코딩해주게 된다. 종합하자면, train에서 진행된 인코딩 내용을 동일하게 test에서도 진행해줄 수 있다.

## 파생변수

### 📖 파생변수 개념

- 변수 생성(Feature Generation)은 이미 존재하는 변수로부터 여러 방법으로 새로운 변수를 만들어내는 것
- 산술적인 방법, 시간, 지역 등의 방법으로 변수를 생성할 수 있음.
- 적절한 파생변수는 머신러닝과 예측을 향상시킬 수 있지만, 부적절한 파생변수 생성은 오히려 역효과를 일으킬 수 있다.
- 다항식 전개(Polynomial Expansion) : 주어진 다항식의 차수 값에 기반하여 파생변수를 생성
- sklearn 라이브러리에서는 `PolynomialFeatures` 객체를 통해 다항식 전개에 기반한 파생변수 생성을 지원하고 있다.

### ⌨️ Polynomial Features

`PolynomialFeatures`의 기본 원리를 살펴보자.

```python
# preprocessing - PolynomialFeatures

from sklearn.preprocessing import PolynomialFeatures

# np.reshape 는 array 의 shape 값을 지정해서 shape를 변환해 준다.

X = np.arange(6).reshape(3, 2)
print(X)
결과값 : 
[[0 1]
 [2 3]
 [4 5]]

# degree == 차수
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
# get_feature_names_out() : fit한 내용들의 columns를 리스트 형태로 출력해준다.
pd.DataFrame(X_poly, columns=poly.get_feature_names_out())
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 18.png)

위 원리를 이용해서 `"MSSubClass"`, `"LotArea"`를 다항식 전개해보자.

```python
house_poly = poly.fit_transform(train[["MSSubClass", "LotArea"]])
pd.DataFrame(house_poly, columns=poly.get_feature_names_out())
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-09-Feature Engineering(Scaling, Transformation, 로그&지수함수, 이산화, 인코딩, 파생변수생성)/Untitled 19.png)

히스토그램을 그렸을 때 어딘가는 많고 적은 데이터가 있다면 그것도 특징이 될 수 있지만, 특징이 잘 구분되지 않는다면 `power transform`등을 통해 값을 제곱을 해주거나 연산을 통해 특징을 더 도드라지게 해줄 수 있다.
