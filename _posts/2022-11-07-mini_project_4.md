---
title: "구내식당 식수 인원 예측 AI 경진대회 알고리즘"
excerpt: "2022-11-07 데이콘의 구내식당 식수 인원 예측 AI 경진대회 알고리즘 실습"

# layout: post
categories:
  - Project
tags:
  - python
  - EDA
  - Learning Machine
  - Seaborn
  - Matplotlib
  - Feature Engineering
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
{% include spotifyplaylist.html id=page.spotifyplaylist %}


출처 : https://dacon.io/competitions/official/235743/overview/description

## 1. 분석 배경 및 목적

- 분석 배경

지금까지는 단순한 시계열 추세와 담당자의 직관적 경험에 의존하여 한국토지주택공사 구내식당 식수 인원을 예측하였으나, 빅데이터 분석으로 얻어지는 보다 정확도 높은 예측을 통해 잔반 발생량을 획기적으로 줄이고자 함.

- 목적

구내식당의 요일별 점심, 저녁식사를 먹는 인원 예측



## 2. 라이브러리 및 데이터 불러오기


```python
# 라이브러리 로드

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import koreanize_matplotlib
```


```python
# 데이터 로드
train = pd.read_csv("data/train.csv", parse_dates=["일자"])
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>일자</th>
      <th>요일</th>
      <th>본사정원수</th>
      <th>본사휴가자수</th>
      <th>본사출장자수</th>
      <th>본사시간외근무명령서승인건수</th>
      <th>현본사소속재택근무자수</th>
      <th>조식메뉴</th>
      <th>중식메뉴</th>
      <th>석식메뉴</th>
      <th>중식계</th>
      <th>석식계</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2016-02-01</td>
      <td>월</td>
      <td>2601</td>
      <td>50</td>
      <td>150</td>
      <td>238</td>
      <td>0.0</td>
      <td>모닝롤/찐빵  우유/두유/주스 계란후라이  호두죽/쌀밥 (쌀:국내산) 된장찌개  쥐...</td>
      <td>쌀밥/잡곡밥 (쌀,현미흑미:국내산) 오징어찌개  쇠불고기 (쇠고기:호주산) 계란찜 ...</td>
      <td>쌀밥/잡곡밥 (쌀,현미흑미:국내산) 육개장  자반고등어구이  두부조림  건파래무침 ...</td>
      <td>1039.0</td>
      <td>331.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2016-02-02</td>
      <td>화</td>
      <td>2601</td>
      <td>50</td>
      <td>173</td>
      <td>319</td>
      <td>0.0</td>
      <td>모닝롤/단호박샌드  우유/두유/주스 계란후라이  팥죽/쌀밥 (쌀:국내산) 호박젓국찌...</td>
      <td>쌀밥/잡곡밥 (쌀,현미흑미:국내산) 김치찌개  가자미튀김  모둠소세지구이  마늘쫑무...</td>
      <td>콩나물밥*양념장 (쌀,현미흑미:국내산) 어묵국  유산슬 (쇠고기:호주산) 아삭고추무...</td>
      <td>867.0</td>
      <td>560.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2016-02-03</td>
      <td>수</td>
      <td>2601</td>
      <td>56</td>
      <td>180</td>
      <td>111</td>
      <td>0.0</td>
      <td>모닝롤/베이글  우유/두유/주스 계란후라이  표고버섯죽/쌀밥 (쌀:국내산) 콩나물국...</td>
      <td>카레덮밥 (쌀,현미흑미:국내산) 팽이장국  치킨핑거 (닭고기:국내산) 쫄면야채무침 ...</td>
      <td>쌀밥/잡곡밥 (쌀,현미흑미:국내산) 청국장찌개  황태양념구이 (황태:러시아산) 고기...</td>
      <td>1017.0</td>
      <td>573.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2016-02-04</td>
      <td>목</td>
      <td>2601</td>
      <td>104</td>
      <td>220</td>
      <td>355</td>
      <td>0.0</td>
      <td>모닝롤/토마토샌드  우유/두유/주스 계란후라이  닭죽/쌀밥 (쌀,닭:국내산) 근대국...</td>
      <td>쌀밥/잡곡밥 (쌀,현미흑미:국내산) 쇠고기무국  주꾸미볶음  부추전  시금치나물  ...</td>
      <td>미니김밥*겨자장 (쌀,현미흑미:국내산) 우동  멕시칸샐러드  군고구마  무피클  포...</td>
      <td>978.0</td>
      <td>525.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016-02-05</td>
      <td>금</td>
      <td>2601</td>
      <td>278</td>
      <td>181</td>
      <td>34</td>
      <td>0.0</td>
      <td>모닝롤/와플  우유/두유/주스 계란후라이  쇠고기죽/쌀밥 (쌀:국내산) 재첩국  방...</td>
      <td>쌀밥/잡곡밥 (쌀,현미흑미:국내산) 떡국  돈육씨앗강정 (돼지고기:국내산) 우엉잡채...</td>
      <td>쌀밥/잡곡밥 (쌀,현미흑미:국내산) 차돌박이찌개 (쇠고기:호주산) 닭갈비 (닭고기:...</td>
      <td>925.0</td>
      <td>330.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
test = pd.read_csv("data/test.csv", parse_dates=["일자"])
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>일자</th>
      <th>요일</th>
      <th>본사정원수</th>
      <th>본사휴가자수</th>
      <th>본사출장자수</th>
      <th>본사시간외근무명령서승인건수</th>
      <th>현본사소속재택근무자수</th>
      <th>조식메뉴</th>
      <th>중식메뉴</th>
      <th>석식메뉴</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-27</td>
      <td>수</td>
      <td>2983</td>
      <td>88</td>
      <td>182</td>
      <td>5</td>
      <td>358.0</td>
      <td>모닝롤/연유버터베이글 우유/주스 계란후라이/찐계란 단호박죽/흑미밥 우거지국 고기완자...</td>
      <td>쌀밥/흑미밥/찰현미밥 대구지리 매운돈갈비찜 오꼬노미계란말이 상추무침 포기김치 양상추...</td>
      <td>흑미밥 얼큰순두부찌개 쇠고기우엉볶음 버섯햄볶음 (New)아삭이고추무절임 포기김치</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-28</td>
      <td>목</td>
      <td>2983</td>
      <td>104</td>
      <td>212</td>
      <td>409</td>
      <td>348.0</td>
      <td>모닝롤/대만샌드위치 우유/주스 계란후라이/찐계란 누룽지탕/흑미밥 황태국 시래기지짐 ...</td>
      <td>쌀밥/보리밥/찰현미밥 우렁된장찌개 오리주물럭 청양부추전 수제삼색무쌈 겉절이김치 양상...</td>
      <td>충무김밥 우동국물 오징어무침 꽃맛살샐러드 얼갈이쌈장무침 석박지</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-29</td>
      <td>금</td>
      <td>2983</td>
      <td>270</td>
      <td>249</td>
      <td>0</td>
      <td>294.0</td>
      <td>모닝롤/핫케익 우유/주스 계란후라이/찐계란 오곡죽/흑미밥 매생이굴국 고구마순볶음 양...</td>
      <td>쌀밥/흑미밥/찰현미밥 팽이장국 수제돈까스*소스 가자미조림 동초나물무침 포기김치 양상...</td>
      <td>흑미밥 물만둣국 카레찜닭 숯불양념꼬지어묵 꼬시래기무침 포기김치</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-02-01</td>
      <td>월</td>
      <td>2924</td>
      <td>108</td>
      <td>154</td>
      <td>538</td>
      <td>322.0</td>
      <td>모닝롤/촉촉한치즈케익 우유/주스 계란후라이/찐계란 누룽지탕/흑미밥 두부김칫국 새우완...</td>
      <td>쌀밥/흑미밥/찰현미밥 배추들깨국 오리대패불고기 시금치프리타타 부추고추장무침 포기김치...</td>
      <td>흑미밥 동태탕 돈육꽈리고추장조림 당면채소무침 모자반무침 포기김치</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-02-02</td>
      <td>화</td>
      <td>2924</td>
      <td>62</td>
      <td>186</td>
      <td>455</td>
      <td>314.0</td>
      <td>모닝롤/토마토샌드 우유/주스 계란후라이/찐계란 채소죽/흑미밥 호박맑은국 오이생채 양...</td>
      <td>쌀밥/팥밥/찰현미밥 부대찌개 닭살데리야끼조림 버섯탕수 세발나물무침 알타리김치/사과푸...</td>
      <td>흑미밥 바지락살국 쇠고기청경채볶음 두부구이*볶은김치 머위된장무침 백김치</td>
    </tr>
  </tbody>
</table>
</div>



## 3. 데이터 파악 및 전처리




```python
# 결측치 확인
train.isnull().sum()
```




    일자                0
    요일                0
    본사정원수             0
    본사휴가자수            0
    본사출장자수            0
    본사시간외근무명령서승인건수    0
    현본사소속재택근무자수       0
    조식메뉴              0
    중식메뉴              0
    석식메뉴              0
    중식계               0
    석식계               0
    dtype: int64




```python
test.isnull().sum()
```




    일자                0
    요일                0
    본사정원수             0
    본사휴가자수            0
    본사출장자수            0
    본사시간외근무명령서승인건수    0
    현본사소속재택근무자수       0
    조식메뉴              0
    중식메뉴              0
    석식메뉴              0
    dtype: int64




```python
# 중복값 확인
train[train.duplicated()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>일자</th>
      <th>요일</th>
      <th>본사정원수</th>
      <th>본사휴가자수</th>
      <th>본사출장자수</th>
      <th>본사시간외근무명령서승인건수</th>
      <th>현본사소속재택근무자수</th>
      <th>조식메뉴</th>
      <th>중식메뉴</th>
      <th>석식메뉴</th>
      <th>중식계</th>
      <th>석식계</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
# info 확인
train.info(), test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1205 entries, 0 to 1204
    Data columns (total 12 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   일자              1205 non-null   datetime64[ns]
     1   요일              1205 non-null   object        
     2   본사정원수           1205 non-null   int64         
     3   본사휴가자수          1205 non-null   int64         
     4   본사출장자수          1205 non-null   int64         
     5   본사시간외근무명령서승인건수  1205 non-null   int64         
     6   현본사소속재택근무자수     1205 non-null   float64       
     7   조식메뉴            1205 non-null   object        
     8   중식메뉴            1205 non-null   object        
     9   석식메뉴            1205 non-null   object        
     10  중식계             1205 non-null   float64       
     11  석식계             1205 non-null   float64       
    dtypes: datetime64[ns](1), float64(3), int64(4), object(4)
    memory usage: 113.1+ KB
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 50 entries, 0 to 49
    Data columns (total 10 columns):
     #   Column          Non-Null Count  Dtype         
    ---  ------          --------------  -----         
     0   일자              50 non-null     datetime64[ns]
     1   요일              50 non-null     object        
     2   본사정원수           50 non-null     int64         
     3   본사휴가자수          50 non-null     int64         
     4   본사출장자수          50 non-null     int64         
     5   본사시간외근무명령서승인건수  50 non-null     int64         
     6   현본사소속재택근무자수     50 non-null     float64       
     7   조식메뉴            50 non-null     object        
     8   중식메뉴            50 non-null     object        
     9   석식메뉴            50 non-null     object        
    dtypes: datetime64[ns](1), float64(1), int64(4), object(4)
    memory usage: 4.0+ KB
    




    (None, None)




```python
# describe 확인
train.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>본사정원수</th>
      <th>본사휴가자수</th>
      <th>본사출장자수</th>
      <th>본사시간외근무명령서승인건수</th>
      <th>현본사소속재택근무자수</th>
      <th>중식계</th>
      <th>석식계</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1205.000000</td>
      <td>1205.000000</td>
      <td>1205.000000</td>
      <td>1205.000000</td>
      <td>1205.000000</td>
      <td>1205.000000</td>
      <td>1205.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2807.815768</td>
      <td>157.913693</td>
      <td>241.142739</td>
      <td>274.117012</td>
      <td>43.506224</td>
      <td>890.334440</td>
      <td>461.772614</td>
    </tr>
    <tr>
      <th>std</th>
      <td>171.264404</td>
      <td>144.190572</td>
      <td>43.532298</td>
      <td>246.239651</td>
      <td>109.937400</td>
      <td>209.505057</td>
      <td>139.179202</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2601.000000</td>
      <td>23.000000</td>
      <td>41.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>296.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2645.000000</td>
      <td>71.000000</td>
      <td>217.000000</td>
      <td>4.000000</td>
      <td>0.000000</td>
      <td>758.000000</td>
      <td>406.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2760.000000</td>
      <td>105.000000</td>
      <td>245.000000</td>
      <td>299.000000</td>
      <td>0.000000</td>
      <td>879.000000</td>
      <td>483.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2962.000000</td>
      <td>185.000000</td>
      <td>272.000000</td>
      <td>452.000000</td>
      <td>0.000000</td>
      <td>1032.000000</td>
      <td>545.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3305.000000</td>
      <td>1224.000000</td>
      <td>378.000000</td>
      <td>1044.000000</td>
      <td>533.000000</td>
      <td>1459.000000</td>
      <td>905.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train.describe(include="object")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>요일</th>
      <th>조식메뉴</th>
      <th>중식메뉴</th>
      <th>석식메뉴</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1205</td>
      <td>1205</td>
      <td>1205</td>
      <td>1205</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>5</td>
      <td>1204</td>
      <td>1198</td>
      <td>1168</td>
    </tr>
    <tr>
      <th>top</th>
      <td>목</td>
      <td>모닝롤/프렌치토스트  우유/주스 계란후라이 누룽지탕/쌀밥 (쌀:국내산) 무채국  김...</td>
      <td>쌀밥/잡곡밥 (쌀:국내산) 시금치된장국  훈제오리구이  실곤약무침  무쌈/양파절임 ...</td>
      <td>*</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>244</td>
      <td>2</td>
      <td>2</td>
      <td>19</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>본사정원수</th>
      <th>본사휴가자수</th>
      <th>본사출장자수</th>
      <th>본사시간외근무명령서승인건수</th>
      <th>현본사소속재택근무자수</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2956.840000</td>
      <td>129.520000</td>
      <td>209.220000</td>
      <td>380.140000</td>
      <td>298.140000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>24.968846</td>
      <td>84.065873</td>
      <td>39.454593</td>
      <td>346.564304</td>
      <td>52.058056</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2924.000000</td>
      <td>50.000000</td>
      <td>131.000000</td>
      <td>0.000000</td>
      <td>179.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2924.000000</td>
      <td>78.250000</td>
      <td>176.500000</td>
      <td>1.000000</td>
      <td>257.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2974.000000</td>
      <td>95.000000</td>
      <td>202.500000</td>
      <td>465.500000</td>
      <td>300.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2975.000000</td>
      <td>137.500000</td>
      <td>245.250000</td>
      <td>681.000000</td>
      <td>333.250000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2983.000000</td>
      <td>489.000000</td>
      <td>279.000000</td>
      <td>1003.000000</td>
      <td>413.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
test.describe(include="object")
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>요일</th>
      <th>조식메뉴</th>
      <th>중식메뉴</th>
      <th>석식메뉴</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>5</td>
      <td>50</td>
      <td>50</td>
      <td>50</td>
    </tr>
    <tr>
      <th>top</th>
      <td>수</td>
      <td>모닝롤/연유버터베이글 우유/주스 계란후라이/찐계란 단호박죽/흑미밥 우거지국 고기완자...</td>
      <td>쌀밥/흑미밥/찰현미밥 대구지리 매운돈갈비찜 오꼬노미계란말이 상추무침 포기김치 양상추...</td>
      <td>흑미밥 얼큰순두부찌개 쇠고기우엉볶음 버섯햄볶음 (New)아삭이고추무절임 포기김치</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>11</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 컬럼명 변경
train_cols=["일자","요일","전체","휴가","출장","시간외근무", "재택근무", "조식","중식","석식","중식계","석식계"]
train.columns = train_cols
```


```python
test_cols=["일자","요일","전체","휴가","출장","시간외근무", "재택근무", "조식","중식","석식"]
test.columns = test_cols
```


```python
# 조식 컬럼 제외
train = train.drop(columns=["조식"])
test = test.drop(columns=["조식"])
```


```python
# 요일을 숫자로 바꾸기
train['요일'] = train['요일'].map({'월':1, '화':2, '수':3, '목':4, '금':5})
test['요일'] = test['요일'].map({'월':1, '화':2, '수':3, '목':4, '금':5})
```


```python
# 날짜를 월, 일 컬럼으로 

train['월'] = pd.DatetimeIndex(train['일자']).month
test['월'] = pd.DatetimeIndex(test['일자']).month
train['일'] = pd.DatetimeIndex(train['일자']).day
test['일'] = pd.DatetimeIndex(test['일자']).day
```


```python
# 상관관계
train_corr = train.corr()
```


```python
# 상관관계 히트맵

mask = np.triu(np.ones_like(train_corr))
sns.heatmap(train_corr, annot=True, fmt=".2f", mask = mask, cmap="coolwarm",vmin=-1,vmax=1);
```


    
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-mini_project_4/output_21_0.png)
    



```python
# 메뉴 전처리
# 밥, 반찬과 그 외로 인코딩
train["중식_밥반찬"] = (train["중식"].map(lambda x : x.split("/")[0]).str.contains("쌀밥|흑미밥|잡곡밥", regex=True)).astype(int)
train["중식_밥반찬"].value_counts()
```




    1    1006
    0     199
    Name: 중식_밥반찬, dtype: int64




```python
test["중식_밥반찬"] = (test["중식"].map(lambda x : x.split("/")[0]).str.contains("쌀밥|흑미밥|잡곡밥", regex=True)).astype(int)
test["중식_밥반찬"].value_counts()
```




    1    48
    0     2
    Name: 중식_밥반찬, dtype: int64




```python
train["석식_밥반찬"] = (train["석식"].map(lambda x : x.split("/")[0]).str.contains("쌀밥|흑미밥|잡곡밥", regex=True)).astype(int)
train["석식_밥반찬"].value_counts()
```




    1    755
    0    450
    Name: 석식_밥반찬, dtype: int64




```python
test["석식_밥반찬"] = (test["석식"].map(lambda x : x.split("/")[0]).str.contains("쌀밥|흑미밥|잡곡밥", regex=True)).astype(int)
test["석식_밥반찬"].value_counts()
```




    1    36
    0    14
    Name: 석식_밥반찬, dtype: int64




```python
# 출근 파생 변수 생성
train["출근"] = train["전체"] - train["휴가"] - train["재택근무"] - train["출장"]
test["출근"] = test["전체"] - test["휴가"] - test["재택근무"]- test["출장"]
```

### 중식계 예측 train data


```python
train[train["중식계"]==0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>일자</th>
      <th>요일</th>
      <th>전체</th>
      <th>휴가</th>
      <th>출장</th>
      <th>시간외근무</th>
      <th>재택근무</th>
      <th>중식</th>
      <th>석식</th>
      <th>중식계</th>
      <th>석식계</th>
      <th>월</th>
      <th>일</th>
      <th>중식_밥반찬</th>
      <th>석식_밥반찬</th>
      <th>출근</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



- 중식계에는 0인 데이터가 없어 train 데이터 그대로 사용

### 석식계 예측 train data


```python
train[train["석식계"]==0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>일자</th>
      <th>요일</th>
      <th>전체</th>
      <th>휴가</th>
      <th>출장</th>
      <th>시간외근무</th>
      <th>재택근무</th>
      <th>중식</th>
      <th>석식</th>
      <th>중식계</th>
      <th>석식계</th>
      <th>월</th>
      <th>일</th>
      <th>중식_밥반찬</th>
      <th>석식_밥반찬</th>
      <th>출근</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>204</th>
      <td>2016-11-30</td>
      <td>3</td>
      <td>2689</td>
      <td>68</td>
      <td>207</td>
      <td>0</td>
      <td>0.0</td>
      <td>나물비빔밥 (쌀:국내산) 가쯔오장국  치킨핑거*요거트D  감자샐러드  오복지무침  ...</td>
      <td>*</td>
      <td>1109.0</td>
      <td>0.0</td>
      <td>11</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>2414.0</td>
    </tr>
    <tr>
      <th>224</th>
      <td>2016-12-28</td>
      <td>3</td>
      <td>2705</td>
      <td>166</td>
      <td>225</td>
      <td>0</td>
      <td>0.0</td>
      <td>콩나물밥 (쌀:국내산) 가쯔오장국  미트볼케찹조림  꽃맛살샐러드  군고구마  배추겉...</td>
      <td>*</td>
      <td>767.0</td>
      <td>0.0</td>
      <td>12</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>2314.0</td>
    </tr>
    <tr>
      <th>244</th>
      <td>2017-01-25</td>
      <td>3</td>
      <td>2697</td>
      <td>79</td>
      <td>203</td>
      <td>0</td>
      <td>0.0</td>
      <td>카레덮밥 (쌀:국내산) 맑은국  유린기  개성감자만두  오이사과무침  포기김치 (김...</td>
      <td>*</td>
      <td>720.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>2415.0</td>
    </tr>
    <tr>
      <th>262</th>
      <td>2017-02-22</td>
      <td>3</td>
      <td>2632</td>
      <td>75</td>
      <td>252</td>
      <td>0</td>
      <td>0.0</td>
      <td>나물비빔밥 (쌀:국내산) 유부장국  생선까스*탈탈소스  파스타샐러드  마늘쫑볶음  ...</td>
      <td>*</td>
      <td>1065.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>22</td>
      <td>0</td>
      <td>0</td>
      <td>2305.0</td>
    </tr>
    <tr>
      <th>281</th>
      <td>2017-03-22</td>
      <td>3</td>
      <td>2627</td>
      <td>53</td>
      <td>235</td>
      <td>0</td>
      <td>0.0</td>
      <td>쌀밥/잡곡밥 (쌀:국내산) 돈육김치찌개  유린기  비엔나볶음  세발나물  깍두기 (...</td>
      <td>*</td>
      <td>953.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>22</td>
      <td>1</td>
      <td>0</td>
      <td>2339.0</td>
    </tr>
    <tr>
      <th>306</th>
      <td>2017-04-26</td>
      <td>3</td>
      <td>2626</td>
      <td>45</td>
      <td>304</td>
      <td>0</td>
      <td>0.0</td>
      <td>비빔밥 (쌀:국내산) 맑은국  오징어튀김  견과류조림  하와이안샐러드  깍두기 (김...</td>
      <td>*</td>
      <td>835.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>2277.0</td>
    </tr>
    <tr>
      <th>327</th>
      <td>2017-05-31</td>
      <td>3</td>
      <td>2637</td>
      <td>43</td>
      <td>265</td>
      <td>0</td>
      <td>0.0</td>
      <td>열무보리비빔밥 (쌀:국내산) 가쯔오장국  탕수만두  콥샐러드  오이지무침  포기김치...</td>
      <td>자기계발의날</td>
      <td>910.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>31</td>
      <td>0</td>
      <td>0</td>
      <td>2329.0</td>
    </tr>
    <tr>
      <th>346</th>
      <td>2017-06-28</td>
      <td>3</td>
      <td>2648</td>
      <td>58</td>
      <td>259</td>
      <td>0</td>
      <td>0.0</td>
      <td>콩나물밥 (쌀:국내산) 얼갈이된장국  삼치구이  잡채  아삭고추무침  깍두기 (김치...</td>
      <td>*자기계발의날*</td>
      <td>745.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>2331.0</td>
    </tr>
    <tr>
      <th>366</th>
      <td>2017-07-26</td>
      <td>3</td>
      <td>2839</td>
      <td>254</td>
      <td>246</td>
      <td>0</td>
      <td>0.0</td>
      <td>나물비빔밥  미소장국  파스타샐러드  소세지오븐구이  오렌지  포기김치 (김치:국내산)</td>
      <td>가정의날</td>
      <td>797.0</td>
      <td>0.0</td>
      <td>7</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>2339.0</td>
    </tr>
    <tr>
      <th>392</th>
      <td>2017-09-01</td>
      <td>5</td>
      <td>2642</td>
      <td>177</td>
      <td>303</td>
      <td>45</td>
      <td>0.0</td>
      <td>쌀밥/잡곡밥 (쌀:국내산) 시래기국  훈제오리구이  두부구이*양념장  쌈무/양파절임...</td>
      <td>*</td>
      <td>663.0</td>
      <td>0.0</td>
      <td>9</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>2162.0</td>
    </tr>
    <tr>
      <th>410</th>
      <td>2017-09-27</td>
      <td>3</td>
      <td>2642</td>
      <td>70</td>
      <td>265</td>
      <td>0</td>
      <td>0.0</td>
      <td>쌀밥/잡곡밥 (쌀:국내산) 콩나물국  삼겹살구이  어묵볶음  상추파무침  포기김치 ...</td>
      <td>쌀밥/잡곡밥 (쌀:국내산) 된장찌개  미니함박조림  계란말이  비름나물  포기김치 ...</td>
      <td>1023.0</td>
      <td>0.0</td>
      <td>9</td>
      <td>27</td>
      <td>1</td>
      <td>1</td>
      <td>2307.0</td>
    </tr>
    <tr>
      <th>412</th>
      <td>2017-09-29</td>
      <td>5</td>
      <td>2642</td>
      <td>214</td>
      <td>248</td>
      <td>22</td>
      <td>0.0</td>
      <td>쌀밥/잡곡밥 (쌀:국내산) 미역국  쇠불고기/잡채  오징어숙회무침  미니케익/식혜 ...</td>
      <td>*</td>
      <td>760.0</td>
      <td>0.0</td>
      <td>9</td>
      <td>29</td>
      <td>1</td>
      <td>0</td>
      <td>2180.0</td>
    </tr>
    <tr>
      <th>424</th>
      <td>2017-10-25</td>
      <td>3</td>
      <td>2645</td>
      <td>75</td>
      <td>289</td>
      <td>0</td>
      <td>0.0</td>
      <td>곤드레밥*강된장 (쌀:국내산) 가쯔오장국  갈치조림  쇠고기잡채  쑥갓두부무침  알...</td>
      <td>*</td>
      <td>786.0</td>
      <td>0.0</td>
      <td>10</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>2281.0</td>
    </tr>
    <tr>
      <th>449</th>
      <td>2017-11-29</td>
      <td>3</td>
      <td>2644</td>
      <td>78</td>
      <td>261</td>
      <td>0</td>
      <td>0.0</td>
      <td>나물비빔밥 (쌀:국내산) 미소장국  코다리조림  과일샐러드  군고구마  깍두기 (김...</td>
      <td>*</td>
      <td>903.0</td>
      <td>0.0</td>
      <td>11</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>2305.0</td>
    </tr>
    <tr>
      <th>468</th>
      <td>2017-12-27</td>
      <td>3</td>
      <td>2665</td>
      <td>169</td>
      <td>255</td>
      <td>0</td>
      <td>0.0</td>
      <td>쌀밥/잡곡밥 (쌀:국내산) 쇠고기미역국  오징어볶음  동그랑땡전  무쌈말이  포기김...</td>
      <td>*</td>
      <td>571.0</td>
      <td>0.0</td>
      <td>12</td>
      <td>27</td>
      <td>1</td>
      <td>0</td>
      <td>2241.0</td>
    </tr>
    <tr>
      <th>492</th>
      <td>2018-01-31</td>
      <td>3</td>
      <td>2655</td>
      <td>56</td>
      <td>223</td>
      <td>0</td>
      <td>0.0</td>
      <td>김치제육덮밥  미소장국  양장피잡채  계란찜  아삭고추무침/귤  알타리김치 (김치:...</td>
      <td>*</td>
      <td>1138.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>31</td>
      <td>0</td>
      <td>0</td>
      <td>2376.0</td>
    </tr>
    <tr>
      <th>502</th>
      <td>2018-02-14</td>
      <td>3</td>
      <td>2707</td>
      <td>418</td>
      <td>159</td>
      <td>0</td>
      <td>0.0</td>
      <td>쌀밥/잡곡밥 (쌀:국내산) 떡국  버섯불고기  오징어숙회무침  취나물  배추겉절이 ...</td>
      <td>쌀밥/잡곡밥 (쌀:국내산) 쇠고기무국  고추잡채*꽃빵  계란찜  오이무침  포기김치...</td>
      <td>850.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>14</td>
      <td>1</td>
      <td>1</td>
      <td>2130.0</td>
    </tr>
    <tr>
      <th>510</th>
      <td>2018-02-28</td>
      <td>3</td>
      <td>2707</td>
      <td>134</td>
      <td>278</td>
      <td>0</td>
      <td>0.0</td>
      <td>곤드레밥*강된장 (쌀:국내산) 어묵국  치킨핑거*요거트소스  도토리묵무침  콩조림 ...</td>
      <td>*</td>
      <td>786.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>2295.0</td>
    </tr>
    <tr>
      <th>529</th>
      <td>2018-03-28</td>
      <td>3</td>
      <td>2714</td>
      <td>45</td>
      <td>252</td>
      <td>0</td>
      <td>0.0</td>
      <td>단호박카레라이스 (쌀:국내산) 유부장국  유린기  볼어묵볶음  오복지  포기김치 (...</td>
      <td>*</td>
      <td>926.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>2417.0</td>
    </tr>
    <tr>
      <th>549</th>
      <td>2018-04-25</td>
      <td>3</td>
      <td>2714</td>
      <td>66</td>
      <td>285</td>
      <td>0</td>
      <td>0.0</td>
      <td>비빔밥 (쌀:국내산) 유부장국  오징어튀김  떡밤초  요플레  포기김치 (김치:국내산)</td>
      <td></td>
      <td>851.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>2363.0</td>
    </tr>
    <tr>
      <th>571</th>
      <td>2018-05-30</td>
      <td>3</td>
      <td>2721</td>
      <td>80</td>
      <td>281</td>
      <td>0</td>
      <td>0.0</td>
      <td>콩나물밥 (쌀:국내산) 유부장국  수제돈가스  파스타샐러드  무생채  포기김치 (김...</td>
      <td></td>
      <td>876.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>2360.0</td>
    </tr>
    <tr>
      <th>589</th>
      <td>2018-06-27</td>
      <td>3</td>
      <td>2728</td>
      <td>66</td>
      <td>277</td>
      <td>0</td>
      <td>0.0</td>
      <td>카레덮밥 (쌀:국내산) 가쯔오장국  깐풍육  구운채소 *발사믹소스 오복지무침  포기...</td>
      <td>*</td>
      <td>957.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>2385.0</td>
    </tr>
    <tr>
      <th>609</th>
      <td>2018-07-25</td>
      <td>3</td>
      <td>2704</td>
      <td>226</td>
      <td>256</td>
      <td>1</td>
      <td>0.0</td>
      <td>쌀밥/잡곡밥 (쌀:국내산) 쇠고기샤브국  유린기  사각어묵볶음  오이사과생채  포기...</td>
      <td></td>
      <td>760.0</td>
      <td>0.0</td>
      <td>7</td>
      <td>25</td>
      <td>1</td>
      <td>0</td>
      <td>2222.0</td>
    </tr>
    <tr>
      <th>633</th>
      <td>2018-08-29</td>
      <td>3</td>
      <td>2996</td>
      <td>103</td>
      <td>258</td>
      <td>0</td>
      <td>0.0</td>
      <td>콩나물밥 (쌀:국내산) 팽이장국  치킨핑거  메추리알조림  과일샐러드  배추겉절이 ...</td>
      <td>*</td>
      <td>915.0</td>
      <td>0.0</td>
      <td>8</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>2635.0</td>
    </tr>
    <tr>
      <th>648</th>
      <td>2018-09-19</td>
      <td>3</td>
      <td>2763</td>
      <td>77</td>
      <td>288</td>
      <td>0</td>
      <td>0.0</td>
      <td>카레덮밥 (쌀:국내산) 유부장국  감자프리타타  메밀전병만두  쨔샤이무침/과일  포...</td>
      <td></td>
      <td>833.0</td>
      <td>0.0</td>
      <td>9</td>
      <td>19</td>
      <td>0</td>
      <td>0</td>
      <td>2398.0</td>
    </tr>
    <tr>
      <th>667</th>
      <td>2018-10-31</td>
      <td>3</td>
      <td>2805</td>
      <td>111</td>
      <td>306</td>
      <td>0</td>
      <td>0.0</td>
      <td>쌀밥/잡곡밥 (쌀:국내산) 콩나물국  수제돈가스  닭살겨자채  반달호박나물  포기김...</td>
      <td>자기계발의날</td>
      <td>930.0</td>
      <td>0.0</td>
      <td>10</td>
      <td>31</td>
      <td>1</td>
      <td>0</td>
      <td>2388.0</td>
    </tr>
    <tr>
      <th>687</th>
      <td>2018-11-28</td>
      <td>3</td>
      <td>2815</td>
      <td>69</td>
      <td>298</td>
      <td>1</td>
      <td>0.0</td>
      <td>나물비빔밥 (쌀:국내산) 가쯔오장국  오징어튀김 (오징어:뉴질랜드) 과일샐러드  군...</td>
      <td>*</td>
      <td>862.0</td>
      <td>0.0</td>
      <td>11</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>2448.0</td>
    </tr>
    <tr>
      <th>706</th>
      <td>2018-12-26</td>
      <td>3</td>
      <td>2846</td>
      <td>184</td>
      <td>241</td>
      <td>0</td>
      <td>0.0</td>
      <td>쌀밥/잡곡밥 (쌀:국내산) 아욱국  돈육굴소스볶음  골뱅이무침*소면  얼갈이나물  ...</td>
      <td>자기계발의날</td>
      <td>695.0</td>
      <td>0.0</td>
      <td>12</td>
      <td>26</td>
      <td>1</td>
      <td>0</td>
      <td>2421.0</td>
    </tr>
    <tr>
      <th>730</th>
      <td>2019-01-30</td>
      <td>3</td>
      <td>2985</td>
      <td>66</td>
      <td>226</td>
      <td>1</td>
      <td>0.0</td>
      <td>카레덮밥 (쌀:국내산,돈육:국내산) 유부장국  새우까스*칠리소스  쫄면무침  오이무...</td>
      <td>자기개발의날</td>
      <td>679.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>2693.0</td>
    </tr>
    <tr>
      <th>747</th>
      <td>2019-02-27</td>
      <td>3</td>
      <td>2806</td>
      <td>100</td>
      <td>274</td>
      <td>0</td>
      <td>0.0</td>
      <td>비빔밥 (쌀:국내산) 유부장국  오징어튀김  떡밤초  음료  포기김치 (김치:국내산)</td>
      <td>*  자기계발의날  *</td>
      <td>944.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>2432.0</td>
    </tr>
    <tr>
      <th>766</th>
      <td>2019-03-27</td>
      <td>3</td>
      <td>2836</td>
      <td>92</td>
      <td>259</td>
      <td>0</td>
      <td>0.0</td>
      <td>단호박영양밥 (쌀:국내산) 가쯔오장국  돈육칠리강정  모듬묵샐러드  숙주나물  배추...</td>
      <td>*  자기개발의날  *</td>
      <td>856.0</td>
      <td>0.0</td>
      <td>3</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>2485.0</td>
    </tr>
    <tr>
      <th>786</th>
      <td>2019-04-24</td>
      <td>3</td>
      <td>2822</td>
      <td>59</td>
      <td>273</td>
      <td>0</td>
      <td>0.0</td>
      <td>카레라이스 (쌀:국내산) 미소장국  언양식불고기  떡볶이  방울토마토  포기김치 (...</td>
      <td>*  자기계발의날  *</td>
      <td>1034.0</td>
      <td>0.0</td>
      <td>4</td>
      <td>24</td>
      <td>0</td>
      <td>0</td>
      <td>2490.0</td>
    </tr>
    <tr>
      <th>809</th>
      <td>2019-05-29</td>
      <td>3</td>
      <td>2825</td>
      <td>50</td>
      <td>237</td>
      <td>0</td>
      <td>0.0</td>
      <td>쌀밥/잡곡밥 (쌀:국내산) 배추된장국  수제돈가스  마파두부  돈나물유자청무침  포...</td>
      <td>*  자기개발의날  *</td>
      <td>896.0</td>
      <td>0.0</td>
      <td>5</td>
      <td>29</td>
      <td>1</td>
      <td>0</td>
      <td>2538.0</td>
    </tr>
    <tr>
      <th>828</th>
      <td>2019-06-26</td>
      <td>3</td>
      <td>2758</td>
      <td>69</td>
      <td>282</td>
      <td>0</td>
      <td>0.0</td>
      <td>카레덮밥 (쌀,돈육:국내산) 가쯔오장국  고구마치즈구이  쫄면무침  무말랭이  포기...</td>
      <td>*  자기개발의날  *</td>
      <td>946.0</td>
      <td>0.0</td>
      <td>6</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>2407.0</td>
    </tr>
    <tr>
      <th>853</th>
      <td>2019-07-31</td>
      <td>3</td>
      <td>2760</td>
      <td>495</td>
      <td>231</td>
      <td>0</td>
      <td>0.0</td>
      <td>곤드레밥*양념장 (쌀:국내산) 맑은국  해물누룽지탕 (오징어:원양산) 메밀전병만두 ...</td>
      <td>자기계발의날</td>
      <td>619.0</td>
      <td>0.0</td>
      <td>7</td>
      <td>31</td>
      <td>0</td>
      <td>0</td>
      <td>2034.0</td>
    </tr>
    <tr>
      <th>872</th>
      <td>2019-08-28</td>
      <td>3</td>
      <td>3305</td>
      <td>123</td>
      <td>274</td>
      <td>0</td>
      <td>0.0</td>
      <td>카레덮밥 (쌀,돈육:국내산) 맑은국  치킨핑거 (닭:국내산) 쫄면야채무침  오복지/...</td>
      <td>*</td>
      <td>899.0</td>
      <td>0.0</td>
      <td>8</td>
      <td>28</td>
      <td>0</td>
      <td>0</td>
      <td>2908.0</td>
    </tr>
    <tr>
      <th>890</th>
      <td>2019-09-25</td>
      <td>3</td>
      <td>3111</td>
      <td>60</td>
      <td>285</td>
      <td>1</td>
      <td>0.0</td>
      <td>곤드레밥*양념장 (쌀:국내산) 맑은국  돈육강정 (돈육:국내산) 사과고구마그라탕  ...</td>
      <td>*</td>
      <td>803.0</td>
      <td>0.0</td>
      <td>9</td>
      <td>25</td>
      <td>0</td>
      <td>0</td>
      <td>2766.0</td>
    </tr>
    <tr>
      <th>912</th>
      <td>2019-10-30</td>
      <td>3</td>
      <td>3121</td>
      <td>122</td>
      <td>294</td>
      <td>1</td>
      <td>0.0</td>
      <td>마파두부덮밥 (쌀,돈육:국내산) 맑은국  치킨핑거 (닭:국내산) 시저샐러드  무비트...</td>
      <td>*</td>
      <td>771.0</td>
      <td>0.0</td>
      <td>10</td>
      <td>30</td>
      <td>0</td>
      <td>0</td>
      <td>2705.0</td>
    </tr>
    <tr>
      <th>932</th>
      <td>2019-11-27</td>
      <td>3</td>
      <td>3104</td>
      <td>134</td>
      <td>288</td>
      <td>1</td>
      <td>0.0</td>
      <td>나물비빔밥 (쌀:국내산) 맑은국  감자치즈구이  군만두  치커리유자청생채  포기김치...</td>
      <td>*</td>
      <td>732.0</td>
      <td>0.0</td>
      <td>11</td>
      <td>27</td>
      <td>0</td>
      <td>0</td>
      <td>2682.0</td>
    </tr>
    <tr>
      <th>955</th>
      <td>2019-12-31</td>
      <td>2</td>
      <td>3111</td>
      <td>709</td>
      <td>149</td>
      <td>22</td>
      <td>0.0</td>
      <td>쌀밥/잡곡밥 (쌀:국내산) 배추된장국  닭볶음탕 (닭:국내산) 부추깻잎전  양배추쌈...</td>
      <td>*</td>
      <td>349.0</td>
      <td>0.0</td>
      <td>12</td>
      <td>31</td>
      <td>1</td>
      <td>0</td>
      <td>2253.0</td>
    </tr>
    <tr>
      <th>973</th>
      <td>2020-01-29</td>
      <td>3</td>
      <td>2821</td>
      <td>101</td>
      <td>214</td>
      <td>4</td>
      <td>0.0</td>
      <td>콩나물밥*양념장 (쌀:국내산,소고기:호주) 가쯔오장국  치킨핑거 (닭:국내산) 꽃맛...</td>
      <td>자기개발의날</td>
      <td>1197.0</td>
      <td>0.0</td>
      <td>1</td>
      <td>29</td>
      <td>0</td>
      <td>0</td>
      <td>2506.0</td>
    </tr>
    <tr>
      <th>993</th>
      <td>2020-02-26</td>
      <td>3</td>
      <td>2872</td>
      <td>109</td>
      <td>190</td>
      <td>4</td>
      <td>0.0</td>
      <td>낙지비빔밥 (쌀:국내,낙지:중국산) 팽이장국  치킨텐더*콘소스D (닭:국내산) 과일...</td>
      <td>자기개발의날</td>
      <td>1105.0</td>
      <td>0.0</td>
      <td>2</td>
      <td>26</td>
      <td>0</td>
      <td>0</td>
      <td>2573.0</td>
    </tr>
    <tr>
      <th>1166</th>
      <td>2020-11-25</td>
      <td>3</td>
      <td>3021</td>
      <td>206</td>
      <td>191</td>
      <td>3</td>
      <td>387.0</td>
      <td>쌀밥/흑미밥/찰현미밥 콩비지김치찌개 해물누룽지탕 탕평채 고추장감자조림 깍두기/수제과...</td>
      <td>＜자기 계발의 날＞</td>
      <td>1146.0</td>
      <td>0.0</td>
      <td>11</td>
      <td>25</td>
      <td>1</td>
      <td>0</td>
      <td>2237.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
train[train["석식계"]==0]["요일"].value_counts()
```




    3    40
    5     2
    2     1
    Name: 요일, dtype: int64



- 석식계에는 0인 데이터가 존재, 대체로 달의 마지막 주 수요일에 해당

- 마지막 주 수요일의 석식계 평균값으로 대치


```python
# 마지막주 수요일의 석식계 평균으로 대체
dinner_nan = train.loc[(train["요일"] == 2) & (train["일"].isin(range(24,32))), "석식계"].replace(0, np.nan).mean()
dinner_nan
```




    522.8983050847457




```python
train.loc[train["석식계"]==0,"석식계"]=dinner_nan
train[train["석식계"]==0]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>일자</th>
      <th>요일</th>
      <th>전체</th>
      <th>휴가</th>
      <th>출장</th>
      <th>시간외근무</th>
      <th>재택근무</th>
      <th>중식</th>
      <th>석식</th>
      <th>중식계</th>
      <th>석식계</th>
      <th>월</th>
      <th>일</th>
      <th>중식_밥반찬</th>
      <th>석식_밥반찬</th>
      <th>출근</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



- 식단이 있으나 석식계가 0인 2개 행은 제거


```python
no = train[(train["석식계"]==0) & (train["석식"].map(lambda x : x.split("/")[0]).str.contains("쌀밥|흑미밥|잡곡밥",regex=True))].index
```


```python
train_dinner = train.drop(no)
```


```python
plt.figure(figsize=(10,5))
sns.lineplot(data=train, x="일자", y="전체");
```


    
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-mini_project_4/output_39_0.png)
    



```python
plt.figure(figsize=(10,1))
sns.boxplot(data=train, x="시간외근무");
```


    
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-mini_project_4/output_40_0.png)
    



```python
plt.figure(figsize=(10,3))
sns.stripplot(data=train, x="시간외근무");
```


    
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-mini_project_4/output_41_0.png)
    



```python
plt.figure(figsize=(10,3))
sns.stripplot(data=train, x="재택근무");
```


    
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-mini_project_4/output_42_0.png)
    



```python
plt.figure(figsize=(5, 2))
sns.barplot(data=train, x="요일", y="출장", errorbar=None);
```


    
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-mini_project_4/output_43_0.png)
    



```python
plt.figure(figsize=(5, 2))
sns.barplot(data=train, x="요일", y="시간외근무",errorbar=None);
```


    
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-mini_project_4/output_44_0.png)
    



```python
plt.figure(figsize=(5, 2))
sns.barplot(data=train, x='요일', y='중식계',errorbar=None);
```


    
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-mini_project_4/output_45_0.png)
    



```python
plt.figure(figsize=(5, 2))
sns.barplot(data=train, x='요일', y='석식계',errorbar=None);
```


    
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-mini_project_4/output_46_0.png)
    



```python
df = train[["요일","중식계","석식계"]]
df.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>요일</th>
      <th>중식계</th>
      <th>석식계</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>요일</th>
      <td>1.000000</td>
      <td>-0.731563</td>
      <td>-0.396205</td>
    </tr>
    <tr>
      <th>중식계</th>
      <td>-0.731563</td>
      <td>1.000000</td>
      <td>0.633010</td>
    </tr>
    <tr>
      <th>석식계</th>
      <td>-0.396205</td>
      <td>0.633010</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize = (20, 10))
sns.histplot(train["중식계"], ax = ax[0])
sns.histplot(train["석식계"], ax = ax[1])
plt.show()
```


    
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-mini_project_4/output_48_0.png)
    



```python
plt.figure(figsize=(5, 2))
sns.barplot(data=train, x='요일', y='휴가',errorbar=None);
```


    
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-mini_project_4/output_49_0.png)
    



```python
plt.figure(figsize=(5, 2))
sns.barplot(data=train, x='요일', y='재택근무',errorbar=None);
```


    
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-mini_project_4/output_50_0.png)
    



```python
plt.figure(figsize=(5, 2))
sns.barplot(data=train, x='요일', y='출장',errorbar=None);
```


    
![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-07-mini_project_4/output_51_0.png)
    


## 4. 학습, 예측 데이터셋 구성


```python
label_name1 = "중식계"
label_name2= "석식계"
```


```python
train.columns
```




    Index(['일자', '요일', '전체', '휴가', '출장', '시간외근무', '재택근무', '중식', '석식', '중식계', '석식계',
           '월', '일', '중식_밥반찬', '석식_밥반찬', '출근'],
          dtype='object')




```python
feature_names1=["요일", "시간외근무","출근","중식_밥반찬","월","일"] 
feature_names2=["요일", "시간외근무","출근","석식_밥반찬","월","일"]
```

### 중식계 train data


```python
X_train1 = train[feature_names1]
display(X_train1.head(2))
X_train1.shape
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>요일</th>
      <th>시간외근무</th>
      <th>출근</th>
      <th>중식_밥반찬</th>
      <th>월</th>
      <th>일</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>238</td>
      <td>2401.0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>319</td>
      <td>2378.0</td>
      <td>1</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>





    (1205, 6)




```python
y_train1 = train[label_name1]
y_train1.shape
```




    (1205,)




```python
X_test1 = test[feature_names1]
X_test1.shape
```




    (50, 6)



### 석식계 train data


```python
X_train2 = train_dinner[feature_names2]
display(X_train2.head(2))
X_train2.shape
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>요일</th>
      <th>시간외근무</th>
      <th>출근</th>
      <th>석식_밥반찬</th>
      <th>월</th>
      <th>일</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>238</td>
      <td>2401.0</td>
      <td>1</td>
      <td>2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>319</td>
      <td>2378.0</td>
      <td>0</td>
      <td>2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>





    (1205, 6)




```python
y_train2 = train_dinner[label_name2]
y_train2.shape
```




    (1205,)




```python
X_test2 = test[feature_names2]
X_test2.shape
```




    (50, 6)



## 5. 머신러닝 알고리즘

### 중식계 예측


```python
from sklearn.ensemble import RandomForestRegressor
model1 = RandomForestRegressor(n_estimators=100, n_jobs=-1, criterion="absolute_error", random_state=42)
model1
```




    RandomForestRegressor(criterion='absolute_error', n_jobs=-1, random_state=42)




```python
model1.fit(X_train1, y_train1)
```




    RandomForestRegressor(criterion='absolute_error', n_jobs=-1, random_state=42)




```python
from sklearn.model_selection import cross_val_predict

y_predict1 = cross_val_predict(model1, X_train1, y_train1, cv=5, n_jobs=-1)
```


```python
from sklearn.metrics import r2_score

r2_score(y_train1, y_predict1)
```




    0.6551438143397688




```python
y_predict1 = model1.predict(X_test1)
y_predict1
```




    array([ 903.56 ,  923.11 ,  653.55 , 1204.23 , 1021.93 , 1005.78 ,
           1020.57 ,  790.28 , 1273.66 , 1017.39 ,  924.57 , 1260.055,
           1068.435,  938.81 ,  947.1  ,  675.63 , 1266.505, 1093.24 ,
           1000.47 ,  959.16 ,  675.92 , 1073.07 , 1038.59 , 1007.92 ,
            623.18 , 1310.545, 1094.41 ,  968.985,  966.19 ,  710.47 ,
           1278.36 ,  999.255,  985.165,  934.51 ,  685.76 , 1229.19 ,
           1005.535,  960.465,  934.91 ,  661.76 , 1266.49 , 1022.93 ,
           1044.45 ,  878.9  ,  717.55 , 1187.77 , 1001.495,  952.75 ,
            910.37 ,  725.09 ])




```python
y_predict1 = y_predict1.tolist()
```

### 석식계 예측


```python
model2 = RandomForestRegressor(n_estimators=100, n_jobs=-1, criterion="absolute_error",random_state=42)
model2
```




    RandomForestRegressor(criterion='absolute_error', n_jobs=-1, random_state=42)




```python
model2.fit(X_train2, y_train2)
```




    RandomForestRegressor(criterion='absolute_error', n_jobs=-1, random_state=42)




```python
y_predict2 = cross_val_predict(model2, X_train2, y_train2, cv=5, n_jobs=-1)
```


```python
r2_score(y_train2, y_predict2)
```




    0.3957326052969119




```python
y_predict2 = model2.predict(X_test2)
y_predict2
```




    array([370.84      , 506.93      , 265.81      , 556.54      ,
           547.1       , 514.01491525, 500.88      , 419.48389831,
           689.41      , 584.6       , 443.64949153, 709.94      ,
           728.09      , 503.43940678, 562.2       , 373.11898305,
           740.32      , 696.74      , 478.09101695, 558.47      ,
           377.27      , 710.47      , 475.25      , 608.1       ,
           397.94288136, 748.72      , 747.96      , 459.18      ,
           626.03      , 391.06898305, 741.47      , 647.        ,
           475.30694915, 550.73      , 356.72      , 656.53      ,
           660.26      , 508.68898305, 506.67      , 336.27      ,
           650.3       , 643.26      , 460.37898305, 498.89      ,
           362.6       , 616.12      , 633.78      , 495.83694915,
           495.75      , 376.74      ])




```python
y_predict2 = y_predict2.tolist()
```

### 제출


```python
submit = pd.read_csv("data/sample_submission.csv")
```


```python
submit.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>일자</th>
      <th>중식계</th>
      <th>석식계</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-27</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-28</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
submit["중식계"] = y_predict1
submit["석식계"] = y_predict2
```


```python
submit.head(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>일자</th>
      <th>중식계</th>
      <th>석식계</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-27</td>
      <td>903.56</td>
      <td>370.84</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-28</td>
      <td>923.11</td>
      <td>506.93</td>
    </tr>
  </tbody>
</table>
</div>




```python
submit.to_csv("data/submit_1.csv", index=False)
```


```python
pd.read_csv("data/submit_1.csv").head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>일자</th>
      <th>중식계</th>
      <th>석식계</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2021-01-27</td>
      <td>903.56</td>
      <td>370.84</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2021-01-28</td>
      <td>923.11</td>
      <td>506.93</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2021-01-29</td>
      <td>653.55</td>
      <td>265.81</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2021-02-01</td>
      <td>1204.23</td>
      <td>556.54</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2021-02-02</td>
      <td>1021.93</td>
      <td>547.10</td>
    </tr>
  </tbody>
</table>
</div>


