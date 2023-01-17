---
title: "KoNLPy, RNN"
excerpt: "2022-12-14 "KoNLPy, RNN"

# layout: post
categories:
  - TIL
tags:
  - python
  - Deep Learning
  - RNN
  - KoNLPy
  - IDF
  - CountVectorizer
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
### ⚠️ 해당 내용은 멋쟁이사자처럼 AI School 오늘코드 박조은 강사의 자료를 토대로 정리한 내용입니다.

## 라이브러리 로드

```python
import pandas as pd
import numpy as np
```

## 데이터 로드

```python
# 데이콘의 해당 데이터셋은 CC-BY-4.0 라이센스
# 데이터 출처 : https://dacon.io/competitions/official/235747/data
# 로컬 PC에서 실습 시 직접 데이콘 사이트에 회원가입하고 다운로드 요망

import os, platform
base_path = "data/klue/"

def file_exist_check(base_path):
    if os.path.exists(f"{base_path}train_data.csv"):
        print(f"{base_path} 경로에 파일이 이미 있음")
        return
    
    if platform.system() == "Linux":
        print(f"파일을 다운로드 하고 {base_path} 경로에 압축을 해제함")
        !wget https://bit.ly/dacon-klue-open-zip
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        !unzip dacon-klue-open-zip -d data/klue
    else:
        print(f"""https://dacon.io/competitions/official/235747/data 에서 다운로드 하고
              실습 경로 {base_path}에 옮겨주세요.""")
    return
    
file_exist_check(base_path)
```

```python
# 학습, 예측 데이터셋
train = pd.read_csv(f"{base_path}train_data.csv")
test = pd.read_csv(f"{base_path}test_data.csv")
train.shape, test.shape

out:
((45654, 3), (9131, 2))
```

```python
# 토픽
topic = pd.read_csv(f"{base_path}topic_dict.csv")
topic
```

|  | topic | topic_idx |
| --- | --- | --- |
| 0 | IT과학 | 0 |
| 1 | 경제 | 1 |
| 2 | 사회 | 2 |
| 3 | 생활문화 | 3 |
| 4 | 세계 | 4 |
| 5 | 스포츠 | 5 |
| 6 | 정치 |  |

## 문자 전처리

```python
# 정규표현식
import re

def preprocessing(text):
    # 한글, 영문, 숫자만 남기고 모두 제거하도록 합니다.
    text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9]', ' ', text)
    # 중복으로 생성된 공백값을 제거합니다.
    text = re.sub('[\s]+', ' ', text)
    # 영문자를 소문자로 만듭니다.
    text = text.lower()
    return text
```

```python
# !pip install tqdm --upgrade
# tqdm 으로 전처리 진행 상태를 표시
from tqdm import tqdm
tqdm.pandas() 

# map을 통해 전처리 일괄 적용
train["title"] = train["title"].progress_map(preprocessing)
test["title"] = test["title"].progress_map(preprocessing)

out:
100%|██████████| 45654/45654 [00:00<00:00, 122698.84it/s]
100%|██████████| 9131/9131 [00:00<00:00, 135279.58it/s]
```

## 형태소 분석

konlpy가 설치되어 있지 않다면 설치를 진행한다. konlpy는 다른 프로그래밍 언어(JAVA, C++)로 만들어진 형태소 분석기를 파이썬 인터페이스로 사용할 수 있는 도구이다. JPype1도 파이썬에서 자바를 사용할 수 있도록 하는 도구다. 인터페이스가 파이썬이지만 내부는 해당 언어로 동작하여 다른 언어도 함께 설치되어 있어야 한다. 그래서 설치는 꼭 공식문서를 참고하도록 하자.

⚠️실습을 목적으로 konlpy를 사용할 예정이라면 Colab 환경에서 하는 것을 추천한다. 위 설명에서도 언급하였지만, 다른 프로그래밍 언어로 만들어져있기 때문에 로컬 파이썬 환경에서 사용하려면 꽤나 번거로운 작업을 수행해야 한다. [konlpy 공식 홈페이지 설치가이드](https://konlpy.org/ko/latest/install/)를 참고하도록 하자. 만약, 윈도우 환경에 아나콘다를 사용하면 해당 [링크](https://velog.io/@soo-im/konlpy-%EC%84%A4%EC%B9%98-%EC%97%90%EB%9F%AC-%ED%95%B4%EA%B2%B0%EC%B1%85-%EC%95%84%EB%82%98%EC%BD%98%EB%8B%A4-JPYPE)를 참고하도록 한다.

### konlpy

```python
# colab 환경에서 설치
!pip install konlpy --upgrade
```

```python
small_text = "버스의 운행시간을 문의합니다. 어?!"
small_text

out:
버스의 운행시간을 문의합니다. 어?!
```

```python
kkma.morphs(u'공부를 하면 할수록 모르는것이 많다는것을 알게 됩니다.')

out:
['공부',
 '를',
 '하',
 '면',
 '하',
 'ㄹ수록',
 '모르',
 '는',
 '것',
 '이',
 '많',
 '다는',
 '것',
 '을',
 '알',
 '게',
 '되',
 'ㅂ니다',
 '.']
```

```python
kkma.pos(u'공부를 하면 할수록 모르는것이 많다는것을 알게 됩니다.')

out:
[('공부', 'NNG'),
 ('를', 'JKO'),
 ('하', 'VV'),
 ('면', 'ECE'),
 ('하', 'VV'),
 ('ㄹ수록', 'ECD'),
 ('모르', 'VV'),
 ('는', 'ETD'),
 ('것', 'NNB'),
 ('이', 'JKS'),
 ('많', 'VA'),
 ('다는', 'ETD'),
 ('것', 'NNB'),
 ('을', 'JKO'),
 ('알', 'VV'),
 ('게', 'ECD'),
 ('되', 'VV'),
 ('ㅂ니다', 'EFN'),
 ('.', 'SF')]
```

### Pecab

```python
!pip install pecab
```

```python
from pecab import PeCab

pecab = PeCab()
pecab.pos("저는 삼성디지털프라자에서 지펠냉장고를 샀어요.")

out:
[('저', 'NP'),
 ('는', 'JX'),
 ('삼성', 'NNP'),
 ('디지털', 'NNP'),
 ('프라자', 'NNP'),
 ('에서', 'JKB'),
 ('지', 'NNP'),
 ('펠', 'NNP'),
 ('냉장고', 'NNG'),
 ('를', 'JKO'),
 ('샀', 'VV+EP'),
 ('어요', 'EF'),
 ('.', 'SF')]
```

### [Stemming(어간 추출)](https://ko.wikipedia.org/wiki/%EC%96%B4%EA%B0%84_%EC%B6%94%EC%B6%9C)

형태론 및 정보 검색 분야에서 어형이 변형된 단어로부터 접사 등을 제거하고 그 단어의 어간을 분리해 내는 것을 의미한다. 여기서 어간은 반드시 어근과 같아야 할 필요는 없으며, 어근과 차이가 있더라도 관련이 있는 단어들이 일정하게 동일한 어간으로 맵핑되게 하는 것이 어간 추출의 목적이다. 1960년대부터 컴퓨터 과학 분야에서 다양한 어간 추출 관련 알고리즘들이 연구되어 왔다. 많은 웹 검색 엔진들은 동일한 어간을 가진 단어들을 동의어로 취급하는 방식으로 질의어 확장을 하여 검색 결과의 품질을 높인다.
어간 추출 프로그램은 흔히 스테밍 알고리즘(stemming algorithm) 또는 스테머(stemmer)라 불린다.

```python
# Okt
# steming 기능을 제공
from konlpy.tag import Okt

okt = Okt()
okt.pos(small_text)

out:
[('버스', 'Noun'),
 ('의', 'Josa'),
 ('운행', 'Noun'),
 ('시간', 'Noun'),
 ('을', 'Josa'),
 ('문의', 'Noun'),
 ('합니다', 'Verb'),
 ('.', 'Punctuation'),
 ('어', 'Eomi'),
 ('?!', 'Punctuation')]
```

```python
okt.pos(small_text, stem = True)

out:
[('버스', 'Noun'),
 ('의', 'Josa'),
 ('운행', 'Noun'),
 ('시간', 'Noun'),
 ('을', 'Josa'),
 ('문의', 'Noun'),
 ('하다', 'Verb'),
 ('.', 'Punctuation'),
 ('어', 'Eomi'),
 ('?!', 'Punctuation')]
```

형태소 분석기(Okt) 불러오기 
['Josa', 'Eomi', 'Punctuation'] : 조사, 어미, 구두점 제거
전체 텍스트에 적용해 주기 위해 함수를 만든다.
1) 텍스트 입력받기
2) 품사태깅 [('문의', 'Noun'), ('하다', 'Verb'), ('?!', 'Punctuation')]
3) 태깅 결과를 받아서 순회
4) 하나씩 순회 했을 때 튜플 형태로 가져오게 된다. ('을', 'Josa') 
5) 튜플에서 1번 인덱스에 있는 품사를 가져온다.
6) 해당 품사가 조사, 어미, 구두점이면 제외하고 append 로 인덱스 0번 값만 다시 리스트에 담아준다.
7) " ".join() 으로 공백문자로 연결해 주면 다시 문장이 완성된다.
8) 전처리 후 완성된 문장을 반환해준다.

```python
def okt_clean(text):
    clean_text = []
    # 품사태깅을 합니다. [('문의', 'Noun'), ('하다', 'Verb'), ('?!', 'Punctuation')]
    # 태깅 결과를 받아서 순회 합니다. 
    for word in okt.pos(text, norm=True, stem=True):
        # 해당 품사가 조사, 어미, 구두점이면 제외하고 append 로 인덱스 0번 값만 다시 리스트에 담아줍니다.
        if word[1] not in ['Josa', 'Eomi', 'Punctuation']:
            clean_text.append(word[0])
    # " ".join() 으로 공백문자로 연결해 주면 다시 문장이 됩니다.
    return " ".join(clean_text)

okt_clean("버스 운행시간을 했었네?")

out:
버스 운행 시간 하다
```

```python
train['title'] = train['title'].progress_map(preprocessing)
test['title'] = test['title'].progress_map(preprocessing)

train['title'] = train['title'].progress_map(okt_clean)
test['title'] = test['title'].progress_map(okt_clean)
```

## 불용어 제거

```python
# 불용어 제거
def remove_stopwords(text):
    tokens = text.split(' ')
    stops = [ '합니다', '하는', '할', '하고', '한다', 
             '그리고', '입니다', '그', '등', '이런', '및','제', '더']
    meaningful_words = [w for w in tokens if not w in stops]
    return ' '.join(meaningful_words)
```

```python
train["title"] = train["title"].map(remove_stopwords)
test["title"] = test["title"].map(remove_stopwords)
```

## 학습, 예측 데이터 만들기

```python
X_train_text = train["title"]
X_test_text = test["title"]

label_name = "topic_idx"

y_train = train[label_name]
```

## 벡터화

### TF-IDF

```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidfvect = TfidfVectorizer()
tfidfvect.fit(X_train_text)
```

```python
# transform : 열(columns, 어휘)의 수가 같은지 확인해볼 것
X_train = tfidfvect.transform(X_train_text)
X_test = tfidfvect.transform(X_test_text)

X_train.shape, X_test.shape

out: ((45654, 28605), (9131, 28605))
```

## 모델 설정과 학습 및 예측과 제출

```python
# 모델
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(random_state = 42)

# CV
from sklearn.model_selection import cross_val_predict

y_predict = cross_val_predict(model, X_train, y_train, cv = 3, n_jobs = -1, verbose = 1)

score = (y_train == y_predict).mean()
score

out:
0.6957112191702808

# 예측
y_test_predict = model.fit(X_train, y_train).predict(X_test)

# 예측 csv에 값 넣기
submit = pd.read_csv(f"{base_path}sample_submission.csv")

submit["topic_idx"] = y_test_predict

file_name = f"{base_path}submit_{score}.csv"

submit.to_csv(file_name, index = False)
```

# 시퀀스 인코딩

## Tokenizer

```python
import pandas as pd

corpus = ["서울 코로나 상생지원금 문의입니다.?",
"인천 지하철 운행시간 문의입니다.!",
"버스 운행시간 문의입니다.#"]
```

Tokenizer는 데이터에 출현하는 모든 단어의 개수를 세고 빈도 수로 정렬해서 num_words 에 지정된 만큼만 숫자로 반환하고, 나머지는 0 으로 반환한다. 단어 사전의 크기를 지정해 주기 위해 vocab_size를 지정해준다. vocab_size는 텍스트 데이터의 전체 단어 집합의 크기를 뜻한다.

```python
from tensorflow.keras.preprocessing.text import Tokenizer

vocab_size = 5  
tokenizer = Tokenizer(num_words = vocab_size)
```

```python
# Tokenizer 에 데이터 실제로 입력
# fit_on_texts와 word_index를 사용하여 key value로 이루어진 딕셔너리를 생성
tokenizer.fit_on_texts(corpus)
```

```python
# tokenizer의 word_index 속성은 단어와 숫자의 키-값 쌍을 포함하는 딕셔너리를 반환
# 이때, 반환 시 자동으로 소문자로 변환되어 들어가며, 느낌표나 마침표 같은 구두점은 자동으로 제거.

word_to_index = tokenizer.word_index
word_to_index

out:
{'문의입니다': 1,
 '운행시간': 2,
 '서울': 3,
 '코로나': 4,
 '상생지원금': 5,
 '인천': 6,
 '지하철': 7,
 '버스': 8}
```

```python
word_to_index.values()

out:
dict_values([1, 2, 3, 4, 5, 6, 7, 8])
```

```python
# df로 변환
wc = tokenizer.word_counts
pd.DataFrame(wc.items()).set_index(0).T
```

|  | 서울 | 코로나 | 상생지원금 | 문의입니다 | 인천 | 지하철 | 운행시간 | 버스 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 1 | 1 | 1 | 3 | 1 | 1 | 2 | 1 |

```python
# texts_to_sequences를 이용하여 text 문장을 숫자로 이루어진 리스트로 변경
# BOW는 등장유무를 보았다면, 시퀀스 방식은 해당 어휘사전을 만들고 어휘의 등장 순서대로 숫자로 변환

corpus_sequences = tokenizer.texts_to_sequences(corpus)
corpus_sequences

out:
[[3, 4, 1], [2, 1], [2, 1]]
```

```python
# ovv(out of vocab)
# ovv_token에 꼭 <oov>가 들어갈 필요는 없다

tokenizer = Tokenizer(num_words= 10, oov_token="<oov>")
tokenizer.fit_on_texts(corpus)
print(tokenizer.word_index)
print(corpus)
corpus_sequences = tokenizer.texts_to_sequences(corpus)
corpus_sequences

out:
{'<ovv>': 1, '문의입니다': 2, '운행시간': 3, '서울': 4, '코로나': 5, '상생지원금': 6, '인천': 7, '지하철': 8, '버스': 9}
['서울 코로나 상생지원금 문의입니다.?', '인천 지하철 운행시간 문의입니다.!', '버스 운행시간 문의입니다.#']
[[4, 5, 6, 2], [7, 8, 3, 2], [9, 3, 2]]
```

## Padding

자연어 처리를 하다보면 각 문장(또는 문서)은 서로 길이가 다를 수 있다. 기계는 길이가 전부 동일한 문서들에 대해서는 하나의 행렬로 보고, 한꺼번에 묶어서 처리할 수 있다. 병렬 연산을 위해서 여러 문장의 길이를 임의로 동일하게 맞춰주는 작업이 필요할 때가 있다.

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

import numpy as np

pads = pad_sequences(corpus_sequences, maxlen = 10)
print(corpus)
print(word_to_index)
print(pads)
np.array(pads)

out:
['서울 코로나 상생지원금 문의입니다.?', '인천 지하철 운행시간 문의입니다.!', '버스 운행시간 문의입니다.#']
{'문의입니다': 1, '운행시간': 2, '서울': 3, '코로나': 4, '상생지원금': 5, '인천': 6, '지하철': 7, '버스': 8}
[[0 0 0 0 0 0 4 5 6 2]
 [0 0 0 0 0 0 7 8 3 2]
 [0 0 0 0 0 0 0 9 3 2]]
array([[0, 0, 0, 0, 0, 0, 4, 5, 6, 2],
       [0, 0, 0, 0, 0, 0, 7, 8, 3, 2],
       [0, 0, 0, 0, 0, 0, 0, 9, 3, 2]])
```

# RNN(Recurrent Neural Network)으로 텍스트 분류하기

- [순환 신경망(Recurrent neural network, RNN)](https://ko.wikipedia.org/wiki/%EC%88%9C%ED%99%98_%EC%8B%A0%EA%B2%BD%EB%A7%9D)은 인공 신경망의 한 종류로, 유닛간의 연결이 순환적 구조를 갖는 특징을 갖고 있다. 이러한 구조는 시변적 동적 특징을 모델링 할 수 있도록 신경망 내부에 상태를 저장할 수 있게 해주므로, 순방향 신경망과 달리 내부의 메모리를 이용해 시퀀스 형태의 입력을 처리할 수 있다. 따라서 순환 인공 신경망은 필기 인식이나 음성 인식과 같이 시변적 특징을 지니는 데이터를 처리하는데 적용할 수 있다.
- 순환 신경망이라는 이름은 입력받는 신호의 길이가 한정되지 않은 동적 데이터를 처리한다는 점에서 붙여진 이름으로, 유한 임펄스 구조와 무한 임펄스 구조를 모두 일컫는다. 유한 임펄스 순환 신경망은 유향 비순환 그래프이므로 적절하게 풀어서 재구성한다면 순방향 신경망으로도 표현할 수 있지만, 무한 임펄스 순환 신경망은 유향 그래프인고로 순방향 신경망으로 표현하는 것이 불가능하다.
- 순환 신경망은 추가적인 저장공간을 가질 수 있다. 이 저장공간이 그래프의 형태를 가짐으로써 시간 지연의 기능을 하거나 피드백 루프를 가질 수도 있다. 이와 같은 저장공간을 게이트된 상태(gated state) 또는 게이트된 메모리(gated memory)라고 하며, LSTM과 게이트 순환 유닛(GRU)이 이를 응용하는 대표적인 예시이다.

## 라이브러리 로드

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## 데이터 로드 및 요약

```python
df = pd.read_csv("https://bit.ly/seoul-120-text-csv")
df.shape

out:
(2645, 5)

# 문서라는 파생변수 생성
df["문서"] = df["제목"] + " " + df["내용"]

df["분류"].value_counts()

out:
행정        1098
경제         823
복지         217
환경         124
주택도시계획     110
문화관광        96
교통          90
안전          51
건강          23
여성가족        13
Name: 분류, dtype: int64
```

분류별 빈도수 값의 불균형이 심해 데이터 예측성능이 떨어질 수 있다. 일부 데이터만 사용하도록 한다.

```python
df = df[df["분류"].isin(["행정","경제","복지"])]
df.shape

out:
(2138, 6)
```

정답 레이블 설정과 X,y 값을 만들어준다

```python
label_name = "분류"
X, y = df["문서"], df[label_name]

X.shape, y.shape

out:
((2138,), (2138,))
```

## Label One-Hot-Encoding

RNN 모델을 만들 예정이며 출력층은 기존에 만들었던 것처럼 만들 예정이다. 

**🤔"행정", "경제", "복지" label을 one-hot-encoding 을 해주는 이유?**
분류 모델의 출력층을 softmax로 사용하기 위해서이다. softmax는 각 클래스의 확률값을 반환하며 각각의 클래스의 합계를 구했을 때 1이 된다.

```python
y_onehot = pd.get_dummies(y)
y_onehot.head(2)
```

|  | 경제 | 복지 | 행정 |
| --- | --- | --- | --- |
| 0 | 0 | 1 | 0 |
| 1 | 1 | 0 | 0 |

```python
# train_test_split 으로 학습과 예측에 사용할 데이터 나누기

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, stratify=y_onehot, random_state=42)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

out:
((1710,), (428,), (1710, 3), (428, 3))
```

```python
display(y_train.value_counts(normalize=True))
display(y_test.value_counts(normalize=True))

out:
경제  복지  행정
0   0   1     0.513450
1   0   0     0.384795
0   1   0     0.101754
dtype: float64

경제  복지  행정
0   0   1     0.514019
1   0   0     0.385514
0   1   0     0.100467
dtype: float64
```

## Vectorization

### Tokenizer

```python
from tensorflow.keras.preprocessing.text import Tokenizer
```

`Tokenizer`는 데이터에 출현하는 모든 단어의 개수를 세고 빈도 수로 정렬해서 `num_words`에 지정된 만큼만 숫자로 반환하고,나머지는 0으로 반환한다. 단어 사전의 크기를 지정해 주기 위해 `vocab_size`(텍스트 데이터의 전체 단어 집합의 크기)를 지정해준다.

```python
vocab_size = 1000
oov_tok = "<oov>"
tokenizer = Tokenizer(oov_token = oov_tok)

# Tokenizer 에 데이터 실제로 입력
# fit_on_texts와 word_index를 사용하여 key value로 이루어진 딕셔너리를 생성
tokenizer.fit_on_texts(X_train)
```

```python
# tokenizer의 word_index 속성은 단어와 숫자의 키-값 쌍을 포함하는 딕셔너리를 반환. 
# 이때, 반환 시 자동으로 소문자로 변환되어 들어가며, 느낌표나 마침표 같은 구두점은 자동으로 제거.

pd.DataFrame(tokenizer.word_counts.items()).sort_values(1, ascending= False)
```

| 0 | 1 |
| --- | --- |
| 242 | 및 |
| 1338 | 돋움 |
| 73 | 수 |
| 458 | 경우 |
| 203 | 또는 |
| ... | ... |
| 16459 | 병과할 |
| 16458 | 형벌을 |
| 16457 | 과하는 |
| 16456 | 경우징계벌을 |
| 36256 | 설치하는가 |

36257 rows × 2 columns