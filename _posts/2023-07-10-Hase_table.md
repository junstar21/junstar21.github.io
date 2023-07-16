---
title: "선형 자료구조 - 해시 테이블"
excerpt: "2023-07-10 Linear Data Structures - Hash Table"

# layout: post
categories:
  - Python algorithm interview
tags:
  - python
  - Hash table
  - Leet code
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
> 해시 테이블 또는 해시 맵은 키를 값에 매핑할 수 있는 구조인, 연관 배열 추상 자료형(ADT)을 구현하는 자료구조다.
> 

해시 테이블의 가장 큰 특징은 대부분의 연산이 분할 상환 분석에 따른 시간복잡도가 O(1)이라는 점이다.

## 해시

> **해시 함수**란 임의 크기 데이터를 고정 크기 값으로 매핑하는 데 사용할 수 있는 함수를 말한다.
> 

**해시 함수**는 해시 테이블의 핵심이다. 

```python
ABC -> A1
1324BC -> CB
AF32B -> D5
```

특정 입력 값들이 특정 함수를 통과하면 2바이트의 고정 크기 값으로 매핑된다. 이 때, 화살표 역할을 하는 함수가 바로 **해시 함수**이다.

해시 테이블을 인덱싱하기 위해 이처럼 해시 함수를 사용하는 것을 **해싱(Hashing**)이라 한다. 해싱은 정보를 가능한 빠르게 저장 및 검색하기 위해 사용되는 중요 기법 중 하나이다. 따라서 최적의 검색이 필요한 분야에 사용되며, 심볼 테이블 등의 자료 구조 구현하기에도 적합하다.

성능 좋은 해시 함수들의 특징은 다음과 같다.

- 해시 함수 값 충돌의 최소화
- 쉽고 빠른 연산
- 해시 테이블 전체에 해시 값이 균일하게 분포
- 사용할 키의 모든 정보를 이용하여 해싱
- 해시 테이블 사용 효율이 높을 것

좋은 해시 함수들은 충돌이 적다고 적어놓았다. 그렇다면, 실제로 충돌이 얼마나 빈번하게 일어나는지 알아보도록 하자.

### 생일 문제

생각보다 충돌은 쉽게 일어난다. 가장 흔한 예시 중 하나가 바로 생일 문제이다. 생일의 가짓수는 365개(윤년 제외)이므로, 여러 사람이 모였을 때 생일이 같은 2명이 존재활 확률이 얼마일까? 실제로 23명만 모여도 50%가 넘고 57명이 모이면 그 때부터는 99%를 넘어간다.

```python
import random

TRIALS = 100000 # 10만번 실험
same_birthdays = 0 # 같은 생일 카운트

# 10만번 실험 진행
for _ in range(TRIALS):
    birthdays = []
    # 23명이 모였을 때, 생일이 같을 경우 Same_birthday += 1
    for i in range(23):
        birthday = random.randint(1, 365)
        if birthday in birthdays:
            same_birthdays += 1
            break
        birthdays.append(birthday)
        
# 전체 10만 번 실험 중 생일이 같은 실험의 확률
print(f'{same_birthdays / TRIALS * 100}%')

out:
50.870000000000005%
---
# 인원수가 57명일 경우
out:
98.98%
```

위 코드는 실제 내용을 파이썬으로 작성하여 돌려본 결과이다. 의외로 적은 사람 수로도 높은 확률이 나타나는 것을 확인 할 수 있다. 이처럼 충돌은 생각보다 쉽게 일어나므로 충돌을 최소하나는 일은 무엇보다 중요하다.

### 비둘기집 원리

왜 충돌이 일어날 수 밖에 없을까? 비둘기집 원리가 이를 잘 설명한다.

> 비둘기집 원리란, n개 아이템을 m개 컨테이너에 넣을 때, n>m이라면 적어도 하나의 컨테이너에는 반드시 2개 이상의 아이템이 들어 있다는 원리를 말한다.
> 

비둘기집 원리에 따라 9개의 공간이 있는 곳에 10개의 아이템이 들어온다면 반드시 1번 이상은 충돌이 발생하게 된다. 좋은 해시 함수라면 충돌이 1번밖에 일어나지 않겠지만, 심하게 좋지 않은 경우 9번 충돌하여 1개의 공간밖에 사용할 수 밖에 없다.

### 로드 팩터

> 로드 팩터(Load Factor)란 해시 테이블에 저장된 데이터 개수 n을 버킷의 개수 k로 나눈 것이다.
load factor = n/k
> 

자바 10에서는 해시맵의 디폴트 로드 팩터를 0.75로 정했으며, ‘시간과 공간 비용의 적절한 절충안’이라고 이야기 한다. 일반적으로 로드 팩터가 증가할수록 해시 테이블의 성능이 점차 감소하게 된다.

### 해시 함수

앞서 헤시 테이블을 인덱싱하기 위해 해시 함수를 사용하는 것을 해싱이라고 서술하였다. 해싱 알고리즘도 여러 종류가 있지만, 가장 단순하면서도 널리 쓰이는 정수형 해싱 기법인 모듈로 연산을 이용한 나눗셈 방식(Modulo-Division Method)을 주로 사용한다.

### 개별 체이닝

아무리 좋은 해시 함수라 할지라도 충돌은 발생한다. 이럴 경우 충돌을 어떤 식으로 처리하는지 살펴보자.

| 키 | 값 | 해시 | 충돌여부 |
| --- | --- | --- | --- |
| 민지 | 15 | 2 | 충돌 |
| 하니 | 47 | 1 |  |
| 해린 | 17 | 2 | 충돌 |
| 다니엘 | 7 | 4 |  |
| 혜인 | 12 | 3 |  |

위 표는 키 값을 해싱한 해시값과 충돌여부를 나타낸다. 해싱한 결과, ‘민지’와 ‘해린’을 해싱한 결과는 충돌한다고 가정한다. 개별 체이닝(Separate Chaining)은 해시 테이블의 기본 방식이기도 하며, 충돌 발생 시 연결 리스트로 연결(link)하는 방식이다. 충돌이 발생한 ‘민지’와 ‘해린’은 ‘민지’의 다음 아이템인 ‘해린’인 형태로 서로 연결 리스트로 연결된다. 이처럼 기본적인 자료 구조와 임의로 정한 간단한 알고리즘만 있으면 되므로, 개별 체이닝 방식은 인기가 높다. 간단히 원리를 요약하면 다음과 같다.

1. 키의 해시값을 계산
2. 해시 값을 이용해 배열의 인덱스를 구하기
3. 같은 인덱스가 있다면 연결리스트로 연결

해시 테이블 구조의 원형이기도 하며 가장 전통적인 방식으로, 흔히 해시 테이블이라 하면 이 방식을 의미한다. 잘 구현한 경우 대부분의 탐색은 O(1)이지만, 최악의 경우 모든 해시 충돌이 발생할 때는 O(n)이 된다.

### 오픈 어드레싱

오픈 어드레싱(Open Addressing) 방식은 충돌 발생 시 탐사를 통해 빈 공간을 찾아나서는 방식이다. 사실상 무한대로 저장하는 체이닝 방식과 달리, 오픈 어드레싱은 전체 슬롯의 개수 이상을 저장할 수 없다. 충돌 발생 시 테이블 공간 내에서 탐사를 통해 빈 공간을 찾아 해결하며, 이때문에 모든 원소가 반드시 자신의 해시값과 일치하는 주소에 저장된다는 보장은 없다.

여러 가지 오픈 어드레싱 방식 중에서 가장 간단한 방식은 선형 탐사(Linear Probing)이다. 개별 체이닝의 예시에서 ‘민지’와 ‘해린’의 충돌이 발생하였다. 충돌이 발생할 경우 해당 위치부터 순차적으로 탐사를 하나씩 진행하여 빈 위치가 나올 때까지 탐사를 진행하고 빈 위치 발견 시 해당 위치에 ‘해린’이 들어가게 된다. 선형 탐색은 구현 방법이 간단하면서도 의외로 성능이 좋은 편이다.

선형 탐사의 한 가지 문제점은 해시 테이블에 저장되는 데이터들이 고르게 분포되지 못하고 뭉치ㅣ는 클러스터링(Clustering) 현상이 나타난다. 이렇게 클러스터링 현상이 일어나게 되면 다른 위치에는 상대적으로 데이터가 없는 상태가 발생하게 되고 이는 탐사 시간이 늘어나게 되어 전체적으로 싱 효율을 떨어뜨리게 된다.

### 언어별 해시 테이블 구현 방식

리스트와 함께 파이썬에서 흔히 쓰이는 자료형인 딕셔너리는 해시 테이블로 구현되어있다. 파이썬의 해시 테이블은 충돌 시 탐색을 오픈 어드레싱 방식으로 하도록 구현되어있다.

### [해시맵 디자인](https://leetcode.com/problems/design-hashmap/)

다음의 기능을 제공하는 해시맵을 디자인하라.

- put(key, value): 키, 값을 해시맵에 삽입한다. 만약 이미 존재하는 키라면 업데이트한다.
- get(key): 키에 해당하는 값을 조회한다. 만약 키가 존재하지 않다면 -1을 리턴한다.
- remove(key): 키에 해당하는 키, 값을 해시맵에서 삭제한다.

```
Input
["MyHashMap", "put", "put", "get", "get", "put", "get", "remove", "get"]
[[], [1, 1], [2, 2], [1], [3], [2, 1], [2], [2], [2]]
Output
[null, null, null, 1, -1, null, 1, null, -1]

Explanation
MyHashMap myHashMap = new MyHashMap();
myHashMap.put(1, 1); // The map is now [[1,1]]
myHashMap.put(2, 2); // The map is now [[1,1], [2,2]]
myHashMap.get(1);    // return 1, The map is now [[1,1], [2,2]]
myHashMap.get(3);    // return -1 (i.e., not found), The map is now [[1,1], [2,2]]
myHashMap.put(2, 1); // The map is now [[1,1], [2,1]] (i.e., update the existing value)
myHashMap.get(2);    // return 1, The map is now [[1,1], [2,1]]
myHashMap.remove(2); // remove the mapping for 2, The map is now [[1,1]]
myHashMap.get(2);    // return -1 (i.e., not found), The map is now [[1,1]]
```

- 풀이

```python
import collections

# Definition for singly-linked list.
class ListNode:
    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.next = None

class MyHashMap:
    # 초기화
    def __init__(self):
        self.size = 1000
        self.table = collections.defaultdict(ListNode)

    # 삽입
    def put(self, key: int, value: int) -> None:
        index = key % self.size
        # 인덱스에 노드가 없다면 삽입 후 종료
        if self.table[index].value is None:
            self.table[index] = ListNode(key, value)
            return

        # 인덱스에 노드가 존재하는 경우 연결 리스트 처리
        p = self.table[index]
        while p:
            if p.key == key:
                p.value = value
                return
            if p.next is None:
                break
            p = p.next
        p.next = ListNode(key, value)

    # 조회
    def get(self, key: int) -> int:
        index = key % self.size
        if self.table[index].value is None:
            return -1

        # 노드가 존재할때 일치하는 키 탐색
        p = self.table[index]
        while p:
            if p.key == key:
                return p.value
            p = p.next
        return -1

    # 삭제
    def remove(self, key: int) -> None:
        index = key % self.size
        if self.table[index].value is None:
            return

        # 인덱스의 첫 번째 노드일때 삭제 처리
        p = self.table[index]
        if p.key == key:
            self.table[index] = ListNode() if p.next is None else p.next
            return

        # 연결 리스트 노드 삭제
        prev = p
        while p:
            if p.key == key:
                prev.next = p.next
                return
            prev, p = p, p.next
```

### [보석과 돌](https://leetcode.com/problems/jewels-and-stones/description/)

J는 보석이며, S는 갖고 있는 돌이다. S에는 보석이 몇개나  있을까? 대소문자는 구분한다.

```
Input: jewels = "aA", stones = "aAAbbbb"
Output: 3
---
Input: jewels = "z", stones = "ZZ"
Output: 0
```

- 나의 풀이: for문과 if문을 이용하여 조회 후 일치하면 카운팅을 올려서 결과 반환

```python
class Solution:
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        jewels_count = 0

        for stone in stones:
            if stone in jewels:
                jewels_count += 1

        return jewels_count
```

- 해시 테이블을 이용한 풀이

```python
class Solution:
    def numJewelsInStones(self, J: str, S: str) -> int:
        freqs = {}
        count = 0

        # 돌(S)의 빈도 수 계산
        for char in S:
            if char not in freqs:
                freqs[char] = 1
            else:
                freqs[char] += 1

        # 보석(J)의 빈도 수 합산
        for char in J:
            if char in freqs:
                count += freqs[char]

        return count
```

- defaultdict를 이용한 비교 생략

```python
import collections

class Solution:
    def numJewelsInStones(self, J: str, S: str) -> int:
        freqs = collections.defaultdict(int)
        count = 0

        # 비교 없이 돌(S) 빈도 수 계산
        for char in S:
            freqs[char] += 1

        # 비교 없이 보석(J) 빈도 수 합산
        for char in J:
            count += freqs[char]

        return count
```

- Counter로 계산 생략

```python
import collections

class Solution:
    def numJewelsInStones(self, J: str, S: str) -> int:
        freqs = collections.Counter(S)  # 돌(S) 빈도 수 계산
        count = 0

        # 비교 없이 보석(J) 빈도 수 합산
        for char in J:
            count += freqs[char]

        return count
```

- 파이썬 다운 방식

```python
class Solution:
    def numJewelsInStones(self, J: str, S: str) -> int:
        return sum(s in J for s in S)
```

### [중복 문자 없는 가장 긴 부분 문자열](https://leetcode.com/problems/longest-substring-without-repeating-characters/description/)

중복 문자가 없는 가장 긴 문자열의 길이를 리턴하라

```
Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
```

```python
pwwkew
wekwwp

left : pwke
right : wekp
```

- 슬라이딩 윈도우와 투포인터로 사이즈 조절

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        used = {}
        max_length = start = 0
        for index, char in enumerate(s):
            # 이미 등장했던 문자라면 `start` 위치 갱신
            if char in used and start <= used[char]:
                start = used[char] + 1
            else:  # 최대 부분 문자열 길이 갱신
                max_length = max(max_length, index - start + 1)

            # 현재 문자의 위치 삽입
            used[char] = index

        return max_length
```

### [상위 k 빈도 요소](https://leetcode.com/problems/top-k-frequent-elements/)

상위 k번 이상 등장하는 요소를 추출하라.

```
Input: nums = [1,1,1,2,2,3], k = 2
Output: [1,2]
```

- 파이썬 다운 방식 풀이

```python
import collections

class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        return list(zip(*collections.Counter(nums).most_common(k)))[0]
```

**🤔 아스테리스크(*)**

파이썬에서 *은 언팩(unpack)이다. 말 그대로 시퀀스를 풀어헤치는 연산자를 뜻하며, 주로 튜플이나 리스트를 언패킹하는데 사용한다. 위 문제의 예시를 이용해서 확인해보자.

```python
collections.Counter(nums).most_common(k)

out:
[(1,3), (2,2)]

# 언팩 미사용 시
return list(zip(collections.Counter(nums).most_common(k)))

out:
[((1,3),), ((2, 2),)]

# 언팩 사용 시
return list(zip(*collections.Counter(nums).most_common(k)))

out:
[(1,2), (3,2)]
```

`collections.Counter(nums).most_common(k)`는 값을 튜플 형태로 돌려준다. 우리가 원하는 정답형태를 만들려면 언패킹이 필요하다.