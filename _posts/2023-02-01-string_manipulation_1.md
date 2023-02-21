---
title: "문자열 조작 - 1"
excerpt: "2023-02-01 String manipulation - 1"

# layout: post
categories:
  - Python algorithm interview
tags:
  - python
  - string
  - Leet code
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
해당 내용은 '[파이썬 알고리즘 인터뷰](https://product.kyobobook.co.kr/detail/S000001932748)' 책의 일부를 발췌하여 정리한 내용입니다.

문자열 조작(String Manipulation)이란 문자열을 변경하거나 분리한느 등의 여러 과정을 말한다. **문자열 조작은 코딩 테스트에서 매우 빈번하게 출제되는 주제** 중 하나이며**, 실무에서도 다양한 분야에 쓰이는 상당히 실용적인 주제**이다. 문자열 처리와 관련한 알고리즘을 쓰이는 대표적인 분야는 다음과 같다.

- 정보 처리 분야 : 특정 키워드로 웹페이즈를 탐색할 때 문자열 처리 애플리케이션을 이용하게 된다.
- 통신 시스템 분야 : 문자나 이메일 등을 전송 시 문자열로 어느 한 곳에서 다른 곳으로 보낸다. 데이터 전송은 문자열 처리 알고리즘 탄생의 기원이며 해당 분야에서 문자열 처리는 매우 중요한 역할을 한다.
- 프로그래밍 시스템 분야 : 프로그램은 그 자체가 문자열로 구성되어 있다. 문자열을 해석하고 처리하여 기계어로 변환하는 역할을 하며, 매우 정교한 문자열 처리 알고리즘 등이 쓰인다.

## 유효한 팰린드롬

[Valid Palindrome - LeetCode](https://leetcode.com/problems/valid-palindrome/)

**🤔 ‘팰린드롬(Palindrome)’이란?**

> 앞뒤가 똑같은 단어나 문장으로, 뒤집어도 같은 말이 되는 단어 또는 문장을 팰린드롬이라고 한다. 문장 중에서 대표적으로 ‘소주 만 병만 주소’로 앞에서 읽거나 뒤에서 읽어도 같은 문장에 해당된다. 팰린드롬의 특징을 응용하여 코딩 테스트에 매우 자제 출제되는 주제이기도 하다.
> 

### 풀이

- 나의 풀이

```python
# 문자열 변환을 위해 re import
import re

class Solution:
    def isPalindrome(self, s: str) -> bool:
				
        s = s.lower() # 모두 소문자로 변환 시켜주기
        s = re.sub('[^a-z0-9]', '', s) # 알파벳과 숫자 외 다른 문자들 제거

        b = s[::-1] # slice 기법으로 역순으로 string 배열

		return s == b # == 연산자를 통해 True 또는 False를 반환
```

- 풀이 1 : 리스트로 변환

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        strs = []
        for char in s:
			if char.isalnum(): # 영문자, 숫자 여부 판별 함수
	            strs.append(char.lower()) # 알파벳 하나씩 리스트에 추가

        # 팰린드롬 여부 판별
        while len(strs) > 1:
            if strs.pop(0) != strs.pop(): # 0번째 인덱스와 마지막 인덱스 확인
                return False

        return True
```

- 풀이 2 : 데크 자료형을 이용한 최적화

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
        strs: Deque = collections.deque()

        for char in s:
            if char.isalnum():
                strs.append(char.lower())

        while len(strs) > 1:
            if strs.popleft() != strs.pop():
                return False

        return True
```

`strs: Deque = collections.deque()` 선언만으로도 실행 시간을 단축시킬 수 있다. 이는 리스트의 `pop(0)`이 O(n)인데 비해, 데크의 `popleft()`는 O(1)이기 때문에 n번 반복 시 `pop(0)`는 O(n^2), `popleft()`는 O(n)으로 성능 차이가 크게 나온다.

- 풀이 3 : 슬라이싱 사용

```python
class Solution:
    def isPalindrome(self, s: str) -> bool:
				
        s = s.lower()
        s = re.sub('[^a-z0-9]', '', s)

		return s == s[::-1]
```

**🔠 문자열 슬라이싱**

> 파이썬의 문자열 슬라이싱은 매우 편리하며, 내부적으로 매우 빠르게 동작한다. 위치를 지정하면 해당 위치의 배열 포인터를 얻게 되며 이를 통해 연결된 객체를 찾아 실제 값을 찾아내는데 이 과정이 매우 빨라서 문자열 조작 시 항상 슬라이싱을 우선 사용하는게 속도 개선에 유리하다.
> 

## 문자열 뒤집기

[Reverse String - LeetCode](https://leetcode.com/problems/reverse-string/)

- 풀이 1 : 투 포인터를 이용한 스왑

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        left, right = 0, len(s) -1
        while left < right:
            s[left], s[right] = s[right], s[left]
            left += 1
            right -= 1
```

- 풀이 2 : `reverse()` 함수 사용

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        return s.reverse()
```

- 번외 : 슬라이싱

```python
class Solution:
    def reverseString(self, s: List[str]) -> None:
        s[:] = s[::-1]
```

원래는 `s = s[::-1]`도 처리가 되지만, 해당 문제는 공간 복잡도를 O(1)로 제한하여 리트 코드에서는 오답 처리가 발생한다. 

## 로그파일 재정렬

[Reorder Data in Log Files - LeetCode](https://leetcode.com/problems/reorder-data-in-log-files/)

- 풀이 : 람다와 + 연산자 활용

```python
class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:
        letters, digits = [], []
        for log in logs:
            if log.split()[1].isdigit(): # isdigit() : 숫자 여부를 판별해주는 함수
                digits.append(log)
            else:
                letters.append(log)

        # 2개의 키를 람다 표현식으로 정렬
        # x.split()[1:] : contents 순서로 정렬
        # x.split()[0] : identifiers 순서로 정렬
        letters.sort(key = lambda x : (x.split()[1:], x.split()[0]))

        return letters + digits
```

- 리트코드 공식 풀이

```python
class Solution:
    def reorderLogFiles(self, logs: List[str]) -> List[str]:

        def get_key(log):
            _id, rest = log.split(" ", maxsplit=1)
            return (0, rest, _id) if rest[0].isalpha() else (1, )

        return sorted(logs, key=get_key)
```