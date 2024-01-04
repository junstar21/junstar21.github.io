---
title: "알고리즘 - 비트 조작 lll"
excerpt: "2024-01-04 Algorithm - Bit Control lll"

# layout: post
categories:
  - Python algorithm interview
tags:
  - python
  - algorithm
  - Bit
  - Leet code
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
해당 내용은 '[파이썬 알고리즘 인터뷰](https://product.kyobobook.co.kr/detail/S000001932748)' 책의 일부를 발췌하여 정리한 내용입니다.

### [UTF-8 검증](https://leetcode.com/problems/utf-8-validation/)

입력값이 UTF-8 문자열이 맞는지 검증하라.

```
Input: data = [197,130,1]
Output: true
Explanation: data represents the octet sequence: 11000101 10000010 00000001.
It is a valid utf-8 encoding for a 2-bytes character followed by a 1-byte character.
---
Input: data = [235,140,4]
Output: false
Explanation: data represented the octet sequence: 11101011 10001100 00000100.
The first 3 bits are all one's and the 4th bit is 0 means it is a 3-bytes character.
The next byte is a continuation byte which starts with 10 and that's correct.
But the second continuation byte does not start with 10, so it is invalid.
```

**🤔 UTF-8이란?**

세계 각국의 글자들을 표현하기 위해 통합된 코드 규격이라 생각하면 쉽다. 조금 더 자세한 정보는 이 [링크](https://jeongdowon.medium.com/unicode%EC%99%80-utf-8-%EA%B0%84%EB%8B%A8%ED%9E%88-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-b6aa3f7edf96)와 이 [링크](https://namu.wiki/w/UTF-8)를 참고하도록 하자.

- 풀이: 첫 바이트를 기준으로 한 판별
    
    ```python
    class Solution:
        def validUtf8(self, data: List[int]) -> bool:
            # 문자 바이트 만큼 10으로 판별
            def check(size):
                for i in range(start + 1, start + size + 1):
                    # 정해진 바이트 수 만큼 해당 바이트는 0b10으로 시작해야 한다.
                    if i >= len(data) or (data[i] >> 6) != 0b10:
                        return False
                return True
    
            start = 0
            while start < len(data):
                # first는 첫 바이트를 의미한다
                first = data[start]
                # 첫 바이트 변수가 0이면 1바이트 문자, 110이면 2바이트 문자
                # 1110 이면 3바이트 문자, 11110 이면 4 바이트 문자
                if (first >> 3) == 0b11110 and check(3):
                    start += 4
                elif (first >> 4) == 0b1110 and check(2):
                    start += 3
                elif (first >> 5) == 0b110 and check(1):
                    start += 2
                elif (first >> 7) == 0:
                    start += 1
                # 위 조건문에 부합하지 않으면 UTF-8 문자열이 아니므로 False를 반환한다
                else:
                    return False
            return True
    ```
    

### 1비트의 개수

부호없는 정수형을 입력받아 1비트의 개수를 출력하라.

- 풀이: 1의 개수 계산
    
    [해밍 거리 문제](https://junstar21.github.io/python%20algorithm%20interview/bit_control_ll/#%ED%95%B4%EB%B0%8D-%EA%B1%B0%EB%A6%AC)에서 사용되었던 풀이를 응용하여 풀이하면 쉽게 문제를 풀 수 있다.
    
    ```python
    class Solution:
        def hammingWeight(self, n: int) -> int:
            return bin(n ^ 0).count('1')
    ```
    
- 풀이: 비트 연산
    
    ```python
    class Solution:
        def hammingWeight(self, n: int) -> int:
            count = 0
            while n:
                # 1을 뺀 값과 AND 연산 횟수 측정
                n &= n -1
                count += 1
            return count
    ```
    
    해당 풀이는 이진수의 특성을 이용한 것이다. 예를 들어, 이진수 1000에서 1을 빼면 0111이 된다. 그렇다면, 1000와 0111을 AND 연산하면 어떤 결과가 나올까?
    
    ```python
    >>> bin(0b1000 & 0b0111)
    '0b0'
    ```
    
    0이 된다. 이러한 특성을 이용해서 1을 뺀 값과 AND 연산 할 때마다 비트가 1씩 빠지게 된다. 이 작업을 반복하여 비트가 0이 될 때까지 진행하면 몇개의 1이 있는지 알 수 있다.