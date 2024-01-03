---
title: "알고리즘 - 비트 조작 ll"
excerpt: "2024-01-03 Algorithm - Bit Control ll"

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

### [해밍 거리](https://leetcode.com/problems/hamming-distance/)

두 정수를 입력받아 몇 비트가 다른지 계산하라.

```
Input: x = 1, y = 4
Output: 2
Explanation:
1   (0 0 0 1)
4   (0 1 0 0)
       ↑   ↑
The above arrows point to positions where the corresponding bits are different.
---
Input: x = 3, y = 1
Output: 1

```

- 풀이: XOR 풀이

```python
class Solution:
    def hammingDistance(self, x: int, y: int) -> int:
        return bin(x ^ y).count('1')
```

자연어 처리에서도 널리 쓰이는 해밍 거리는 두 정수 또는 두 문자열의 차이를 뜻한다. 문자열의 경우 해밍 거리는 다른 자리의 문자 개수이며, 이진수의 경우 다른 위치의 비트 개수가 된다.

### [두 정수의 합](https://leetcode.com/problems/sum-of-two-integers/)

두 정수 a와 b의 합을 구하라. + 또는 - 연산자는 사용할 수 없다.

```
Input: a = 1, b = 2
Output: 3
---
Input: a = 2, b = 3
Output: 5
```

- 풀이: [전가산기](https://ko.wikipedia.org/wiki/%EA%B0%80%EC%82%B0%EA%B8%B0) 구현

```python
class Solution:
    def getSum(self, a: int, b: int) -> int:
        # 비트 마스킹
        MASK = 0xFFFFFFFF
        
        # 이진수에서 양의 최대값 정의
        # 이진수에서 음의 최소값은 정해진 비트 수의 양의 최대값 다음이다.
        INT_MAX = 0x7FFFFFFF

        # 이진수 변환 후 전처리
        # 0b 제거 & 입력값을 32비트로 가정하였기에
        # 비어있는 앞자리를 0으로 채워서 32비트 자리수로 만듦
        a_bin = bin(a & MASK)[2:].zfill(32)
        b_bin = bin(b & MASK)[2:].zfill(32)

        result = []
        carry = 0
        sum = 0
        # 뒷부분부터 전가산기 통과
        for i in range(32):
            A = int(a_bin[31 - i])
            B = int(b_bin[31 - i])

            # 전가산기 구현
            Q1 = A & B
            Q2 = A ^ B
            Q3 = Q2 & carry
            sum = carry ^ Q2
            carry = Q1 | Q3

            result.append(str(sum))
        # carry가 남아있는 경우, 자릿수가 하나 더 올라간 것이므로 1을 추가
        if carry == 1:
            result.append('1')

        # 마지막 마스킹 작업을 통해 초과 자릿수 처리
        result = int(''.join(result[::-1]), 2) & MASK
        # 음수 처리
        if result > INT_MAX:
            # 마스킹값과 XOR 연산 후 NOT 처리를 통해 음수로 전환
            result = ~(result ^ MASK)

        return result
```

- 풀이: 좀 더 간소한 구현

전가산기의 핵심만 살려서 간단하게 동작하도록 하였다.

```python
class Solution:
    def getSum(self, a: int, b: int) -> int:
        # 비트 마스킹
        MASK = 0xFFFFFFFF
        
        # 이진수에서 양의 최대값 정의
        # 이진수에서 음의 최소값은 정해진 비트 수의 양의 최대값 다음이다.
        INT_MAX = 0x7FFFFFFF

        # 합, 자리수 처리
        # a에는 carry 값을 고려하지 않은 a와 b의 합을 담음
        # b에는 자릿수를 올려가며 carry를 담음
        while b != 0:
            a, b = (a ^ b) & MASK, ((a & b) << 1) & MASK

        # 음수 처리
        if a > INT_MAX:
            a = ~(a ^ MASK)

        return a
```