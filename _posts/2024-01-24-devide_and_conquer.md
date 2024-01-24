---
title: "알고리즘 - 분할 정복"
excerpt: "2024-01-24 Divide and Conquer"

# layout: post
categories:
  - Python algorithm interview
tags:
  - python
  - algorithm
  - Leet code
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
해당 내용은 '[파이썬 알고리즘 인터뷰](https://product.kyobobook.co.kr/detail/S000001932748)' 책의 일부를 발췌하여 정리한 내용입니다.

# 분할 정복(Divide and Conquer)

> 분할 정복은 다중 분기 재귀를 기반으로 하는 알고리즘 디자인 패러다임을 말한다.
> 

분할 정복은 문제를 직접 해결 할 수 있을 정도로 간단한 문제가 될 때까지 문제를 재귀적으로 쪼갠 후, 하위 결과들을 조합해 원래 문제의 결과로 만들어내는 알고리즘이다. 대표적인 분할 정복 알고리즘으로는 [병합 정렬](https://junstar21.github.io/python%20algorithm%20interview/order/)을 예로 들 수 있다.

- 분할: 문제를 동일한 유형의 여러 하위 문제로 나눔
- 정복: 가장 작은 단위의 하위 문제를 해결
- 조합: 하위 문제에 대한 결과를 원래 문제에 대한 결과로 조합

말 그대로 문제를 ‘분할’한 뒤, 각각 ‘정복’하여 그 결과를 ‘조합’하는 것이 분할 정복이다.

### [과반수 엘리먼트](https://leetcode.com/problems/majority-element/)

과반수를 차지하는(절반을 초과하는) 엘리먼트를 출력하라.

```
Input: nums = [3,2,3]
Output: 3
---
Input: nums = [2,2,1,1,1,2,2]
Output: 2
```

- 나의 풀이: collections 모듈 사용

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        return collections.Counter(nums).most_common()[0][0]
```

과반수를 차지한다는 것은 결국 해당 리스트에서 가장 많은 수를 가진 인자를 뽑아내라는 것이다. 우선, 분할 정복이 아닌 가장 파이썬스러운 풀이를 하기 위해 한줄 코드로 답안을 작성하였다.

- 풀이: 분할 정복

```python
class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        if not nums:
            return None
        if len(nums) == 1:
            return nums[0]

        half = len(nums) // 2
        a = self.majorityElement(nums[:half])
        b = self.majorityElement(nums[half:])

        return [b,a][nums.count(a) > half]
```

### [괄호를 삽입하는 여러 가지 방법](https://leetcode.com/problems/different-ways-to-add-parentheses/)

숫자와 연산자를 입력받아 가능한 모든 조합의 결과를 출력하라.

```
Input: expression = "2-1-1"
Output: [0,2]
Explanation:
((2-1)-1) = 0
(2-(1-1)) = 2
---
Input: expression = "2*3-4*5"
Output: [-34,-14,-10,-10,10]
Explanation:
(2*(3-(4*5))) = -34
((2*3)-(4*5)) = -14
((2*(3-4))*5) = -10
(2*((3-4)*5)) = -10
(((2*3)-4)*5) = 10
```

- 풀이: 분할 정복을 이용한 다양한 조합

```python
class Solution:
    def diffWaysToCompute(self, input: str) -> List[int]:
        def compute(left, right, op):
            results = []
            for l in left:
                for r in right:
                    results.append(eval(str(l) + op + str(r)))
            return results

        # .isdigit(): 문자열이 '숫자'로만 이루어져있는지 확인하는 함수.
        if input.isdigit():
            return [int(input)]

        results = []
        for index, value in enumerate(input):
            if value in "-+*":
                left = self.diffWaysToCompute(input[:index])
                right = self.diffWaysToCompute(input[index + 1:])

                results.extend(compute(left, right, value))
        return results
```