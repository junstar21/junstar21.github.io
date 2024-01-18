---
title: "알고리즘 - 그리디 알고리즘 l"
excerpt: "2024-01-18 Greedy Algorithm l"

# layout: post
categories:
  - Python algorithm interview
tags:
  - python
  - algorithm
  - greedy
  - Leet code
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
해당 내용은 '[파이썬 알고리즘 인터뷰](https://product.kyobobook.co.kr/detail/S000001932748)' 책의 일부를 발췌하여 정리한 내용입니다.

## 그리디 알고리즘(Greedy Algorithm)

> 그리디 알고리즘은 글로벌 최적을 찾기 위해 각 단계에서 로컬 최적의 선택을 하는 휴리스틱 문제 해결 알고리즘이다.
> 

그리디 알고리즘이란 바로 눈 앞의 이익만을 좇는 알고리즘이다. 그리디 알고리즘은 최적화 문제를 대상으로 한다. 최적해를 찾기가 가능하면 그것을 목표로 삼고, 어려운 경우 주어진 시간 내 가장 괜찮은 해를 찾는 것을 목표로 삼는다. 합리적인 시간 내에 최적의 답을 찾을 수 있다는 점에서 매우 유용한 알고리즘이다.

그리디 알고리즘이 잘 작동하는 문제들은 탐욕 선택 속성(Greedy Choice Property, 앞의 선택이 이후 선택에 영향을 주지 않는 것)을 갖고 있는 최적 부분 구조(Optimal Substructure, 문제의 최적 해결 방법이 부분 문제에 대한 최적 해결 방법으로 구성된 경우)인 문제들이다.

앞서 살펴본 [다익스트라 알고리즘](https://junstar21.github.io/python%20algorithm%20interview/short_cut_problem/)은 대표적인 그리디 알고리즘의 예로서 최적의 해를 찾을 수 있다. 또한, 압축 알고리즘인 허프만 코딩 알고리즘, 의사결정 트리로 유명한 ID3 알고리즘 등이 그리디 알고리즘에 해당된다.

최적 부분 구조 문제를 푼다는 점에서 다이나믹 프로그래밍과 비교되지만 서로 풀 수 있는 문제의 성격이 다르며 아록리즘의 접근 방식 또한 다르다.

### [주식을 사고 팔기 가장 좋은 시점 ll](https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii/)

여러 번의 거래로 낼 수 있는 최대 이익을 산출하라.

```
Input: prices = [7,1,5,3,6,4]
Output: 7
Explanation: Buy on day 2 (price = 1) and sell on day 3 (price = 5), profit = 5-1 = 4.
Then buy on day 4 (price = 3) and sell on day 5 (price = 6), profit = 6-3 = 3.
Total profit is 4 + 3 = 7.
---
Input: prices = [1,2,3,4,5]
Output: 4
Explanation: Buy on day 1 (price = 1) and sell on day 5 (price = 5), profit = 5-1 = 4.
Total profit is 4.
---
Input: prices = [7,6,4,3,1]
Output: 0
Explanation: There is no way to make a positive profit, so we never buy the stock to achieve the maximum profit of 0.
```

- 풀이: 그리디 알고리즘

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        result = 0

        for i in range(len(prices) - 1):
            if prices[i + 1] > prices[i]:
                result += prices[i + 1] - prices[i]

        return result
```

### [키에 따른 대기열 재구성](https://leetcode.com/problems/queue-reconstruction-by-height/description/)

여러 명의 사람들이 줄을 서있다. 각각의 사람은 (h, k)의 두 정수 쌍을 가지는데, h는 그 사람의 키, k는 앞에 줄 서 있는 사람들 중 자신의 키 이상인 사람들의 수를 뜻한다. 이 값이 올바르도록 줄을 재정렬하는 알고리즘을 작성하라.

```
Input: people = [[7,0],[4,4],[7,1],[5,0],[6,1],[5,2]]
Output: [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]]
Explanation:
Person 0 has height 5 with no other people taller or the same height in front.
Person 1 has height 7 with no other people taller or the same height in front.
Person 2 has height 5 with two persons taller or the same height in front, which is person 0 and 1.
Person 3 has height 6 with one person taller or the same height in front, which is person 1.
Person 4 has height 4 with four people taller or the same height in front, which are people 0, 1, 2, and 3.
Person 5 has height 7 with one person taller or the same height in front, which is person 1.
Hence [[5,0],[7,0],[5,2],[6,1],[4,4],[7,1]] is the reconstructed queue.
---
Input: people = [[6,0],[5,0],[4,0],[3,2],[2,2],[1,4]]
Output: [[4,0],[5,0],[2,2],[3,2],[1,4],[6,0]]
```

- 우선순위 큐를 이용

[heap](https://junstar21.github.io/python%20algorithm%20interview/heap/) 기능을 적극 이용하여 풀이를 진행한다.

```python
class Solution:
    def reconstructQueue(self, people: List[List[int]]) -> List[List[int]]:
        heap = []
        # 키 역순, 인덱스 삽입
        for person in people:
            heapq.heappush(heap, (-person[0], person[1]))

        result = []

        # 키 역순, 인덱스 추출
        while heap:
            person = heapq.heappop(heap)
            result.insert(person[1], [-person[0], person[1]])
        return result
```