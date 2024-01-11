---
title: "알고리즘 - 슬라이딩 윈도우 l"
excerpt: "2024-01-11 Sliding Window l"

# layout: post
categories:
  - Python algorithm interview
tags:
  - python
  - algorithm
  - sliding window
  - Leet code
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
해당 내용은 '[파이썬 알고리즘 인터뷰](https://product.kyobobook.co.kr/detail/S000001932748)' 책의 일부를 발췌하여 정리한 내용입니다.

## 슬라이딩 윈도우

> 슬라이딩 윈도우(Sliding Window)란 고정 사이즈의 윈도우가 이동하면서 윈도우 내에 있는 데이터를 이용해 문제를 풀이하는 알고리즘을 말한다.
> 

[투 포인터](https://junstar21.github.io/python%20algorithm%20interview/array/#%EC%84%B8-%EC%88%98%EC%9D%98-%ED%95%A9)와 함께 슬라이딩 윈도우는 알고리즘 문제 풀이에 매우 유용하게 사용되는 중요한 기법이다. 언뜻 보면 두 기법은 비슷해 보이나, 투 포인터는 정렬된 배열을 대상으로 주로 사용되고 슬라이딩 윈도우는 정렬 여부와 상관없이 활용된다는 차이가 있다.

```
# 괄호 안에 있는 숫자가 윈도우 내에 있는 데이터를 의미한다
# 투 포인터
[1, 2, 3, 4, 5] -> 1, [2, 3, 4, 5] -> 1, 2, [3, 4], 5

# 슬라이딩 윈도우
[1, 3, 4], 2, 5 -> 1, [3, 4, 2], 5 -> 1, 3, [4, 2, 5]
```

투 포인터는 윈도우 사이즈가 가변적이며, 좌우 포인터가 자유롭게 이동할 수 있으며, 슬라이딩 윈도우는 윈도우 사이즈가 고정이며 좌 또는 우 한방향으로만 이동한다.

| 이름 | 정렬 여부 | 윈도우 사이즈 | 이동 |
| --- | --- | --- | --- |
| 투 포인터 | 대부분 O | 가변 | 좌우 포인터 양방향 |
| 슬라이딩 윈도우 | X | 고정 | 좌 또는 우 단방향 |

### [최대 슬라이딩 윈도우](https://leetcode.com/problems/sliding-window-maximum/)

배열 nums가 주어졌을 때 k 크기의 슬라이딩 윈도우를 오른쪽 끝까지 이동하면서 최대 슬라이딩 윈도우를 구하라.

```
Input: nums = [1,3,-1,-3,5,3,6,7], k = 3
Output: [3,3,5,5,6,7]
Explanation:
Window position                Max
---------------               -----
[1  3  -1] -3  5  3  6  7       3
 1 [3  -1  -3] 5  3  6  7       3
 1  3 [-1  -3  5] 3  6  7       5
 1  3  -1 [-3  5  3] 6  7       5
 1  3  -1  -3 [5  3  6] 7       6
 1  3  -1  -3  5 [3  6  7]      7
---
Input: nums = [1], k = 1
Output: [1]

```

- 나의 풀이: 슬라이딩 윈도우

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        s = 0
        result = []

        while k <= len(nums):
            window = nums[s : k]
            result.append(max(window))
            s += 1
            k += 1

        return result
```

코드만 보면 매우 간단하게 풀이가 되는 것 처럼 보인다. 하지만, nums의 자료 개수가 10만이 되고 k가 50000인 case에서는 타임아웃 에러로 풀이를 진행할 수 없다. 조금 더 효율적인 풀이가 필요하다.

- 풀이: 브루트 포스

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        # 예외조항 포함

        if not nums:
            return nums

        r = []
        for i in range(len(nums) - k + 1):
            r.append(max(nums[i: i+k]))
        
        return r
```

이 풀이 역시 해당 case에서 타임아웃이 되고 만다. 결국, 매번 윈도우의 최대값을 계산하는 max()함수 때문에 시간 복잡도는 O(k*n)으로 크다. 이를 최적화 할 수 있는 방법을 찾아야 한다.

- 풀이: 큐를 이용한 최적화

매번 max를 계산하는 것보다, 새로 들어온 값이 기존 max값 보다 크면 대체하는 방식으로 가면 매번 max() 함수를 사용할 필요가 없어지게 된다. 이는 곧 [선입선출(FIFO)](https://junstar21.github.io/python%20algorithm%20interview/stack_queue_ll/#%ED%81%90)형태로 풀이할 수 있으며, 대표저인 자료형인 큐를 사용하여 다음과 같이 구현한다.

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        results = []
        
        # Queue를 사용하기 위해 collections의 deque를 사용
        window = collections.deque()
        
        # 시스템 상에서 가장 최소값을 지정
        current_max = float('-inf')
        for i, v in enumerate(nums):
            # 할당한 윈도우에 값을 삽입
            window.append(v)
            
            # 제시된 윈도우(k)보다 i값(현재 인덱스)가 작으면 아래 코드를 무시하고 처음으로 돌아간다.
            if i < k - 1:
                continue

            # 새로 추가된 값이 기존 최대값보다 큰 경우 교체
            if current_max == float('-inf'):
                current_max = max(window)
            elif v > current_max:
                current_max = v

            results.append(current_max)

            # 최대값이 윈도우에서 빠지면 초기화
            # popleft()를 통해 윈도우가 이동하면서 동시에 최대값이 빠지는지 판별한다.
            if current_max == window.popleft():
                current_max = float('-inf')
        return results
```

교재 발간 기준으로는 테스트를 통과하였지만 현 시점(24년 1월 11일 기준)에서는 또 하나의 테스트 케이스를 통과하지 못했다. 이를 개선하기 위해 다른 풀이를 사용하였다.

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        results = []
        window = deque()
        
        for i, v in enumerate(nums):
            # Remove elements outside of the current window
            while window and window[0] < i - k + 1:
                window.popleft()
            
            # Remove elements smaller than the current value from the back
            while window and nums[window[-1]] < v:
                window.pop()
            
            window.append(i)

            # Append the maximum value to the result list when the window is complete
            if i >= k - 1:
                results.append(nums[window[0]])

        return results
```

### [부분 문자열이 포함된 최소 윈도우](https://leetcode.com/problems/minimum-window-substring/)

문자열 S와 T를 입력받아 O(n)에 T의 모든 문자가 포함된 S의 최소 윈도우를 찾아라.

```
Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
---
Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.
---
Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.
```

- 풀이: 투 포인터, 슬라이싱 윈도우로 최적화

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        # collection.Counter()로 필요한 문자열과 개수를 지정한다
        need = collections.Counter(t)

        # 필요한 문자열의 길이를 정한다
        missing = len(t)

        # 포인터 설정
        left = start = end = 0

        # 오른쪽 포인터 이동
        for right, char in enumerate(s, 1):
            # 해당 문자열 개수가 기록된 need에서 문자열이 양수일 때,
            # 즉, 필요 문자열이 양수일 때만 missing을 하나 씩 제거
            missing -= need[char] > 0

            # s의 모든 문자열이 need에 개수 -1 카운팅
            need[char] -= 1

            # 필요 문자가 0이면 왼쪽 포인터 이동 판단
            if missing == 0:
                # 필요 문자(need)가 음수일 경우, t보다 더 많은 필요 문자열이 나왔기에
                # 왼쪽 포인터를 이동시켜 t가 필요한 최소한의 문자열만 확보
                while left < right and need[s[left]] < 0:
                    need[s[left]] += 1
                    left += 1
                # 가장 작은 값인 경우 start와 end를 지정
                if not end or right - left <= end - start:
                    start, end = left, right
                need[s[left]] += 1
                missing += 1
                left += 1
        # 슬라이싱 기법으로 정답 반환
        return s[start:end]
```

- 풀이: Counter로 좀 더 편리한 풀이

```python
class Solution:
    def minWindow(self, s: str, t: str) -> str:
        t_count = collections.Counter(t)
        current_count = collections.Counter()

        start = float('-inf')
        end = float('inf')

        left = 0
        # 오른쪽 포인터 이동
        for right, char in enumerate(s, 1):
            current_count[char] += 1

            # AND 연산 결과로 왼쪽 포인터 이동 판단
            while current_count & t_count == t_count:
                if right - left < end - start:
                    start, end = left, right
                current_count[s[left]] -= 1
                left += 1

        return s[start: end] if end - start <= len(s) else ''
```

전보다 코드가 간결해졌지만, 처리 속도는 기존 코드보다 10배 이상 느려지는 것을 확인 할 수 있다. 편리하긴 하나, 실무나 코딩 테스트에서는 사용하기 어려운 풀이가 될 것으로 예상한다.