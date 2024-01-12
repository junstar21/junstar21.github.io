---
title: "알고리즘 - 슬라이딩 윈도우 ll"
excerpt: "2024-01-12 Sliding Window ll"

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

### [가장 긴 반복 문자 대체](https://leetcode.com/problems/longest-repeating-character-replacement/)

대문자로 구성된 문자열 s가 주어졌을 떄 k번만큼의 변경으로 만들 수 있는, 연속으로 반복된 문자열의 가장 긴 길이를 출력하라.

```
Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace the two 'A's with two 'B's or vice versa.
---
Input: s = "AABABBA", k = 1
Output: 4
Explanation: Replace the one 'A' in the middle with 'B' and form "AABBBBA".
The substring "BBBB" has the longest repeating letters, which is 4.
There may exists other ways to achieve this answer too.
```

- 풀이: 투 포인터, 슬라이딩 윈도우, Counter 모두 이용

```python
class Solution:
    def characterReplacement(self, s: str, k: int) -> int:
        left = right = 0
        counts = collections.Counter()
        for right in range(1, len(s) + 1):
            # 문자가 등장할 때 마다 개수 추가
            counts[s[right - 1]] += 1
            
            # 가장 흔하게 등장하는 문자 탐색
            max_char_n = counts.most_common(1)[0][1]

            # k 초과시 왼쪽 포인터 이동
            if right - left - max_char_n > k:
                counts[s[left]] -= 1
                left += 1
        return right - left
```