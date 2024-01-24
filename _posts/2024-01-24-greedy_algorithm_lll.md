---
title: "알고리즘 - 그리디 알고리즘 lll"
excerpt: "2024-01-24 Greedy Algorithm lll"

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

### [쿠키 부여](https://leetcode.com/problems/assign-cookies/)

아이들에게 1개씩 쿠키를 나눠줘야 한다. 각 아이 child_i마다 그팩터(Greed Factor)`gi`를 갖고 있으며, 이는 아이가 만족하는 최소 쿠키의 크기를 말한다. 각 쿠키 `cookie_j`는 크기 `sj`를 갖고 있으며, `sj≥gi`이어야 아이가 만족하는 쿠키를 받는다. 최대 몇명의 아이들에게 쿠키를 줄 수 있는지 출력하라.

```
Input: g = [1,2,3], s = [1,1]
Output: 1
Explanation: You have 3 children and 2 cookies. The greed factors of 3 children are 1, 2, 3.
And even though you have 2 cookies, since their size is both 1, you could only make the child whose greed factor is 1 content.
You need to output 1.
---
Input: g = [1,2], s = [1,2,3]
Output: 2
Explanation: You have 2 children and 3 cookies. The greed factors of 2 children are 1, 2.
You have 3 cookies and their sizes are big enough to gratify all of the children,
You need to output 2.
```

- 나의 풀이: 그리디 알고리즘

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        # 순서대로 정렬
        g.sort()
        s.sort()

        g_pos = s_pos = 0
        # 그리디 진행
        while len(g) > g_pos and len(s) > s_pos:
            if s[s_pos] >= g[g_pos]:
                g_pos += 1
            s_pos += 1

        return g_pos
```

- 풀이: 이진탐색

```python
class Solution:
    def findContentChildren(self, g: List[int], s: List[int]) -> int:
        # 순서대로 정렬
        g.sort()
        s.sort()

        result = 0

        for i in s:
            # 이진 검색으로 더 큰 인덱스 탐색
            index = bisect.bisect_right(g, i)
            if index > result:
                result += 1
        return result
```

`bisect.bisect_right()`은  찾아낸 값의 다음 인덱스를 반환하는 기능을 가지고 있다.

```python
>>> bisect.bisect_left([1,2,3,4], 3)
2
>>> bisect.bisect_right([1,2,3,4], 3)
3
```