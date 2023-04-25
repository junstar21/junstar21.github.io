---
title: "선형 자료구조 - 역순 연결리스트 ll"
excerpt: "2023-04-17 Linear Data Structures - Reverse connected list ll"

# layout: post
categories:
  - Python algorithm interview
tags:
  - python
  - Conneted layer
  - Leet code
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
해당 내용은 '[파이썬 알고리즘 인터뷰](https://product.kyobobook.co.kr/detail/S000001932748)' 책의 일부를 발췌하여 정리한 내용입니다.

## [역순 연결리스트 ll](https://leetcode.com/problems/reverse-linked-list-ii/)

주어진 인덱스 구간 사이를 역순으로 정렬하라.

```
Input: head = [1,2,3,4,5], left = 2, right = 4
Output: [1,4,3,2,5]
```

### 풀이 : 반복 구조로 노드 뒤집기

```python
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:

        # 예외처리를 위한 구문
        if not head or left == right:
            return head

        # 루트값과 시작점 설정
        root = start = ListNode(None)
        root.next = head

        # 시작값과 끝값 설정
        for _ in range(left - 1):
            start = start.next
        end = start.next

        for _ in range(right - left):
            tmp, start.next, end.next = start.next, end.next, end.next.next
            start.next.next = tmp

        return root.next
```