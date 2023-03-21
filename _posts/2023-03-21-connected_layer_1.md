---
title: "선형 자료구조 - 연결리스트 1"
excerpt: "2023-03-21 Linear Data Structures - Conneted layers 1"

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

## 연결 리스트

> 연결 리스트는 데이터 요소의 선형 집합으로, 데이터의 순서가 메모리에 물리적인 순서대로 저장되지는 않는다.
> 

연결 리스트(Linked List)는 컴퓨터과학에서 배열과 함께 가장 기본이 되는 대표적인 선형 자료구조 중 하나로 다양한 추상 자료형 구현의 기반이 된다. 

연결 리스트는 배열과 달리 특정 인덱스에 접근하기 위해서는 전체를 순서대로 읽어야 하므로 O(n)이 소요 된다. 반면, 시작 또는 끝 지점에 아이템을 추가하거나 삭제, 추출하는 작업은 O(1)에 가능하다.

[연결리스트에 대한 기초 내용](https://jae04099.tistory.com/entry/%EC%9E%90%EB%A3%8C%EA%B5%AC%EC%A1%B0-Linked-List%EC%97%B0%EA%B2%B0%EB%A6%AC%EC%8A%A4%ED%8A%B8-%EA%B8%B0%EC%B4%88-python)

## [팰린드롬 연결 리스트](https://leetcode.com/problems/palindrome-linked-list/)

연결 리스트가 팰린드롬 구조인지 판별하라.

```python
Input: head = [1,2,2,1]
Output: true

Input: head = [1,2]
Output: false
```

- 리스트 변환

```python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        q: List = [] # 받을 리스트 선언

        if not head: 
            return True

        node = head
        # 리스트 변환
        while node is not None: 
            q.append(node.val)
            node = node.next

        # 팰린드롬 판별
        while len(q) > 1:
            if q.pop(0) != q.pop():
                return False

        return True
```

- 런너를 이용한 풀이

```python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        rev = None
        slow = fast = head
        # 런너를 이용해 역순 연결 리스트 구성
        while fast and fast.next:
            fast = fast.next.next
            rev, rev.next, slow = slow, rev, slow.next
        if fast:
            slow = slow.next

        # 팰린드롬 여부 확인
        while rev and rev.val == slow.val:
            slow, rev = slow.next, rev.next
        return not rev
```

🤔 **런너 기법이란?**

> 연결 리스트를 순회할 때 2개의 포인터를 사용하는 기법이다. 한 포인터는 빠르게, 한 포인터는 느리게 이동하며 병합 지점이나 중간 위치, 길이 등을 판별 할 때 유용하게 사용할 수 있다.
> 

## [두 정렬 리스트의 병합](https://leetcode.com/problems/merge-two-sorted-lists/)

정렬되어있는 리스트 두개를 합쳐라.

```
Input: list1 = [1,2,4], list2 = [1,3,4]
Output: [1,1,2,3,4,4]
```

- 풀이 : 재귀 구조로 연결

```python
class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        # 우선순위가 높은 순 : 연산> -> not l1 -> and -> or
        if (not list1) or (list2 and (list1.val > list2.val)):
            list1, list2 = list2, list1
        if list1:
            list1.next = self.mergeTwoLists(list1.next, list2)
        return list1
```

해당 문제풀이는 연결리스트이기 때문에, `list1`과 `list2`를 스왑할 때 변수만 스왑되는 것이 아닌, 뒤에 연결된 값까지 같이 스왑되는 것에 유의하도록 하자.

## [역순 연결 리스트](https://leetcode.com/problems/reverse-linked-list/)

연결리스트를 뒤집어라

```
Input: head = [1,2,3,4,5]
Output: [5,4,3,2,1]
```

- 나의 풀이 : 반복문으로 뒤집기

```python
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
				# node = head : node는 연결리스트의 첫번째 값을 가리킴
				# prev = None : prev는 연결리스트의 마지막 값을 가리킴
        node, prev = head, None

        while node:
						# next = node.next : next는 node의 다음 값의 연결지점을 의미함
						# node.next = prev : node의 next(다음을 가리키는 지점)이 prev임을 의미
            next, node.next = node.next, prev
            prev, node = node, next

        return prev
```

입력되는 연결리스트가 1→2→3→4→5→None이라고 하자. `node, prev = head, None`에서 node는 head, 1을 가리키고 prev는 None을 가리킨다. `while`문으로 들어가면 `next, node.next = node.next, prev`이 실행된다. 따로 나눠서 구문을 살펴보자. `next = node.next`는 현재 지정된 `node`의 다음값을 의미한다. 현 상황에서 `node`는 1을 가리키고 있기 때문에 `node.next`의 값은 2가 된다. 이제 다음 구문인 `prev, node = node, next`로 넘어가도록 하자. `prev = node`는 처음에 지정한 `prev = None`에서 `prev = node`로 바꾼다는 것을 의미한다. 현재의 `node`는 1을 가리키고 있기에 `prev`는 1이 된다. `node = next`는 `node`를 다음 값으로 이동시켜준다. 현재 지정된 `node`는 1이기에 2로 옮기도록 한다. 지금까지 진행된 prev의 연결리스트 상태를 정리하면 ‘`None ← 1`’이 된다. 이런 식으로 `node`가 `None`이 될때까지 구문을 반복하게 되며, `prev`에 생기는 연결리스트는 `head`의 역순으로 연결되게 된다.

## [페어의 노드 스왑](https://leetcode.com/problems/swap-nodes-in-pairs/)

연결 리스트를 입력받아 페어 단위로 스왑하라.

```
Input: head = [1,2,3,4]
Output: [2,1,4,3]
```

- 나의 풀이

```python
class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:

        cur = head

        while cur and cur.next:
						# 값만 교환
            cur.val, cur.next.val = cur.next.val, cur.val
            cur = cur.next.next

        return head
```

연결 역순 리스트와 비슷한 알고리즘을 사용하여 문제를 풀이하였다. 

## [홀짝 연결리스트](https://leetcode.com/problems/odd-even-linked-list/)

연결 리스트 홀수 노드 다음에 짝수 노드가 오도록 작성하라. 공간복잡도 O(1), 시간복잡도 O(n)

```
Input: head = [1,2,3,4,5]
Output: [1,3,5,2,4]
---
Input: head = [2,1,3,5,6,4,7]
Output: [2,3,6,7,1,5,4]
```

- 풀이 : 반복구조로 홀짝 처리

```python
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None:
            return None

        odd = head # 홀수 노드 포인터
        even = head.next # 짝수 노드 포인터
        even_head = head.next

        while even and even_head:
            odd.next, even.next = odd.next.next, even.next.next
            odd, even = odd.next, even.next

        odd.next = even_head

        return head
```