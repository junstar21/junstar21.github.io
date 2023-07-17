---
title: "선형 자료구조 - 데크, 우선순위 큐"
excerpt: "2023-07-06 Deque, Priory stack"

# layout: post
categories:
  - Python algorithm interview
tags:
  - python
  - Stack
  - Leet code
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
해당 내용은 '[파이썬 알고리즘 인터뷰](https://product.kyobobook.co.kr/detail/S000001932748)' 책의 일부를 발췌하여 정리한 내용입니다.

## 데크

> 데크(Deque)는 더블 엔디드 큐(Double-Ended Queue)의 줄임말로, 글자 그대로 양쪽 끝을 모두 추출할 수 있는, 큐를 일반화한 형태의 추상 자료형(ADT)이다.
> 

### [원형 데크 디자인](https://leetcode.com/problems/design-circular-deque/)

- 풀이

```python
class MyCircularDeque:

    def __init__(self, k: int):
        self.head, self.tail = ListNode(None), ListNode(None)
        self.k, self.len = k, 0
        self.head.right, self.tail.left = self.tail, self.head

    # 이중 연결 리스트에 신규 노드 삽입
    def _add(self, node: ListNode, new: ListNode):
        n = node.right
        node.right = new
        new.left, new.right = node, n
        n.left = new

    def _del(self, node: ListNode):
        n = node.right.right
        node.right = n
        n.left = node

    def insertFront(self, value: int) -> bool:
        if self.k == self.len:
            return False
        self.len += 1
        self._add(self.head, ListNode(value))
        return True

    def insertLast(self, value: int) -> bool:
        if self.k == self.len:
            return False
        self.len += 1
        self._add(self.tail.left, ListNode(value))
        return True

    def deleteFront(self) -> bool:
        if self.len == 0:
            return False
        self.len -= 1
        self._del(self.head)
        return True
        
    def deleteLast(self) -> bool:
        if self.len == 0:
            return False
        self.len -= 1
        self._del(self.tail.left.left)
        return True

    def getFront(self) -> int:
        return self.head.right.val if self.len else -1
 
    def getRear(self) -> int:
        return self.tail.left.val if self.len else -1
        
    def isEmpty(self) -> bool:
        return self.len == 0

    def isFull(self) -> bool:
        return self.len == self.k      

# Your MyCircularDeque object will be instantiated and called as such:
# obj = MyCircularDeque(k)
# param_1 = obj.insertFront(value)
# param_2 = obj.insertLast(value)
# param_3 = obj.deleteFront()
# param_4 = obj.deleteLast()
# param_5 = obj.getFront()
# param_6 = obj.getRear()
# param_7 = obj.isEmpty()
# param_8 = obj.isFull()
```

## 우선순위 큐

> 우선순위 큐는 큐 또는 스택과 같은 추상 자료형과 유사하지만 추가로 각 요소의 ‘우선순위’와 연관되어 있다.
> 

우선순위 큐는 특정 조건에 따라 우선순위가 가장 높은 요소가 추출되는 자료형이다(ex. 최대값 추출). 이외에도 최단 경로를 탐색하는 다익스트라(Dijkstra)알고리즘 등 우선순위 큐는 다양한 분야에 활용되며, Heap 자료구조와도 관련이 깊다.(Heap은 차후에 포스팅할 예정이다.)

### k개 정렬 리스트 병합

k개의 정렬된 리스트를 1개의 정렬된 리스트로 병합하라.

```
Input: lists = [[1,4,5],[1,3,4],[2,6]]
Output: [1,1,2,3,4,4,5,6]
Explanation: The linked-lists are:
[
  1->4->5,
  1->3->4,
  2->6
]
merging them into one sorted list:
1->1->2->3->4->4->5->6
```

- 풀이

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:

        root = result = ListNode(None)
        heap = []

        # 각 연결 리스트의 루트를 힙에 저장
        for i in range(len(lists)):
            if lists[i]:
                heapq.heappush(heap, (lists[i].val, i, lists[i]))

        # 힙 추출 이후 다음 노드는 다시 저장
        while heap:
            node = heapq.heappop(heap)
            idx = node[1]
            result.next = node[2]

            result = result.next
            if result.next:
                heapq.heappush(heap, (result.next.val, idx, result.next))

        return root.next
```

**🤔 Heap연산이란?**

Heap은 트리 기반의 자료구조이다. 부모는 자식보다 항상 작다는 조건을 충족하는 형태이다. 다만, 부모 노드가 하상 작다는 조건만 만족할 뿐 서로 정렬되어 있지는 않다.

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-07-06-deque_priory_stack/Untitled.png)