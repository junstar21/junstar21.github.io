---
title: "선형 자료구조 - 스택, 큐 l"
excerpt: "2023-04-25 Linear Data Structures - Stack, Queue l"

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

스택(Stack)과 큐(Queue)는 가장 고전적인 자료구조 중 하나이며, 특히 스택은 대부분의 애플리케이션을 만들 때 사용되는 자료구조이다. 스택은 LIFO(Last-In-First-Out, 후입선출)로 처리된다. 접시를 쌓아 올리면 마지막에 쌓은 접시가 맨 위에 놓이고, 마지막에 놓인 접시가 먼저 꺼내게 되는 형태와 비슷하다. 큐는 FIFO(First-In-First-Out, 선입선출)로 처리되며 먼저 줄을 선 사람이 먼저 입장하는 형태와 비슷하다.

## 스택

- `push()` : 요소를 컬렉션에 추가
- `pop()` : 아직 제거되지 않은 가장 마지막에 삽입된 요소를 제거

### 연결 리스트를 이용한 스택 ADT 구현

```python
# 연결리스트를 담을 Node 클래스 정의
class Node:
	def __init__(self, item, next):
		self.item = item # 노드의 값
		self.next = next # 다음 노드를 가리키는 포인터

# Stack의 연산인 push()와 pop() 정의
class Stack:
	def __init__(self):
		self.last = None

	def push(self, item):
		# 연결 리스트에 요소를 추가하면서 가장 마지막 값을 next로 지정하고 포인터인 last를 마지막으로 이동
		self.last = Node(item, self.last)

	def pop(self):
		# 가장 마지막 아이템을 끄집어내고 last 포인터를 한칸 앞으로 전진
		item = self.last.item
		self.last = self.last.next
		return item
```

### [유효한 괄호](https://leetcode.com/problems/valid-parentheses/)

괄호의 입력값이 올바른지 판별하라.

```
Input: s = "()[]{}"
Output: true

Input: s = "(]"
Output: false
```

- 풀이

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        # dictionary를 이용한 table 구성
        table = {
            ')' : '(',
            '}' : '{',
            ']' : '[',
        }

        for char in s:
            # 여는 괄호들은 stack에 보내기
            if char not in table:
                stack.append(char)
            # 예외처리 & table에 있는 char의 value값이 stack.pop()과 일치하지 않으면 False 반환
            elif not stack or table[char] != stack.pop():
                return False
        return len(stack) == 0
```

## [중복 문자 제거](https://leetcode.com/problems/remove-duplicate-letters/)

중복된 문자를 제외하고 사전식 순서로 나열하라.

```
Input: s = "bcabc"
Output: "abc"

Input: s = "cbacdcbc"
Output: "acdb"
```

- 문제 설명

첫번 째 예제는 중복 문자를 제외하고 순서대로 나열한 결과값이다. ‘bcabc’에서 앞쪽의 b와 c는 뒤에도 b와 c가 나오기 때문에 중복이므로 제외를 시킨 결과값을 나타낸다. 만약, ‘bcabc’가 아닌 ‘dbcabc’가 나온다면, ‘dabc’가 되어야 한다. 왜냐하면 d는 중복값을 가지지 않기 때문에 첫번째 문자열로 나오게 되며 사전식 순서로 나열하기에 이 후 순서는 고려하지 않고 중복값만 제외한다. 이는 두번 째 예제에서 잘 설명된다. ‘cbacdcbc’라는 문자열 속에서 a만 유일하게 중복값이 없다. 따라서, a가 첫번 째 문자로 나오게 된다. 이후, 사전식 순서로 배열해야 하기 때문에 바로 뒤에 나오는 알파벳 c, d, b 순으로 나열하게 되고 중복문자들은 제거하게 된다. 

- 풀이 1 : 재귀를 이용

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        # 집합으로 정렬
        for char in sorted(set(s)):
            # 접미사 suffix를 만들어 분리하고 확인
            # s에 해당하는 알파벳의 인덱스를 반환하여 슬라이싱 기법으로 범위 지정
            suffix = s[s.index(char):]
            # 전체 집합과 접미사 집합이 일치할때 분리 진행
            if set(s) == set(suffix):
                return char + self.removeDuplicateLetters(suffix.replace(char, ''))
        return ''
```

- 풀이 2 : 스택을 이용한 문자 제거

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        # collections.Counter() : 리스트 안에 요소들이 몇개 있는지를 딕셔너리 형태로 반환
        counter, seen, stack = collections.Counter(s), set(), []

        for char in s:
            counter[char] -= 1
            if char in seen:
                continue
            # 뒤에 붙일 문자가 남아있다면 스택에서 제거
            while stack and char < stack[-1] and counter[stack[-1]] > 0:
                seen.remove(stack.pop())
            stack.append(char)
            seen.add(char)

        return ''.join(stack)
```