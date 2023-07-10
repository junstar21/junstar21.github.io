---
title: "선형 자료구조 - 스택, 큐 ll"
excerpt: "2023-07-04 Linear Data Structures - Stack, Queue ll"

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

### [일일 온도](https://leetcode.com/problems/daily-temperatures/)

온도 리스트를 입력받아, 더 따듯한 날씨를 위해서는 며칠을 더 기다려야 하는지를 출력하라.

```
Input: temperatures = [73,74,75,71,69,72,76,73]
Output: [1,1,4,2,1,1,0,0]
```

- 풀이

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        
        # 정답값을 받을 리스트 선언
        answer = [0]* len(temperatures)
        # 낮은 온도값을 담아 둘 리스트 선언
        stack = []

        # enumerate : 인덱스 값, 리스트 값 반환
        for i, cur in enumerate(temperatures):
            # 리스트 값이 가장 마지막에 입력된 temperatures의 인덱스와 비교
            while stack and cur > temperatures[stack[-1]]:
                last = stack.pop()
                # 정답값 리스트의 인덱스를 수정
                answer[last] = i - last
            stack.append(i)

        return answer
```

## 큐

First-In-First-Out(FIFO, 선입선출)로 처리되는 큐는 스택에 비해 상대적으로 쓰임새가 적지만, 데크나 우선순위 큐(Priority Queue)같은 변형들은 여러 분야에서 매우 유용하게 쓰인다. 이외에도 너비탐색, 캐시 등 구현할 때 널리 사용된다.

### [큐를 이용한 스택 구현](https://leetcode.com/problems/implement-stack-using-queues/)

큐를 이용해 다음 연산을 지원하는 스택을 구현하라.

```
push(x): 요소 x를 스택에 삽입
pop(): 스택의 첫 번째 요소를 삭제
top(): 스택의 첫 번째 요소를 가져오기
empty(): 스택이 비어있는지 여부를 리턴

Input
["MyStack", "push", "push", "top", "pop", "empty"]
[[], [1], [2], [], [], []]
Output
[null, null, null, 2, 2, false]

Explanation
MyStack myStack = new MyStack();
myStack.push(1);
myStack.push(2);
myStack.top(); // return 2
myStack.pop(); // return 2
myStack.empty(); // return False
```

- 풀이

함수의 기능들을 사용하면 겉으로 볼 때는 스택의 구조를 가진 것 처럼 보이지만, 함수 내부에 작동하는 구조는 큐여야하는 문제이다. 스택과 큐의 자료 구성 순서를 기억하고 문제를 풀어야 한다.

```python
# 스택 : 선입후출
# 큐 : 선입선출

class MyStack:

    def __init__(self):
        self.q = collections.deque()

    def push(self, x: int) -> None:
        self.q.append(x)
        # 요소 삽입 후 맨 앞에 두는 상태로 재정렬
				# 스택은 가장 마지막에 들어간 값이 첫 번째로 나오게 해야함
        for _ in range(len(self.q) - 1):
            self.q.append(self.q.popleft())

    def pop(self) -> int:
        return self.q.popleft()

    def top(self) -> int:
        return self.q[0]

    def empty(self) -> bool:
        return len(self.q) == 0
```

### 스택을 이용한 큐 구현

스택을 이용해 다음 연산을 지원하는 큐를 구현하라.

```
push(x): 요소 x를 마지막에 삽입
pop(): 스택의 첫 번째 요소를 삭제
peek(): 스택의 첫 번째 요소를 가져오기
empty(): 스택이 비어있는지 여부를 리턴

Input
["MyQueue", "push", "push", "peek", "pop", "empty"]
[[], [1], [2], [], [], []]
Output
[null, null, null, 1, 1, false]

Explanation
MyQueue myQueue = new MyQueue();
myQueue.push(1); // queue is: [1]
myQueue.push(2); // queue is: [1, 2] (leftmost is front of the queue)
myQueue.peek(); // return 1
myQueue.pop(); // return 1, queue is [2]
myQueue.empty(); // return false
```

- 풀이

```python
class MyQueue:

    def __init__(self):
        # 스택 과정을 담을 리스트 선언
        self.input = []
        self.output = []

    def push(self, x: int) -> None:
        self.input.append(x)

    def pop(self) -> int:
        self.peek()
        return self.output.pop()

    def peek(self) -> int:
        if not self.output:
            while self.input:
                self.output.append(self.input.pop())
        return self.output[-1]

    def empty(self) -> bool:
        return self.input == [] and self.output == []

# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()
```

### [원형 큐 디자인](https://leetcode.com/problems/design-circular-queue/)

k개의 저장 공간을 가지는 원형 큐를 디자인하라.

```
Implement the MyCircularQueue class:

MyCircularQueue(k) Initializes the object with the size of the queue to be k.
int Front() Gets the front item from the queue. If the queue is empty, return -1.
int Rear() Gets the last item from the queue. If the queue is empty, return -1.
boolean enQueue(int value) Inserts an element into the circular queue. Return true if the operation is successful.
boolean deQueue() Deletes an element from the circular queue. Return true if the operation is successful.
boolean isEmpty() Checks whether the circular queue is empty or not.
boolean isFull() Checks whether the circular queue is full or not.
You must solve the problem without using the built-in queue data structure in your programming language.

Input
["MyCircularQueue", "enQueue", "enQueue", "enQueue", "enQueue", "Rear", "isFull", "deQueue", "enQueue", "Rear"]
[[3], [1], [2], [3], [4], [], [], [], [4], []]
Output
[null, true, true, true, false, 3, true, true, true, 4]

Explanation
MyCircularQueue myCircularQueue = new MyCircularQueue(3);
myCircularQueue.enQueue(1); // return True
myCircularQueue.enQueue(2); // return True
myCircularQueue.enQueue(3); // return True
myCircularQueue.enQueue(4); // return False
myCircularQueue.Rear();     // return 3
myCircularQueue.isFull();   // return True
myCircularQueue.deQueue();  // return True
myCircularQueue.enQueue(4); // return True
myCircularQueue.Rear();     // return 4
```

- 풀이

투 포인터 방식으로 접근하여 문제를 풀이한다.

```python
class MyCircularQueue:

    def __init__(self, k: int):
        self.q = [None] * k
        self.maxlen = k
        self.p1 = 0
        self.p2 = 0

    def enQueue(self, value: int) -> bool:
        if self.q[self.p2] is None:
            self.q[self.p2] = value
            self.p2 = (self.p2 + 1) % self.maxlen
            return True
        else:
            return False

    def deQueue(self) -> bool:
        if self.q[self.p1] is None:
            return False
        else:
            self.q[self.p1] = None
            self.p1 = (self.p1 + 1) % self.maxlen
            return True

    def Front(self) -> int:
        if self.q[self.p1] is None:
            return -1
        else:
            return self.q[self.p1]

    def Rear(self) -> int:
        if self.q[self.p2 - 1] is None:
            return -1
        else:
            return self.q[self.p2 - 1]        

    def isEmpty(self) -> bool:
        return self.p1 == self.p2 and self.q[self.p1] is None

    def isFull(self) -> bool:
        return self.p1 == self.p2 and self.q[self.p1] is not None

# Your MyCircularQueue object will be instantiated and called as such:
# obj = MyCircularQueue(k)
# param_1 = obj.enQueue(value)
# param_2 = obj.deQueue()
# param_3 = obj.Front()
# param_4 = obj.Rear()
# param_5 = obj.isEmpty()
# param_6 = obj.isFull()
```