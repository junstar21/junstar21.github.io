---
title: "알고리즘 - 그리디 알고리즘 ll"
excerpt: "2024-01-23 Greedy Algorithm ll"

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

### [태스크 스케줄러](https://leetcode.com/problems/task-scheduler/)

A에서 Z로 표현된 태스크가 있다. 각 간격마다 CPU는 한 번의 태스크만 실행할 수 있고, n번의 간격 내에는 동일한 태스크를 실행할 수 없다. 더 이상 태스크를 실행할 수 없을 경우 아이들(idle)상태가 된다. 모든 태스크를 실행하기 위한 최소 간격을 출력하라.

```
Input: tasks = ["A","A","A","B","B","B"], n = 2
Output: 8
Explanation:
A -> B -> idle -> A -> B -> idle -> A -> B
There is at least 2 units of time between any two same tasks.
---
Input: tasks = ["A","A","A","B","B","B"], n = 0
Output: 6
Explanation: On this case any permutation of size 6 would work since n = 0.
["A","A","A","B","B","B"]
["A","B","A","B","A","B"]
["B","B","B","A","A","A"]
...
And so on.
---
Input: tasks = ["A","A","A","A","A","A","B","C","D","E","F","G"], n = 2
Output: 16
Explanation:
One possible solution is
A -> B -> C -> A -> D -> E -> A -> F -> G -> A -> idle -> idle -> A -> idle -> idle -> A
```

- 풀이: 우선순위 큐 이용

```python
class Solution:
    def leastInterval(self, tasks: List[str], n: int) -> int:
        counter = collections.Counter(tasks)
        result = 0

        while True:
            sub_count = 0

            # most_common() : 개수가 많은 순으로 정렬된 튜플 배열 리스트를 리턴
            for task, _ in counter.most_common(n + 1):
                sub_count += 1
                result += 1

                # 실행된 태스크를 제외
                counter.subtract(task)

                # 0 이하인 아이템을 목록에서 완전히 제거
                # 비어있는 collections.Counter()를 더하여 0 이하를 제거
                counter += collections.Counter()

            if not counter:
                break

            result += n - sub_count + 1

        return result
```

### [주유소](https://leetcode.com/problems/gas-station/)

원형으로 경로가 연결된 주유소 목록이 있다. 각 주유소는 `gas[i]`만큼의 기름을 갖고 있으며, 다음 주유소로 이동하는데 `cost[i]`가 필요하다. 기름이 부족하면 이동할 수 없다고 할 떄 모든 주유소를 방문할 수 있는 출발점의 인덱스를 출력하라. 출발점이 존재하지 않을 경우 -1을 리턴하며, 출발점은 유일하다.

```
Input: gas = [1,2,3,4,5], cost = [3,4,5,1,2]
Output: 3
Explanation:
Start at station 3 (index 3) and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 4. Your tank = 4 - 1 + 5 = 8
Travel to station 0. Your tank = 8 - 2 + 1 = 7
Travel to station 1. Your tank = 7 - 3 + 2 = 6
Travel to station 2. Your tank = 6 - 4 + 3 = 5
Travel to station 3. The cost is 5. Your gas is just enough to travel back to station 3.
Therefore, return 3 as the starting index.
---
Input: gas = [2,3,4], cost = [3,4,3]
Output: -1
Explanation:
You can't start at station 0 or 1, as there is not enough gas to travel to the next station.
Let's start at station 2 and fill up with 4 unit of gas. Your tank = 0 + 4 = 4
Travel to station 0. Your tank = 4 - 3 + 2 = 3
Travel to station 1. Your tank = 3 - 3 + 3 = 3
You cannot travel back to station 2, as it requires 4 unit of gas but you only have 3.
Therefore, you can't travel around the circuit once no matter where you start.
```

- 풀이: 전체 주유소 연료와 사용량 총합을 통해 시작점 추리기

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        # 모든 주유소를 방문 할 수 있는 가스가 남아있는지 판별
        if sum(gas) < sum(cost):
            return -1

        start, fuel = 0, 0

        for i in range(len(gas)):
            # 출발이 불가능한 지점 판단
            if gas[i] + fuel < cost[i]:
                start = i + 1
                fuel = 0
            # 출발이 가능하면 연료를 계산
            else:
                fuel += gas[i] - cost[i]
            
        return start
```