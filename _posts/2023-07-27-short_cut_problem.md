---
title: "비선형 자료구조 - 최단 경로 문제"
excerpt: "2023-07-27 Non-linear Data Structures - Short Cut Problem"

# layout: post
categories:
  - Python algorithm interview
tags:
  - python
  - Short Cut
  - Leet code
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---

> 최단 경로 문제는 각 간선의 가중치 합이 최소가 되는 두 정점(또는 노드) 사이의 경로를 찾는 문제다.
> 

쉽게 생각하면 네입게이션으로 목적지 경로를 탐색할 때, 최적의 경로 문제가 바로 최소 비용이 되는 최단 경로 문제다. 가장 유명한 최단 경로 알고리즘은 다익스트라(Dijkstra) 알고리즘일 것이다. 다익스트라 알고리즘은 항상 노드 주변의 최단 경로만을 택하는 대표적인 그리디 알고리즘 중 하나로, 단순할 뿐만 아니라 실행 속도 또한 빠르다. 다익스트라 알고리즘은 BFS를 이용하는 대표적인 알고리즘이다.

### 네트워크 딜레이 타임

K부터 출발해 모든 노드가 신호를 받을 수 있는 시간을 계산하라. 불가능할 경우 -1을 리턴한다. 입력값(u, v, w)는 각각 출발지, 도착지, 소요 시간으로 구성되며, 전체 노드의 개수는 N으로 입력받는다.

```
Input: times = [[2,1,1],[2,3,1],[3,4,1]], n = 4, k = 2
Output: 2
---
Input: times = [[1,2,1]], n = 2, k = 1
Output: 1
---
Input: times = [[1,2,1]], n = 2, k = 2
Output: -1
```

- 풀이: 다익스트라 알고리즘 구현

모든 노드가 신호를 받는데 걸리는 시간은 결국 가장 오래 걸리는 노드까지의 시간이라고 볼 수 있다. 이는 앞선 설명한 다익스트라 알고리즘으로 추출할 수 있다.

```python
import collections
import heapq

class Solution:
    def networkDelayTime(self, times: List[List[int]], N: int, K: int) -> int:
        graph = collections.defaultdict(list)
        # 그래프 인접 리스트 구성
        for u, v, w in times:
            graph[u].append((v, w))

        # 큐 변수: [(소요 시간, 정점)]
        Q = [(0, K)]
        dist = collections.defaultdict(int)

        # 우선 순위 큐 최소값 기준으로 정점까지 최단 경로 삽입
        while Q:
            time, node = heapq.heappop(Q)
            if node not in dist:
                dist[node] = time
                for v, w in graph[node]:
                    alt = time + w
                    heapq.heappush(Q, (alt, v))

        # 모든 노드 최단 경로 존재 여부 판별
        if len(dist) == N:
            return max(dist.values())
        return -1
```

### K 경유지 내 가장 저렴한 항공권

시작점에서 도착점까지의 가장 저렴한 가격을 계산하되, K개의 경유지 이내에 도착하는 가격을 리턴하라. 경로개 존재하지 않을 경우 -1을 리턴한다.

```
Input: n = 4, flights = [[0,1,100],[1,2,100],[2,0,100],[1,3,600],[2,3,200]], src = 0, dst = 3, k = 1
Output: 700
Explanation:
The graph is shown above.
The optimal path with at most 1 stop from city 0 to 3 is marked in red and has cost 100 + 600 = 700.
Note that the path through cities [0,1,2,3] is cheaper but is invalid because it uses 2 stops.
```

- 풀이: 다익스트라 알고리즘 응용

지난 번 문제풀이와 비슷한 형태이다. 다만, 변수들과 구해야하는 값이 다르기 때문에 이 부분만 변형을 시켜준다.

```python
import collections
import heapq

class Solution:
    def findCheapestPrice(self, n: int, flights: List[List[int]], src: int, dst: int, K: int) -> int:
        graph = collections.defaultdict(list)
        # 그래프 인접 리스트 구성
        for u, v, w in flights:
            graph[u].append((v, w))

        # 큐 변수: [(가격, 정점, 남은 가능 경유지 수)]
        Q = [(0, src, K)]

        # 우선 순위 큐 최소값 기준으로 도착점까지 최소 비용 판별
        while Q:
            price, node, k = heapq.heappop(Q)
            if node == dst:
                return price
            if k >= 0:
                for v, w in graph[node]:
                    alt = price + w
                    heapq.heappush(Q, (alt, v, k - 1))
        return -1
```

경로를 묻는 문제가 아니기에 dist와 전체 경로 탐색 여부를 제거하였다.