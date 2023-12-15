---
title: "알고리즘 - 이진 정렬 ll"
excerpt: "2023-12-15 Algorithm - Bineary Search ll"

# layout: post
categories:
  - Python algorithm interview
tags:
  - python
  - algorithm
  - bineary Search
  - Leet code
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
해당 내용은 '[파이썬 알고리즘 인터뷰](https://product.kyobobook.co.kr/detail/S000001932748)' 책의 일부를 발췌하여 정리한 내용입니다.

### [2D 행렬 검색 ll](https://leetcode.com/problems/search-a-2d-matrix-ii/)

m*n 행렬에서 값을 찾아내는 효율적인 알고리즘을 구현하라.

```
Input: matrix = [[1,4,7,11,15],[2,5,8,12,19],[3,6,9,16,22],[10,13,14,17,24],[18,21,23,26,30]], target = 5
Output: true
```

- 나의 풀이 : 이중 for문

우선 간단하게 이중 for문을 사용하여 문제를 풀이하였다. 파이썬다운 방식으로 풀이하려고 노력한 코드이나, 시간상 효율적인 코드는 아니다.

```python
class Solution:
    def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
        for line in matrix:
            for i in line:
                if i == target:
                    return True
```

- 풀이: 행 뒷쪽에서 탐색

```python
class Solution:
    def searchMatrix(self, matrix, target):
        # 예외 처리
        if not matrix:
            return False

        # 첫 행의 맨 뒤
        row = 0
        col = len(matrix[0]) - 1

        while row <= len(matrix) - 1 and col >= 0:
            if target == matrix[row][col]:
                return True
            # 타겟이 작으면 왼쪽으로
            elif target < matrix[row][col]:
                col -= 1
            # 타겟이 크면 아래로
            elif target > matrix[row][col]:
                row += 1
        return False
```

- 풀이: 가장 파이썬다운 방식

```python
class Solution:
    def searchMatrix(self, matrix, target):
        return any(target in row for row in matrix)
```

**🤔 any()와 all() 함수**

any()는 포함된 값중 하나라도 참이면 항상 참으로 출력한다.

```python
>>> any([True, False, False)]
True
```

반면 all()은 모든 값이 참이여야 True를 출력한다.

```python
>>> all([True, False, False)]
False
>>> all([True, True, True)]
True
```