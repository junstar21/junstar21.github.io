---
title: "알고리즘 - 이진 정렬 l"
excerpt: "2023-09-25 Algorithm - Bineary Search l"

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

## 이진 검색(Bineary Search)

> 이진 검색(Binary Search)이란 정렬된 배열에서 타겟을 찾는 검색 알고리즘이다.
> 

이진 검색(Binary Search)은 대표적인 로그 시간 알고리즘(O(log n))이며 이진 탐색 트리(Binary Search Tree)와 유사점이 많다.

### [이진 검색](https://leetcode.com/problems/binary-search/)

정렬된 nums를 입력받아 이진 검색으로 target에 해당하는 인덱스를 찾아라.

```
Input: nums = [-1,0,3,5,9,12], target = 9
Output: 4
Explanation: 9 exists in nums and its index is 4
---
Input: nums = [-1,0,3,5,9,12], target = 2
Output: -1
Explanation: 2 does not exist in nums so return -1
```

- 풀이: 재귀 풀이

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        def binary_search(left, right):
            if left <= right:
                mid = (left + right) // 2

                if nums[mid] < target:
                    return binary_search(mid + 1, right)
                elif nums[mid] > target:
                    return binary_search(left, mid - 1)
                else:
                    return mid
            else:
                return -1

        return binary_search(0, len(nums) - 1)
```

- 풀이: 반복 풀이

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2

            if nums[mid] < target:
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
            else:
                return mid
        return -1
```

- 풀이: 이진 검색 모듈

이진 검색 알고리즘을 지원하는 `bisect` 모듈을 기본으로 제공하기에 이를 사용하면 파이썬다운 방식으로 문제를 풀이할 수 있다.

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        index = bisect.bisect_left(nums, target)

        if index < len(nums) and nums[index] == target:
            return index
        else:
            return -1
```

**⚠️ 이진 검색 알고리즘 버그**

`mid = (left + right) // 2`의 알고리즘에는 사소한 버그가 하나 존재한다. 만약, left + right가 int 자료형이 허용하는 최대값을 넘어버리면 오버플로우(Overflow)문제가 발생하게 된다. 이를 그러면 어떻게 보안하여 구현하면 될까? `mid = left + (right - left) // 2`로 수정할 수 있다. 

### [회전 정렬된 배열 검색](https://leetcode.com/problems/search-in-rotated-sorted-array/description/)

특정 피벗을 기준으로 회전하여 정렬된 배열에서 target 값의 인덱스를 출력하라.

```
Input: nums = [4,5,6,7,0,1,2], target = 0
Output: 4
---
Input: nums = [4,5,6,7,0,1,2], target = 3
Output: -1
---
Input: nums = [1], target = 0
Output: -1
```

- 피벗을 기준으로 하는 이진 검색

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        # 예외 처리
        if not nums:
            return -1

        # 최소값 찾아 피벗 설정
        left, right = 0, len(nums) - 1
        while left < right:
            mid = left + (right - left) // 2

            if nums[mid] > nums[right]:
                left = mid + 1
            else:
                right = mid

        pivot = left
        # 피벗 기준 이진 검색
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            mid_pivot = (mid + pivot) % len(nums)

            if nums[mid_pivot] < target:
                left = mid + 1
            elif nums[mid_pivot] > target:
                right = mid - 1
            else:
                return mid_pivot
        return -1
```

### [두 배열의 교집합](https://leetcode.com/problems/intersection-of-two-arrays/)

두 배열의 교집합을 구해라.

```python
Input: nums1 = [1,2,2,1], nums2 = [2,2]
Output: [2]
---
Input: nums1 = [4,9,5], nums2 = [9,4,9,8,4]
Output: [9,4]
Explanation: [4,9] is also accepted.
```

- 나의 풀이: set()과 list()를 이용한 풀이

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        return list(set(nums1) & set(nums2))
```

- 풀이: 이진 검색으로 일치 여부 판별

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        result: Set = set()
        nums2.sort()
        for n1 in nums1:
            # 이진 검색으로 일치 여부 판별
            i2 = bisect.bisect_left(nums2, n1)
            if len(nums2) > 0 and len(nums2) > i2 and n1 == nums2[i2]:
                result.add(n1)

        return result
```

- 풀이: 가장 빠른 응답시간

```python
class Solution:
    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        s = set(nums1)
        common = set()
        for num in nums2:
            if num in s:
                common.add(num)
        return common
```

### [두 수의 합 II](https://leetcode.com/problems/two-sum-ii-input-array-is-sorted/)

정렬된 배열을 받아 덧셈을 하여 타겟을 만들 수 있는 배열의 두 숫자 인덱스를 리턴하라.

```
Input: numbers = [2,7,11,15], target = 9
Output: [1,2]
Explanation: The sum of 2 and 7 is 9. Therefore, index1 = 1, index2 = 2. We return [1, 2].
---
Input: numbers = [2,3,4], target = 6
Output: [1,3]
Explanation: The sum of 2 and 4 is 6. Therefore index1 = 1, index2 = 3. We return [1, 3].
---
Input: numbers = [-1,0], target = -1
Output: [1,2]
Explanation: The sum of -1 and 0 is -1. Therefore index1 = 1, index2 = 2. We return [1, 2].
```

- 풀이: 투 포인터

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        left, right = 0, len(numbers) - 1

        while not left == right:
            if numbers[left] + numbers[right] < target:
                left += 1
            elif numbers[left] + numbers[right] > target:
                right -= 1
            else:
                return left + 1, right + 1
```

- 풀이: bisect 모듈 활용

```python
class Solution:
    def twoSum(self, numbers: List[int], target: int) -> List[int]:
        for k, v in enumerate(numbers):
            expected = target - v
            i = bisect.bisect_left(numbers, expected, k + 1)
            if i < len(numbers) and numbers[i] == expected:
                return k + 1, i + 1
```

- 풀이: 가장 빠른 응답시간

```python
class Solution:
    def twoSum(self, numbers, target):
        num_dict = {}  # Create a dictionary to store numbers and their indices
        for i, num in enumerate(numbers):
            complement = target - num  # Calculate the complement needed to reach the target
            if complement in num_dict:
                return [num_dict[complement] + 1, i + 1]  # Return the indices of the two numbers
            num_dict[num] = i  # Store the current number and its index in the dictionary
        return None  # If no solution is found, return None
```