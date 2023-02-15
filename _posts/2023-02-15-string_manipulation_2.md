---
title: "문자열 조작 - 2"
excerpt: "2023-02-15 String manipulation - 2"

# layout: post
categories:
  - Python algorithm interview
tags:
  - python
  - string
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
해당 내용은 '[파이썬 알고리즘 인터뷰](https://product.kyobobook.co.kr/detail/S000001932748)' 책의 일부를 발췌하여 정리한 내용입니다.

## [그룹 애너그램](https://leetcode.com/problems/group-anagrams/)

### 풀이

```python
class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        # 존재하지 않는 키를 삽입할 경우 발생하는 keyError가 나지 않도록 디폴트 생성
        anagrams = collections.defaultdict(list)

        for word in strs:
            # 정렬하여 딕셔너리에 추가
            anagrams[''.join(sorted(word))].append(word)
        return list(anagrams.values())
```

### 여러 가지 정렬 방법

파이썬은 팀소트(Timsort)를 기반으로 실제데이터를 빠르게 정렬할 수 있다.

`sorted()` : 파이썬 리스트를 정렬하는 함수

```python
a = [2, 5, 1, 9, 7]
sorted(a)

out:
[1, 2, 5, 7, 9]
```

`sorted()`는 숫자뿐만 아니라 문자도 정렬이 가능하다.

```python
b = 'zbdaf'
sorted(b)

out:
['a', 'b', 'd', 'f', 'z']
```

정렬한 리스트 ['a', 'b', 'd', 'f', 'z']를 다시 결합하려면 `join()`을 이용할 수 있다.

```python
b = 'zbdaf'
"".join(sorted(b))

out:
'abdfz'
```

`sorted()`에 또한 `key=` 옵션을 지정해 정렬을 위한 키 또는 함수를 별도로 지정해줄 수 있다.

```python
c = ['ccc', 'aaaa', 'd', 'bb']
# key= len : 길이 순서로 정렬하기
sorted(c, key = len)

out:
['d', 'bb', 'ccc', 'aaaa']
```

함수를 이용해 키를 정의하는 방법을 더 살펴보자.

```python
a = ['cde', 'cfc', 'abc']

def fn(s):
	return s[0], s[-1]

sorted(a, key= fn)

out:
['abc', 'cfc', 'cde']
```

일반적인 `sorted()`를 사용했다면 알파벳 순서에 따라 [’abc’, ‘cde’, ‘cfc’] 순으로 출력되었겠지만, 여기서는 두번째 키로 마지막 문자열을 보게 했기 때문에 ['abc', 'cfc', 'cde'] 순으로 출력된다.

람다를 이용하면 함수를 정의하지 않고도 한줄로 바로 처리할 수 있다.

```python
sorted(a, key = lambda s : (s[0], s[-1]))
```

## [가장 긴 팰린드롬 부분 문자열](https://leetcode.com/problems/longest-palindromic-substring/)

### 풀이

```python
class Solution:
    def longestPalindrome(self, s: str) -> str:
        # 팰린드롬 판별 및 투포인터 확정
        def expand(left: int, right: int) -> str:
            while left >= 0 and right < len(s) and s[left] == s[right]:
                left -=1
                right += 1
            return s[left + 1: right]

        # 해당 사항이 없을 때 빠르게 리턴
        if len(s) < 2 or s == s[::-1]:
            return s

        result = ''
        # 슬라이드 윈도우 우측으로 이동
        for i in range(len(s) - 1):
            result = max(result,
                            expand(i, i + 1),
                            expand(i, i + 2),
                            key= len)

        return result
```