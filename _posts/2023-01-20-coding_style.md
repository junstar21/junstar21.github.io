---
title: "python 코딩 스타일"
excerpt: "2023-01-20 Coding style"

# layout: post
categories:
  - Study
tags:
  - python
  - Coding style
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
해당 내용은 '[파이썬 알고리즘 인터뷰](https://product.kyobobook.co.kr/detail/S000001932748)' 책의 일부를 발췌하여 정리한 내용입니다.

## 코딩 스타일

**좋은 코드에 정답은 없다.** 하지만, 많은 사람들이 선호하는 방식은 있다.

- [PEP 8](https://peps.python.org/pep-0008/)
- [구글의 파이선 스타일 가이드](https://google.github.io/styleguide/pyguide.html)

두 가이드가 좀 더 실용적인 관점에서 좋은 파이선 코드를 작성하는데 많은 도움이 된다. 좋은 코드란 모두가 이해할 수 있을 때 더 높은 가치를 발휘한다. 또한, 파이썬이 지향하는 방식인 스네이크 케이스(변수명 사이에 ‘_’ 활용 ex: python_snake)를 사용하도록 한다(실제, 연구 결과 스네이크 방식이 인지하기 더 쉽다는 결과가 있다).

## 변수명과 주석

```python
# Number 1
def numMatchingSubseq(self, S: str, words: List[str]) -> int:
	a = 0

	for b in words:
		c= 0
		for i in range(len(b)):
			d = S[C:].find(b[i])
			if d <0 :
				a -= 1
				break
			else:
				c += d + 1
		a += 1
	
	return a
```

```python
# Number 2

def numMatchingSubseq(self, S: str, words: List[str]) -> int:
	matched_count = 0

	for word in words:
		pos= 0
		for i in range(len(word)):
		# Find matching position for each character.
			found_pos = S[pos:].find(word[i])
			if found_pos <0 :
				matched_count -= 1
				break
			else: # If found, take step position forward.
				pos += found_pos + 1
		matched_count += 1
	
	return matched_count
```

1번과 2번은 같은 내용의 코드이다. 1번은 임의의 변수를 지정한 것이고 2번은 PEP 8문서 기준에 따라서 변수명을 작성해주고 주석을 달아놓은 코드이다. 확연히 2번 코드가 눈으로 보고 이해하기 더 쉬운 코드임을 한눈에 알 수 있다. 저자의 경우, 코드에 주석을 상세히 달아놓는 것을 선호한다고 언급하였다.

## 리스트 컴프리헨션

리스트 컴프리헨션은 파이썬의 매우 강력한 기능 중 하나이지만, 지나치게 남발할 경우 가독성을 떨어뜨릴 수 있다.

```python
str1s = [str[i:i + 2].lower() for i in range(len(str1) - 1) if re.findall('[a-z]{2}', str1[i:i + 2].lower())]
```

줄이 너무 길어서 벌써부터 피로도가 올라간다. 줄바꿈을 통해 아래와 같이 간결하게 코드를 정리해보자.

```python
str1s = [
		str[i:i + 2].lower() for i in range(len(str1) - 1)
		if re.findall('[a-z]{2}', str1[i:i + 2].lower())
]
```

첫 번째 코드보다는 보기는 좋아졌지만 그래도 풀어쓰는 것보단 가독성이 떨어져보인다. 가독성을 위해 아예 풀어서 작성을 해보자.

```python
str1s = []
for i in range(len(str1) - 1):
	if re.findall('[a-z]{2}', str1[i:i + 2].lower()):
			str1s.append(str1[i:i + 2].lower())
```

경우에 따라서 가독성을 위해 모두 풀어쓰는 것도 검토해볼만 하다. 리스트 컴프리헨션은 **대체로 표현식이 2개를 넘지 않아야 한다**. 다만, 아래와 같이 작성하면 가독성이 지나치게 떨어지므로 주의가 필요하다.

```python
return [(x, y, z)
				for x in range(5)
				for y in range(5)
				if x != y
				for z in range(5)
				if y != z]
```

## 구글 파이썬 스타일 가이드

- 함수의 기본값으로 가변 객체(Mutable Object) 가 아닌 **불변 객체(Immutable Object)**를 사용한다. `None`을 명시적으로 할당하는 것도 좋은 방법이다.
    
    ```python
    # Not a good function
    
    def foo(a, b = []):
    ...
    def foo(a, b: Mapping = {}):
    
    # Good function
    
    def foo(a, b = None):
    		if be is None:
    			b = []
    ...
    def foo(a, b: Optional[Sequence] = None):
    		if b is None:
    				b = []
    ```
    
- True, False를 판별할 때는 암시적(Implict)인 방법을 사용하는 편이 간결하고 가독성이 좋다.
    
    ```python
    # Good example
    
    if not users:
    		print('no users')
    
    if foo == 0:
    		self.handle_zero()
    
    if i % 10 == 0:
    		self.handle_multiple_of_ten()
    
    # Bad example
    
    if len(users) == 0:
    		print('no users')
    
    if foo is not None and not foo:
    		self.handle_zero()
    
    if not i % 10:
    		self.handle_multiple_of_ten()
    ```
    
- 최대 줄 길이는 80자로 한다. 또한, 줄의 가로길이는 길어서 안된다는 암묵적인 약속이 있다.