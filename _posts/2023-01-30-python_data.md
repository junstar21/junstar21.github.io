---
title: "자료형"
excerpt: "2023-01-30 python data type"

# layout: post
categories:
  - Python algorithm interview
tags:
  - python
  - list
  - dictionary
  - Arbitrary-Precision
  - set
  - Sequence
  - Immutable Object
  - Mutable Object
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
해당 내용은 '[파이썬 알고리즘 인터뷰](https://product.kyobobook.co.kr/detail/S000001932748)' 책의 일부를 발췌하여 정리한 내용입니다.

## 자료형

### 숫자

파이썬에서는 숫자 정수형으로 `int`만 제공한다. `int`는 임의 정밀도를 지원한다. `int`는 `object`의 하위 클래스이기도 하며 다음과 같은 구조를 띈다.

```
object > int > bool
```

**🤔 임의 정밀도란?**

> 임의 정밀도 정수형이란 무제한 자릿수를 제공하는 정수형을 의미한다. 예를 들어 123456789101112131415라는 큰 수를 파이썬은 다음과 같이 표현한다.
> 
> 
> 
> | 사이즈 | 3 |
> | --- | --- |
> | 값 | (437976919*2^(30*0)) + (87719511*2^(30*1)) + (107*2^(30*2)) |
> 
> 임의 정밀도로 계산할 경우 속도가 저하된다는 단점이 있지만, 숫자를 단일형으로 처리할 수 있어 언어를 매우 단순한 구조로 만들 수 있다.
> 

### 집합

집합 자료형인 `set`는 중복값을 갖지 않는 자료형이다.

```python
a = set()
a

out:
set()

type(a)

out:
set

```

`{}`를 쓰기 때문에 딕셔너리와 혼동의 여지가 있으나, 선언 형태만 보면 금방 타입을 판단 할 수 있다.

```python
a = {'a', 'b', 'c'}
type(a)

out:
set

a = {'a':'A', 'b':'B', 'c':'C'}
type(a)

out:
dict
```

`set`는 입력순서가 유지 되지 않으며, 중복값이 있을 경우 하나만 유지한다.

```python
a = {3, 2, 3, 5}
a

out:
{2, 3, 5}
```

### 시퀀스(Sequence)

Sequence는 어떤 특정 대상의 순서 있는 나열을 뜻한다. 파이썬에서는 `list`라는 시퀀스 타입이 사실상 배열의 역할을 수행한다. 시퀀스는 값을 변경할 수 없는 불변과 값을 변경할 수 있는 가변으로 나뉘어 진다. 불변에는 `str`, `turple`, `bytes`가 해당된다.

```python
a = 'abc'
id('abc')

out:
2326567697776

id(a)
out:
2326567697776

a = 'def'
id('def')

out:
2326570403568

id(a)

out:
2326570403568
```

메모리 주소를 출력해보면 ‘`abc`’와 ‘`def`’는 다른 메모리에 할당된 모습을 확인할 수 있다. a변수는 처음에 ‘`abc`’를 참조하였다가 ‘`def`’로 나중에 다시 참조됬을 뿐, ‘`abc`’와 ‘`def`’는 사라지지 않고 메모리 어딘가에 남아있다.

## 원시 타입

C나 java같은 프로그래밍 언어들은 원시타입(Primitive Type)을 제공한다. 정수의 크기나 부호에 따라 매우 다양한 원시 타입을 제공한다. 원시타입을 사용하는 이유는 정수의 크기나 부호에 따라 최적의 타입을 선택하여 메모리 사용량을 줄임으로서 성능을 높히기 위함이다.

반면, 파이선은 앞선 임의 정밀도에 나온 내용처럼 성능보단 편리한 기능 제공에 우선순위를 둔 언어이다. 따라서, 원시타입을 제공하지 않는 반면 객체의 다양한 기능과 편의성을 택하였다.

## 객체

파이썬은 크게 불변 객체(Immutable Object)와 가변 객체(Mutable Object)로 나뉜다.

| 클래스 | 설명 | 불변 객체 |
| --- | --- | --- |
| bool | 부울 | O |
| int | 정수 | O |
| float | 실수 | O |
| list | 리스트 | X |
| tuple | 리스트와 튜플의 차이 : 불변 여부, 튜플은 불변 | O |
| str | 문자 | O |
| set | 중복값을 가지지 않는 집합 자료형 | X |
| dict | 딕셔너리 | X |

### 불변 객체

파이썬에서 변수를 할당하는 작 업은 해당 객체에 대해 참조를 한다는 의미이다. 여기에는 예외가 없다.

```python
10
a = 10
b = a

print(id(10), id(a), id(b))

out:
2326562171472 2326562171472 2326562171472
```

파이썬은 모든 것이 객체이므로, 메모리 상에 위치한 객체의 주소를 얻어오는 id()함수를 실행한 결과 모두 동일하다.

### 가변 객체

`list`는 값이 바뀔 수 있으며, 이 말은 다른 변수가 참조하고 있을 때 그 변수의 값 또한 변경된다는 이야기이다.

```python
a = [1, 2, 3, 4, 5]
b = a

b

out:
[1, 2, 3, 4, 5]

a[2] = 4

print(a, b)

out:
[1, 2, 4, 4, 5] [1, 2, 4, 4, 5]
```

**🔠 is 와 == 의 차이**

> `is` : `id()` 값을 비교하는 함수.
> 
> `==` : 값을 비교하는 연산자
> 
> 
> ```python
> a = [1, 2, 3]
> 
> print('a == a:', a == a, 'a == list(a):', a == list(a), 
>       'a is a:', a is a, 'a is list(a):', a is list(a), sep='\n')
> 
> out:
> a == a:
> True
> a == list(a):
> True
> a is a:
> True
> a is list(a):
> False
> ```
> 
> 값은 동일하지만 `list()`로 한 번 더 묶어주면 별도의 객체로 복사되어 다른 ID를 가지기 때문에 `is`는 `False` 처리가 된다.
>