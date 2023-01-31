---
title: "python 기본 알고리즘"
excerpt: "2023-01-19 Basic python algorithms"

# layout: post
categories:
  - Python algorithm interview
tags:
  - python
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---

해당 내용은 '[파이썬 알고리즘 인터뷰](https://product.kyobobook.co.kr/detail/S000001932748)' 책의 일부를 발췌하여 정리한 내용입니다.

## Generator


```python
# yield는 제네레이터가 여기까지 실행 중이던 값을 내보낸다는(단어의 사전적 의미처럼 '양보하다')의미로 
# 중간값을 리턴한 다음 함수는 종료되지 않고 계속해서 맨 끝에 도달할 때까지 실행된다.

def get_natural_number():
    n = 0
    while True:
        n += 1
        yield n
```


```python
get_natural_number()
```




    <generator object get_natural_number at 0x000002ABE924A890>



* 함수의 리턴값이 제네레이터가 된 것을 확인할 수 있다.
* 값을 생성하려면 next()로 추출하면 된다.


```python
# 30개의 값을 생성하고 싶다면 다음과 같이 30번 동안 next()를 수행하면 된다.

g = get_natural_number()
for _ in range(0,30):
    print(next(g))
```

    1
    2
    3
    4
    5
    6
    7
    8
    9
    10
    11
    12
    13
    14
    15
    16
    17
    18
    19
    20
    21
    22
    23
    24
    25
    26
    27
    28
    29
    30
    


```python
# 여러 타입의 값을 하나의 함수에서 생성하는 것도 가능

def generator():
    yield 1
    yield 'string'
    yield True
    
g = generator()
g
```




    <generator object generator at 0x000002ABE930BA50>




```python
next(g)
```




    1




```python
next(g)
```




    'string'




```python
next(g)
```




    True



## Enumerate

인덱스와 값을 추출해준다.


```python
a = [1, 2, 3, 5, 34, 12412]

enumerate(a)
```




    <enumerate at 0x2abe92e0d40>




```python
list(enumerate(a))
```




    [(0, 1), (1, 2), (2, 3), (3, 5), (4, 34), (5, 12412)]



## Divmod

몫과 나머지를 한번에 추출해준다.


```python
divmod(5,3)
```




    (1, 2)



## print


```python
# 기본적으로 ,를 쓰면 띄어쓰기가 적용된다
print('A1', 'A2')
```

    A1 A2
    


```python
# sep 파라미터로 중간을 지정해줄 수 있다.

# sep = ,
print('A1', 'A2', sep = ',')

# sep = /
print('A1', 'A2', sep = '/')
```

    A1,A2
    A1/A2
    


```python
# end 파라미터를 통해서 줄바꿈 처리를 하지 않도록 할 수 있다.

print('AA', end = "") # end : 공백처리
print('BB')
```

    AABB
    


```python
# 리스트를 출력시 .join()으로 묶어서 처리한다

a = ['A', 'B']
print(' '.join(a))
```

    A B
    


```python
# f-string을 사용하면 변수명을 쉽게 출력해줄 수 있다.

idx = 1
fruit = 'apple'

print(f'{idx + 1} : {fruit}')
```

    2 : apple
    

## Pass

pass는 목업 인터페이스부터 구현한 다음에 추후 구현 시 중간 단계에서 예상되는 에러를 넘어가게 해줄 수 있는 기능이다.


```python
# 에러가 나는 코드

class MyClass(object):
    def method_a(self):
        
    def method_b(self):
        print("Method B")
        
c = MyClass()
```


      Cell In [20], line 6
        def method_b(self):
        ^
    IndentationError: expected an indented block
    


위 문제는 method_a가 아무런 처리를 하지 않았기 때문에 엉뚱하게 method_b()에서 발생한다. 물론, 필요한 오류이긴 하나 한참 개발 중에 오류를 맞딱드리면 생각보다 곤란하다. 이럴 경우 pass를 삽입하여 간단히 처리해준다.


```python
class MyClass(object):
    def method_a(self):
        # pass 추가
        pass
    
    def method_b(self):
        print("Method B")
        
c = MyClass()
```

python에서 pass는 Null 연산으로 아무것도 하지 않는 기능이다. 온라인 코테에서도 유용하게 활용할 수 있다.

## locals

로컬에 선언된 모든 변수를 조회할 수 있는 강력한 명령어. 디버깅에 많은 도움을 준다.


```python
import pprint
pprint.pprint(locals())
```

    {'In': ['',
            "# yield는 제네레이터가 여기까지 실행 중이던 값을 내보낸다는(단어의 사전적 의미처럼 '양보하다')의미로 \n"
            '# 중간값을 리턴한 다음 함수는 종료되지 않고 계속해서 맨 끝에 도달할 때까지 실행된다.\n'
            '\n'
            'def get_natural_number():\n'
            ...
             9: <enumerate object at 0x000002ABE92E0D40>,
             10: [(0, 1), (1, 2), (2, 3), (3, 5), (4, 34), (5, 12412)],
             11: (1, 2)},
     'a': ['A', 'B'],
     'c': <__main__.MyClass object at 0x000002ABEA5FD790>,
     'exit': <IPython.core.autocall.ZMQExitAutocall object at 0x000002ABE8D97FA0>,
     'fruit': 'apple',
     'g': <generator object generator at 0x000002ABE930BA50>,
     'generator': <function generator at 0x000002ABE92B1790>,
     'get_ipython': <bound method InteractiveShell.get_ipython of <ipykernel.zmqshell.ZMQInteractiveShell object at 0x000002ABE8E851C0>>,
     'get_natural_number': <function get_natural_number at 0x000002ABE92B1310>,
     'idx': 1,
     'open': <function open at 0x000002ABE7503DC0>,
     'os': <module 'os' from 'c:\\Users\\junhy\\anaconda3\\lib\\os.py'>,
     'pprint': <module 'pprint' from 'c:\\Users\\junhy\\anaconda3\\lib\\pprint.py'>,
     'quit': <IPython.core.autocall.ZMQExitAutocall object at 0x000002ABE8D97FA0>,
     'sys': <module 'sys' (built-in)>}
    
