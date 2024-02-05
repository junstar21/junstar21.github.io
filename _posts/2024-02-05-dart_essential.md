---
title: "Dart - 기본기"
excerpt: "2024-02-05 Dart - essential"

# layout: post
categories:
  - Dart
tags:
  - Dart
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
본 내용은 인프런 중 [[코드팩토리] [입문] Dart 언어 4시간만에 완전정복](https://www.inflearn.com/course/dart-%EC%96%B8%EC%96%B4-%EC%9E%85%EB%AC%B8/dashboard) 강의 내용을 정리한 내용입니다.

Dart의 Console은 모두 `void main()` 안에서 실행되는 결과를 출력해준다. 따라서 앞으로 작성한 코드들은 모두 `void main()` 코드 안에서 실행된다.

```dart
void main() {
	/// 작성되는 예시코드들
}
```

편의를 위해서 `void main()`은 생략하여 작성을 진행할 예정이며, 자세한 내용은 하단에 설명될 예정이다.

## 변수 선언

```dart
// variable: 기본 변수 선언명. 입력된 변수의 타입에 따라 자동으로 인식해준다.
var name = '코드팩토리';

// name에 할당된 내용을 변경하고 싶으면 name만 사용하면 된다.
// var name = '플러터 프로그래밍'; 으로 변경하려면 에러가 발생한다.
// 이미 var name을 선언하였기 때문이다.
name = '플러터 프로그래밍';

// integer: 정수
int number1 = 2;
int number2 = 4;

// double: 실수
double number3 = 2.5;
double number4 = 0.5;

// boolean: 맞다 or 틀리다
bool isTrue = true;
bool istFalse = false;

// String: 글자 타입
String name2 = '레드벨벳';
String name3 = '코드팩토리';

// print시 변수에 담긴 내용을 출력해주게 하기
print('$name2');
```

**🤔 `var`과 다른 변수선언명들을 구분 짓는 이유?**

var은 할당된 변수에 따라서 컴퓨터가 자동으로 타입을 지정하고 인식하지만, 사람에 관점에서는 직관적이지가 않다. 즉, 코드를 리뷰하는 사람이나 코드 작성자가 차후에 코드를 리뷰할 때 보다 더 쉬운 이해를 위해 변수 타입에 따라서 직접 변수명을 작성해준다.

## `Null`

```dart
// nullable: null이 될 수 있다.
// non-nullable: null이 될 수 없다.
// null: 아루런 값도 있지 않다.
String name2 = '블랙핑크';
name2 = null; // 해당 파트에서 에러가 발생한다. name2는 non-nullable 상태이기 때문이다.

String? name2 = '블랙핑크';
name2 = null; // String name2를 선언할 때 String 뒤에 물음표를 붙혀줌으로서 nullable 상태로 만들어준다.
```

## `final`, `const`

```dart
// final: 변수의 값을 한번 선언한 뒤로는 변경이 불가능하다.
final String name = '코드팩토리';
name = '인프런'; // 해당 파트에서 에러가 발생한다. 변수 변경이 불가능한 final을 선언하였기 때문이다.

// const: 
const String name2 = '블랙핑크';
name2 = '코드팩토리'; // 해당 파트에서 에러가 발생한다. 변수 변경이 불가능한 const을 선언하였기 때문이다.

// final과 const는 변수명 앞에 어떤 타입인지 선언하는 type을 생략해도 가능하다.
final name3 = '코드팩토리';
const name4 = '블랙핑크';
```

🤔 `final`과 `const`의 차이점은 무엇인가?

const의 경우 빌드타임의 값을 알고 있어야 한다. 빌드타임이라함은 프로그래밍 코드를 실행하였을 때, 컴퓨터가 코드 언어를 이진수로 실행할 때를 의미한다.  const로 선언한 변수가 현재 시간이고 코드를 실행할 때 현재 시간에 대한 정확한 값이 지정되지 않았기 때문에 에러가 발생한다. 반면, final의 경우 빌드타임 값이 필요하지 않기 때문에 현재 시간을 정상적으로 가져올 수 있다.

## Operator

### 기본 Operator

```dart
int number = 2;

// %: 나머지 값을 반환해준다.
print(number % 2);
>>> 0

// ++: 변수값에 1을 더해준다.
number ++;
print(number);
>>> 3

// --: 변수값에 1을 빼준다.
number --
print(number);
>>> 2 
```

### Null 조건의 Operator

```dart
double? number = 4.0;
print(number);
>>> 4

number = 2.0;
print(number);
>>> 2

number = null;
print(number);
>>> null

number ?? = 3.0 // ??: number의 값이 null일 때 저장된 값을 바꿔주기 위해 사용
print(number);
>>> 3
```

### 비교 Operator

`<`, `>`: 비교 연산자이다. 비교값에 따라서 Boolian 값을 나타낸다.

`>=`, `<=` : 이상, 이하를 나타내는 연산자이다. 

`==` : 값이 같은지를 나타내는 연산자이다. 같으면 True, 다르면 False를 나타낸다.

`!=` : 값이 틀린지를 나타내는 연산자이다. 같으면 False, 다르면 True를 나타낸다.

### Type Operator

```dart
int number1 = 1;

// number1이 int type인지를 판별할 때는 is를 사용한다.
print(number1 is int);
>>> true
print(number1 is String);
>>> False

// number1이 int type이 아닌지를 판별할 때는 is를 사용한다.
print(number1 is! int);
>>> false
print(number1 is! String);
>>> true
```

### 논리 Operator

`&&`: AND 역할을 한다. 지정한 조건이 둘다 참이 되어야 `true`를 반환한다.

`||`: OR역할을 한다. 지정한 조건 중 하나라도 참이 되면 `true`를 반환한다.

```dart
bool result = 12 > 10 && 1 > 0;
print(result);
>>> true

bool result2 = 12 > 10 && 0 > 1;
print(result2);
>>> false

bool result3 = 12 > 10 || 1 > 0;
print(result3);
>>> true

bool result4 = 12 > 10 || 0 > 1;
print(result4);
>>> true
```

## List

말 그대로 리스트를 만든다. 파이썬과 다르게 `<>`(제너릭)안에 어떤 타입의 변수를 넣을지를 지정해준다.

```dart
List<String> blackPink = ['제니', '지수', '로제', '리사']; //String Type만 리스트에 포함될 수 있다.
List<int> numbers = [1, 2, 3, 4, 5, 6];
```

### Index

리스트에서 어떤 값을 뽑아낼 때 인덱스(순서)를 활용한다. 인덱스의 순서는 항상 0부터 시작한다.

```dart
print(blackPink[0]);
>>> 제니
```

### List 함수

```dart
/// List의 길이
print(blackPink.length);
>>> 4

// List에 값을 추가
blackPink.add('코드팩토리');
print(blackPink);
>>> ['제니', '지수', '로제', '리사', '코드팩토리']

// List에 값을 제거
blackPink.remove('코드팩토리');
print(blackPink);
>>> ['제니', '지수', '로제', '리사']

// 리스트 내 값이 몇번 째 인덱스인지 확인
print(blackPink.indexOf('로제'));
>>> 2
```

## Map

파이썬의 딕셔너리와 비슷한 개념이라고 볼 수 있다. Key와 Value 값을 가지고 있다.

```dart
// 해리포터의 영문과 한글명
Map<String, String> dictionary {
	'Harry Potter' : '해리포터',
	'Ron Weasley' : '론 위즐리',
	'Hermione Granger' : '헤르미온느 그레인저',
};

print(dictionary);
>>> {'Harry Potter' : '해리포터', 'Ron Weasley' : '론 위즐리',	'Hermione Granger' : '헤르미온느 그레인저'}

// 해당 map이 해리포터 등장인물인지 참/거짓을 판별
Map<String, bool> isHarryPoter = {
	'Harry Potter' : true,
	'Ron Weasley' : true,
	'Ironman' : false,
};

// isHarryPoter에 추가하기
isHarryPotter.addAll({
	'Spiderman' : false,
});

// isHarryPoter에 제거하기
isHarryPotter.remove('Harry Poter')

// 키값 또는 벨류값만 출력하기
print(isHarryPotter.keys);
>>>	('Ron Weasley', 'Ironman' ,'Spiderman')

print(isHarryPotter.values);
>>> (true, false, false)
```

## Set

List와 비슷하나 중복값을 제거해주는 기능을 가졌다. 파이썬 Set과 동일한 기능이다.

```dart
final Set<String> names = {
	'Code Factory',
	'Flutter',
	'Black Pink',
	'Flutter'
};

print(names);
>>> {Code Factory, Flutter, Black Pink}
```

## If 문

조건문 if다. 해당 조건이 참이면 코드를 실행해주는 기능을 가지고 있다.

```dart
if(//조건 부여){
	//조건이 참이면 실행되는 코드
}else if(// 다른 조건 부여){
	//다른 조건이면 실행되는 코드
} else {
	// 나머지 조건이면 실행되는 코드
}
```

## Switch 문

if문과 흡사한 기능을 한다.

```dart
int number = 1;

switch(number % 3){
	case 0: // 나머지가 0일 때 실행된다.
		print('나머지가 0입니다.');
		break; // switch문은 조건이 실행되었을 때 끝에 항상 break를 넣어주어야한다.
	case 1:
		print('나머지가 1입니다.');
		break;
	default: // 처음 switch문 조건에 부합할 때 실행된다.
		print('나머지가 2입니다.);
		break;
}
```

## loop문

파이썬에서 for i in ~~과 동일한 기능을 한다.

```dart
for (int i = 0; // 초기조건
		 i < 10; // 충족 조건
		 i ++ // 충족될 때까지 실행){
	print(i);
}

```

## while loop 문

for문과 다르게 초기 조건 지정이 필요없다.

```dart
int total = 0

while(total < 10){
	total += 1;
}
```

## enum

```dart
// enum은 void 밖에서 선언한다.
enum Status{
	approved, //승인
	pending, //대기
	rejected, //거절
}

void main(){
	Status status = Status.pending;
	
	if(status == Status.approved){
		print('승인입니다.');
	}else if(status == Status.pending){
		print('대기입니다.');
	}else{
		print('거절입니다.');
	}
}
```

enum을 선언하는 이유는 정확히 enum안에 있는 값만 확실하게 알기 위해서 선언하는 것이다. 즉, 코드 리뷰어와 미래에 코드를 다시 확인할 코드 작성자의 편의(오탈자 방지 등)를 위해서 사용한다.

## 함수

### 함수선언

```dart
void main() {
  addNumbers();
}

// 세개의 숫자 (x, y, z)를 더하고 짝수인지 홀수인지 알려주는 함수
addNumbers() {
  int x = 10;
  int y = 20;
  int z = 30;
  
  int sum = x + y + z;
    
  print('x : $x');
  print('y : $y');
  print('z : $z');
  
  if (sum % 2 == 0){
    print('짝수입니다.');
  } else{
    print('홀수입니다.');
  }
}
```

### 매개변수(parameter / argument)

외부에서 변수를 받아서 실행하는 함수를 선언한다.

```dart
void main() {
  addNumbers(10, 20, 30); // x, y, z값을 입력
}

//입력 받은 수를 더하고 짝수인지 홀수인지 알려주는 함수
addNumbers(int x, int y, int z) {
  int sum = x + y + z;
    
  print('x : $x');
  print('y : $y');
  print('z : $z');
  
  if (sum % 2 == 0){
    print('짝수입니다.');
  } else{
    print('홀수입니다.');
  }
}
```

- optional parameter : 선택적으로 존재할 수 있는 파라미터

```dart
void main() {
  addNumbers(10);
}

// int y 와 in z는 []에 둘러쌓여있다.
// [] 안에 들어간 변수는 optional parameter로 지정되어
// 입력이 안되면 기본으로 저장된 y = 20, z = 30이 들어가게 된다.
addNumbers(int x, [int y = 20, int z = 30]) {
  int sum = x + y + z;
    
  print('x : $x');
  print('y : $y');
  print('z : $z');
  
  if (sum % 2 == 0){
    print('짝수입니다.');
  } else{
    print('홀수입니다.');
  }
}
```

- named parameter : 이름이 있는 파라미터. 순서의 영향을 받지 않는다.

```dart
void main() {
  addNumbers(x: 10, y: 20, z:30);
}

// 세개의 숫자 (x, y, z)를 더하고 짝수인지 홀수인지 알려주는 함수
addNumbers({
  required int x,
  required int y,
  required int z,
}) {
  int sum = x + y + z;
    
  print('x : $x');
  print('y : $y');
  print('z : $z');
  
  if (sum % 2 == 0){
    print('짝수입니다.');
  } else{
    print('홀수입니다.');
  }
}
```

named parameter에서 optional parameter를 사용하고 싶다면 required를 제거하고 기본값을 넣어주면 가능하다.

```dart
void main() {
  addNumbers(x: 10, y: 20);
}

// 세개의 숫자 (x, y, z)를 더하고 짝수인지 홀수인지 알려주는 함수
addNumbers({
  required int x,
  required int y,
  int z = 30,
}) {
  int sum = x + y + z;
    
  print('x : $x');
  print('y : $y');
  print('z : $z');
  
  if (sum % 2 == 0){
    print('짝수입니다.');
  } else{
    print('홀수입니다.');
  }
}
```

### `void main(){}`

Dart는 `void main(){}`(이하 void)에서 실행된 내용이 console에 나타난다. 따라서 우리가 외부에서 지정한 함수를 이용하여 void에서 사용할 수 있다.

```dart
void main() {
  int result = addNumbers(10, y: 20);
  int result2 = addNumbers(10, y: 30, z: 40);
  
  print('result: $result');
  print('result2: $result2');
  
  print('sum : ${result + result2}');
}

int addNumbers(int x, {
  required int y,
  int z = 30,
}) => x + y + z; // => : 함수에서 반환하는 값 지정. return과 같은 기능을 한다
```

## Type Def

함수를 편리하게 사용할 수 있는 기능 중 하나.

```dart
void main() {
  // operation 함수의 기능 지정하기: +
  Operation operation = add;

  int result = operation(10, 20, 30);
  print(result);
  
  // operation 함수 기능 변경: -
  operation = subtract;
  int result2 = operation(10, 20, 30);
  print(result2);
}

// signature 지정
typedef Operation = int Function(int x, int y, int z);

// +
int add(int x, int y, int z) => x + y + z;

// -
int subtract(int x, int y, int z) => x - y - z;
```

```
// Console 결과 창
60
-40
```

위 코드는 이해를 돕기 위한 코드이다. 실제에서는 아래와 같이 작성한다.

```dart
void main() {
  int result = calculate(10, 20, 30, add);
  print(result);
  
  int result2 = calculate(40, 50, 60, subtract);
  print(result2);
}

// signature
typedef Operation = int Function(int x, int y, int z);

// +
int add(int x, int y, int z) => x + y + z;

// -
int subtract(int x, int y, int z) => x - y - z;

// 계산
int calculate(int x, int y, int z, Operation operation){
  return operation(x, y, z);
}
```