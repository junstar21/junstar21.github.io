---
title:  "혼란한 Matplotlib에서 질서 찾기"
excerpt: "2022-11-10 Matplotlib의 객체 지향 방식 코딩과 plot에 대해 알아보자."

categories:
  - Study
tags:
  - python
  - seaborn
  - Matplotlib
  - plot
# layout: post
# title: Your Title Here
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
{% include spotifyplaylist.html id=page.spotifyplaylist %}


**해당 글은 PyCon Korea 2022에서 [혼란한 Matplotlib에서 질서 찾기](https://youtu.be/ZTRKojTLE8M)라는 주제로 발표한 이제현님의 영상을 정리한 내용입니다.**

# 큰 틀의 process

Seaborn(시각화 환경설정) → Matplotlib(화면구성) → 데이터 얹기(NetworkX, sklearn, seaborn, geopandas등) → Matplotlib(부가요소 설정 → 중요 데이터 강조 → 보조 요소 설정)

# 문제 1. 안 예쁜 Matplotlib

## 해결 1. seaborn 사전설정

글자가 눈에 잘 들어오도록 설정

```python
sns.set_context("talk") # <- 구성 요소 배율 설정. (fron, line, marker 등)
sns.set_palette("Set2") # <- 배색 설정
sns.set_style("whitegrid") # <- 눈금, 배경, 격자 설정

plt.scatter(x, y, alpha = 0.5) # <- alpha : 투명도를 설정해줄 수 있다.
```

- `talk` 옵션은 발표하기 좋은 크기로 키울 수 있다.
- `set_palette` 기능을 통해서 색을 변경해줄 수 있다.
- `"whitegrid"` : 뒤에 격자를 깔고 눈금을 없애는 디자인
- 위 코드는 **맨 처음에 한번만** 실행해준다.

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-10-혼란한 Matplotlib에서 질서 찾기/Untitled.png)


# 문제 2. 시각화 모범 사례 재현

## 해결 2. 객체 지향 방식

### 🤔 **상태 기반 방식(state-based framework)**

- 간단하고 빠르게 형상만 확인하기에 유리
    - 따라서 강의에서 많이 사용하는 방식
- 그림을 그리는 순서에 맞게 코딩을 진행
- 공간 제어를 코드 순서에 맞게 제어해야됨
- 작업 과정에서 오류가 생기면 순서가 맞는 부분에 다시 가서 코드를 작성하거나 수정을 해야하는 번거로움이 있음
- 코딩을 체계를 갖추기가 어려움.

### 🤔 객체 지향 방식(object-oriented framework)

```python
fig, axs = plt.subplots(ncols = 2, figsize = (8,4),    # 레이아웃 사전 설정
                        gridspec_kw = {"wspace" : 0.1},
                        constrained_layout = True)

axs[0].plot(x, power, marker = "o", ms = 10, label = "power")   # 대상 지정 시각화
axs[1].plot(x, torque, marker = "o", ls = ":", label = "torque")

for ax in axs:
    ax.set_xlabel("time")   #  for loop 반복
    ax.legend()

axs[0].set_ylabel("output")  # axs[0] 하위 객체 추가

fig.suptitle("performance")  # fig 하위 객체 추가
```

- 상태 기반 방식에 비하여 코드가 훨신 짜임새 있고 줄어드는 효과를 볼 수 있다.
- Matplotlib 생태계 활용 가능
    - Matplotlib 호환 라이브러리로 작성
    - 객체 제어 : 강조 등 분석가 의도 반영
- **결과물의 일부를 수정하기 유리한 방식**

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-10-혼란한 Matplotlib에서 질서 찾기/Untitled 1.png)

### 객체 유형별 속성

- Artist 객체 : 선, 면, 문자 등 여럿이 존재
    - 선 : 색`(c)`, 굵기(line width)`(lw)`, 라인 스타일`(ls)`, 불투명도`(alpha)` 등
    - 면 : 면(face color)`(fc)`, 윤곽선(edge color)`(ec)` 등
- 객체 속성 추출 : `객체.get_속성()`
- 개체 속성 제어 : `객체.set_속성()`

```python
ax.collections[0].set_fc("cornflowerblue") # ax.collections[0]의 면 색상을 cornflowerblue로 변경
ax.collections[2].set_sizes([100]) # ax.collections[1]의 사이즈를 100으로 키운다
ax.lines[0].set_c("#00FF00") # 0번째의 선 색상을 바꿔라
ax.lines[1].set_lw("12") # 1번째의 굵기를 변경
```

- 일부 표현을 강조하는데 객체 지향 방식이 매우 유리함.

## 시각화 모범 사례 재현

기본 plot 상태는 아래와 같다.

```python
fig, ax = plt.subplots()
sns.violinplot(x = "species", y = "body_mass_g", data = df_peng, hue = "sex",
               split = True, ax = ax)
ax.set(xlabel = "", ylabel = "",
      title = "Body mass of penguins (g)")
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-10-혼란한 Matplotlib에서 질서 찾기/Untitled 2.png)

### 부가 요소 설정

데이터 잉크레이션 : 데이터를 칠하는데 들어가는 잉크와 부가적인 부분을 칠하는데 사용되는 잉크의 비율. 데이터를 제외한 부가요소는 최대한 줄이는 것이 좋다. 이제 필요없는 부분들을 최대한 줄여나가는 작업을 하도록 한다.

```python
ax.set_ylim(ymin = 2000) # y축 범위를 조정해서 범례를 옮길 자리를 만들어준다

ax.spines[["left", "top", "right"]].set_visible(False) # 왼쪽, 윗쪽, 오른쪽 테두리를 삭제
ax.tick_params(axis = "y", lenght = 0) # y축 왼쪽에 나와있던 선의 길이를 0으로 설정
ax.grid(axis = "y", c = "lightgray") # y축에서 그리드를 생성

```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-10-혼란한 Matplotlib에서 질서 찾기/Untitled 3.png)

초기에 비해 보다 더 시각화가 나아진 모습을 확인할 수 있다. 이제 `“Gentoo”` 를 강조하기 위한 작업을 진행하도록 한다.

### 중요 데이터 강조

```python
# Adelie와 Chinstrap의 데이터 윤곽선을 바꿔주도록 한다.

for i, obj in enumerate(ax.collections):
    ec = "gray" if i < 6 else "k"
    lw = 0.5 if i <6 else 1
    obj.set_ec(ec)
    obj.set_lw(lw) # violin plot edge width & color 다르게 적용
		if (i+3)%3 == 0:       # 모든 Median marker크게
			obj.set_sizes([60])
		if i <6:               # 비 중요 데이터는 흐리게 만들어주기
			obj.set_fc(set_hls(obj.get_fc(), ds = -0.3, dl = 0.2))

# Gentoo box plot line을 짙게 만들어준다.
for i, line in enumerate(ax.lines):
	if > 3:
		line.set_color("k") 

# 범례를 우측 하단으로 옮겨주기
handles = ax.collections[-3:-1] #Legend 새로 만들 준비
labels = ["Male", "Female"]
ax.legend(handles, labels, fontsize = 14, 
            title = "sex", title_fontsize = 14,        # Legend 새로 생성(위치, font 등 조정)
            edgecolor = "lightgray", loc = "lower right")
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2022-11-10-혼란한 Matplotlib에서 질서 찾기/Untitled 4.png)

중요 데이터만 색으로 강조된 모습을 확인할 수 있다.

### 보조 도형 활용

- 도형 객체 삽입 : `Axes.add_artist()`, etc
    - 데이터 의미 설명, 데이터 간 관계 표현
    - plot으로 부족한 표현력을 보완