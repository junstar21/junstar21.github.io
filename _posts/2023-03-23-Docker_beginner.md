---
title: "Docker 입구 수업"
excerpt: "2023-03-23 Docker for beginners"

# layout: post
categories:
  - TIL
tags:
  - Docker
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---

해당 내용은 [생활코딩 Docker 입구 수업 유튜브](https://www.youtube.com/playlist?list=PLuHgQVnccGMDeMJsGq2O-55Ymtx0IdKWf) 내용의 일부를 발췌 및 정리한 내용입니다. 

# 도커(Docker)란?

리눅스 컨테이너스라는 커널 컨테이너 기술을 이용하여 만든 컨테이너 기술 중 하나. 

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-03-23-Docker_beginner/Untitled.png)

운영 체제 위에 가상환경을 만들어 독립적인 환경(컨테이너)에서 돌아가는 시스템. 컨테이너를 만들고 그 안에서 환경을 구성하기에 다른 외부 컨테이너에 영향을 끼치지 않아 프로젝트 관리 및 배포와 서버 운용 등에 매우 유용한 기술이다. 

# 도커 설치

[도커 공식 홈페이지 설치페이지](https://docs.docker.com/get-docker/)에 들어가서 본인의 운영체제에 맞는 도커를 설치해주면 된다. 앞선 소개에서 도커는 리눅스 운영체제 기반의 기술이기에 맥과 윈도우 운영체제에서는 리눅스보다 속도가 떨어지는 단점이 있지만, 리눅스를 설치하고 도커를 설치하는 것보다 속도와 편의성에 있어 좋기 때문에 도커 데스크탑을 설치하는 것을 권장한다.

# 도커의 구성 요소

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-03-23-Docker_beginner/Untitled 1.png)

- Docker hub : 앱스토어와 비슷한 역할로, 도커의 서버 상에 수많은 이미지들이 저장되어 있다. 원하는 이미지를 검색하고 이를 pull(다운로드)하여 사용할 수 있다.
- 이미지(image) : 일종의 설치파일 같은 개념이라 볼 수 있다. 어떠한 환경으로 컨테이너를 조성할 것인지에 대한 정보들이 담겨져있다. 이미지가 변경되지 않으면, 한개의 이미지로 수백만의 컨테이너를 만들어도 컨테이너의 내용은 이미지가 설정해놓은대로 생성되기 때문에 동일한 컨테이너를 생성할 수 있다.
- 컨테이너(container) : 이미지가 설정해놓은 프로파일에 따라서 생성되는 독립적인 환경이다. 앞서 한개의 이미지로 여러 개의 컨테이너를 생성할 수 있지만, 컨테이너는 다른 컨테이너에 영향을 줄 수 없다. 생성된 컨테이너에 목적에 맞게 환경을 조성해줄 수 있다.

# 도커 실행 방법

도커는 기본적으로 명령어 인터페이스로 구성되어 있다. GUI 환경에서도 사용이 가능하지만, 명령어를 알아두는 것이 유용하다.

- Windows : Docker Desktop을 실행하면 GUI 환경에서 도커를 제어할 수 있다. 명령어로 제어하기 위해서는 CMD창을 실행시켜서 `Docker` 와 관련된 명령어들을 입력하여 실행할 수 있다.
- Mac : 윈도우와 마찬가지로 Docker Desktop을 이용하여 GUI 환경에서 사용이 가능하며, Terminal 프로그램을 실행시켜 관련 명령어들을 입력하면 된다.

# 도커 명령어

도커 명령어는 [공식홈페이지의 Reference](https://docs.docker.com/reference/) → Command-line reference → Docker CLI를 들어가면 어떤 명령어들이 있는지, 그리고 그 기능에 대해서 설명을 해준다. 여기서는 기본적이고 많이 쓰이는 코드들에 대해 설명하도록 한다.

## [이미지 `pull`](https://docs.docker.com/engine/reference/commandline/pull/)

예를 들어 `httpd`라는 이미지를 pull하도록 하자. 

```docker
docker pull httpd
```

해당 작업이 잘 수행되었는지 확인하기 위해 다음과 같은 명령어를 입력한다.

```docker
docker images
```

해당 명령어는 컴퓨터에 어떤 이미지들이 설치되어있는지를 확인하는 명령어이다. 명령어를 입력하면 아래와 같은 출력이 나오게 된다.

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-03-23-Docker_beginner/Untitled 2.png)

가장 마지막 하단의 REPOSITORY를 확인하면 `httpd`가 설치된 것을 확인할 수 있다.

## 컨테이너

### [`run`](https://docs.docker.com/engine/reference/commandline/run/)

설치한 이미지를 토대로 컨테이너를 생성해보자.

```docker
docker run httpd
```

정상적으로 컨테이너가 생성되었는지 확인하기 위해 아래와 같은 명령어를 입력해준다.

```docker
docker ps
```

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-03-23-Docker_beginner/Untitled 3.png)

이러면 컨테이너의 정보들이 나타나게 된다. 이번에는 새로운 컨테이너를 이름을 붙혀 생성해보도록 하자.

```docker
docker run --name ws2 httpd
```

`—name`은 run 명령어의 옵션을 사용하여 컨테이너에 직접 이름을 붙히는 명령어이다. `—name ws2`는 컨테이너의 이름을 ws2로 지정하겠다는 뜻이다. 다시 컨테이너 확인을 해보기 위해 `docker ps`를 입력해보도록 하자.

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-03-23-Docker_beginner/Untitled 4.png)

### [`stop`](https://docs.docker.com/engine/reference/commandline/stop/)

실행중인 ws2 컨테이너를 종료해보자.

```docker
docker stop ws2
```

종료 후, `docker ps`를 통해 상태창을 확인하면 목록에 ws2 컨테이너는 포함되지 않는다. 다만, 이때는 삭제가 아닌 종료가 되었기 때문에 현재 실행중인 컨테이너 목록에는 실리지 않는다. 생성된 모든 컨테이너를 확인하기 위해서는 `docker ps -a`를 입력하면 확인이 가능하다.

### [`start`](https://docs.docker.com/engine/reference/commandline/start/)

중지한 ws2 컨테이너를 다시 실행시키고 싶다면 `docker start ws2`를 입력하여 중지된 컨테이너를 다시 실행시켜줄 수 있다.

### [`rm`](https://docs.docker.com/engine/reference/commandline/rm/)

ws2 컨테이너를 삭제하고 싶다면 `docker rm ws2`를 입력하여 컨테이너를 삭제할 수 있다. 하지만, 현 상태에서는 ws2 컨테이너를 삭제할 수 없다. 컨테이너가 실행중이기 때문이다. 따라서, 컨테이너를 중지(`stop`)하고 삭제하거나 강제로 삭제를 해줄 수 있다. `docker rm —force ws2`를 입력하면 한번에 삭제를 할 수 있다.

## [이미지 삭제 `docker rmi`](https://docs.docker.com/engine/reference/commandline/rmi/)

pull한 httpd 이미지를 삭제해보자.

```docker
docker rmi httpd
```

해당 명령어를 수행하고 `docker images`를 입력하면 목록에 httpd가 사라진 것을 확인할 수 있다.

# 도커 네트워크

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-03-23-Docker_beginner/Untitled 5.png)

많은 소프트웨어들은 네트워크를 통해 동작한다. 도커도 마찬가지이다. 컨테이너가 설치된 도커 운영체제를 Host라고 부른다. 웹페이지를 파일로 만들어 특정 디렉토리에 저장하였고 이를 외부 웹브라우저를 통해서 접근을 하려고 한다. 외부 웹에서 도커 내 컨테이너 내 파일 접근 시 2개의 포트가 존재한다. Host와 컨테이너. 이 2개의 포트가 맞아야 Web server가 정상적으로 요청한 파일의 디렉토리를 안내해준다. 위 그림은 이 설명을 함축적으로 설명해주는 내용이다. 이제 프롬프트에서 네트워크 조성과 접속을 해보도록 하자.

우선, httpd 이미지가 필요하다. 이미지가 설치되어있으면 해당 과정을 생략해도 되며, 설치가 되지 않았다면 이미지 pull 파트 참고하여 설치하도록 한다. 이후 아래 명령어 실행한다.

```docker
docker run --name ws3 -p 80:80 httpd
```

-p는 컨테이너 포트를 호스트로 공개, 연결한다는 의미이다. 해당 명령어는 이름이 ws3고 연결 포트는 80:80으로 하는 컨테이너를 생성하는 명령어다. 컨테이너가 정상적으로 생성되고 연결되었는지 확인하기 위해 웹브라우저로 접속을 시도해보자. [localhost:80/index.html](http://localhost:80/index.html)을 주소창에 입력해보자. 정상적으로 브라우저에 **It works!**라는 텍스트를 화면에서 볼 수 있다.

# 명령어 실행

우리가 네트워크를 연결하는데 성공하였어도 화면에 It works!만 계속 뜬다면 연결하는 의미가 없다. 따라서 컨테이너 내부 환경을 조성하는 방법에 대해 알아보도록 한다.

### [`exec`](https://docs.docker.com/engine/reference/commandline/exec/)

exec는 선택한 컨테이너 내부에 들어갈 수 있도록 하는 명령어이다.

```docker
docker exec ws3 pwd
docker exec ws3 ls
```

해당 명령어를 작성하면 ws3 컨테이너 내부에 들어가 pwd, ls 명령어를 실행한다. 다만, 이렇게 사용하면 해당 명령어를 실행하고 다시 컨테이너 외부로 프롬프트가 나오게 된다. 컨테이너 내부에서 계속 머무르게 하기 위해서는 다음과 같은 명령어를 실행하면 된다.

```docker
docker exec -it ws3 /bin/bash
```

컨테이너의 종류에 따라서는 해당 명령어가 실행되지 않을 수 있다. 그 때는 다음 명령어를 실행시켜 주도록 하자.

```docker
docker exec -it ws3 /bin/sh
```

이러면 프롬프트가 컨테이너 내부에서 작동하게 된다. 컨테이너 내부에서 작업을 마치고 나올 때는 `exit`을 입력하면 밖으로 나올 수 있다.

```docker
docker exec -it ws3 /bin/bash
```

# 호스트와 컨테이너의 파일 시스템 연결

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-03-23-Docker_beginner/Untitled 6.png)

지속적으로 컨테이너에 들어가 내부 환경을 수정하는 것은 그리 바람직하지 않다. 컨테이너가 삭제되거나 하면 곤란한 상황이 일어나기 때문이다. 따라서, 이번에는 호스트에서 파일수정을 진행하고 실행은 컨테이너에 맡기는 방법을 알아보도록 하자.

바탕화면에 ‘htdocs’라는 폴더를 만들고 그 안에 index.html이라는 파일을 생성한다. 그리고 html파일을 다음과 같이 작성한다.

```html
<html>
    <body>
        Hello, Docker!
    </body>
</html>
```

이제 컨테이너의 파일 시스템과 호스트의 파일 시스템을 연결해도록 한다. 터미널에서 다음과 같이 입력한다.

```docker
docker run -p 8888:80 -v C:\Users\사용자명\Desktop\htdocs\:/usr/local/apache2/htdocs/ httpd
```

참고로 `C:\Users\사용자명\Desktop\htdocs\`이 부분은 html파일이 위치한 폴더를 의미한다. 운영체제마다 입력 방법이 다르니 주의하도록 한다. 연결이 잘 되었는지 확인하기 위해 브라우저 주소창에 [localhost:8888/index.html](http://localhost:8888/index.html)을 입력해보자. 그러면 ‘Hello, Docker!’라는 글자가 뜬 것을 확인할 수 있다.p