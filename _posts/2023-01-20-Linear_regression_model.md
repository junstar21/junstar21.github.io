---
title: "Linear regression model"
excerpt: "2023-01-18 Linear regression model"

# layout: post
categories:
  - TIL
tags:
  - python
  - Machine learning
  - Supervised learning
  - Regression
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---

**⚠️ 해당 내용은 Coursera의 Supervised Machine Learning: Regression and Classification 강의 내용을 정리한 내용입니다.**

# **Linear regression model**

- Superviesed learning - Regression model
    
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-01-20-Linear_regression_model/Untitled.png)
    

## Terminology

- Training set : Data used to train the model
    
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-01-20-Linear_regression_model/Untitled 1.png)
    
    - Notation:
        - x = ‘input’ variable / feature
        - y = ‘output’ variable / ‘target’ vareable
        - m = number of training examples
        - (x, y) = singile training example
        - ($x^i, y^i)$ = (i)th training example

## roadmap of machine learning

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-01-20-Linear_regression_model/Untitled 2.png)

## How to represent f(function)?

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-01-20-Linear_regression_model/Untitled 3.png)

- $f_w,_b(x) = wx + b$ and $f(x) = wx + b$ are the same meaning.
- w, b : parameters / coefficients / weights

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-01-20-Linear_regression_model/Untitled 4.png)

- Find w, b : $ŷ^i$ is close to $y^i$ for all ($x^i, y^i$)

## Cost function : Squared error cost function

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-01-20-Linear_regression_model/Untitled 5.png)

- Commenly used in linear regression