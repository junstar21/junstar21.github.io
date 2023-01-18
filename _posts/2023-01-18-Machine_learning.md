---
title: "About Machine learning"
excerpt: "2023-01-18 Machine learning and algorithms"

# layout: post
categories:
  - TIL
tags:
  - python
  - Machine learning
  - Supervised learning
  - Classification
  - Regression
  - Unsupervised learning
  - Clustering
spotifyplaylist: spotify/playlist/2KaQr0nx66AX399ZLLuTVf?si=43a48325c8fc4b16
---
### **⚠️ 해당 내용은 Coursera의 Supervised Machine Learning: Regression and Classification 강의 내용을 정리한 내용입니다.**

# What is machine learning?

> Field of study that gives computers the ability to learn without being explicitly programmed.” Arthur Samuel (1959)
> 

Start from checkers games. learns from tons of games.

## Machine learning algorithms

- Supervised learning : use most in real-world appliciations. rapid advancements
- Unsuperviesd learning
- Recommender systems
- Reinforcement learning

In this course, we are gonna learn Superviesed learning, Unsuperviese leaning, and Recommender systems.

- **Practical advice for applying learning algorithms** : even more important then learning algorithms

# Supervised learning

X(input) → y(output label) : Learns from being given “**right answers**”

## applications

| Input(X) | Output(y) | Application |
| --- | --- | --- |
| email | spam?( 0/1) | spam filtering |
| audio | text transcripts | speech recognition |
| English | Spanish | machine translation |
| ad, user info | cilck? (0/1) | online advertising |
| image, radar info | position of other cars | self-driving car |
| image of phone | defect? (0/1) | visual inspection |

## Regression : Housing price prediction

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-01-18-Machine_learning/Untitled.png)

- How much value will it be if the house size is 750 feet^2? ⇒ predicting by X, y
- **Regresison** : **Predict** a number **from infinitely many possible outputs**

## Classification : Breast cancer detection

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-01-18-Machine_learning/Untitled 1.png)

- malignant : danger of cancer, benign : possible of a just tumor
- **Classification** : predict **categories, small number** of possibile outputs

### Two or more inputs

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-01-18-Machine_learning/Untitled 2.png)

- Found a boundary(pink line)

# **Unsupervised learning**

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-01-18-Machine_learning/Untitled 3.png)

- Data only comes with inputs X, but not output labels y
- Algorithm has to find **structure** in the data
- Find a similar group or cluster ⇒ **Clustering Algorithm**
- Find unusual data points ⇒ **Anomaly detection**
- Compress data using fewer numbers ⇒ **Dimensionality reduction**

### Example of Clustring

- Google news : keyword ‘panda, twin, zoo’
    
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-01-18-Machine_learning/Untitled 4.png)
    
- DNA microarray
    
    ![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-01-18-Machine_learning/Untitled 5.png)
    

### Clustring : Grouping customers

![]({{ site.url }}{{ site.baseurl }}/assets/images/2023-01-18-Machine_learning/Untitled 6.png)