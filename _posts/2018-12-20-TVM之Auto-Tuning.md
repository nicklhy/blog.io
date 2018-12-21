---
title: TVM之神经网络Auto-Tuning
date: 20181220
author: nicklhy
layout: post
categories:
  - 其他
status: publish
type: post
published: false
---

（本文主要参考TVM在arxiv上的[论文](https://arxiv.org/pdf/1802.04799.pdf)和官方[tutorials](https://docs.tvm.ai/tutorials)）。

说起深度学习发展到如今的盛况，各类开源深度学习框架绝对是功不可没，从最早的Theano、Caffe，到如今的TensorFlow、PyTorch、MXNet、Keras等，AI学术圈的无数“炼丹老法师”们其实都是在用这些工具尝试各种不同的“炼丹姿势”以获得传说中最state-of-the-art的“丹药”，然后就可以把“炼丹”经历发表到各种全球顶级期刊、会议上扬名立万走向人生巅峰了，相比之下，这些工具作者们得到的关注则是远小于后面这些工具使用者，不过今天我倒不是说要在这篇文章里如何如何为前者正名，引起大众的关注，只是想简单记录一下最近让我非常兴奋的一个关于模型通用硬件加速的小实验。



## 背景介绍

比较初级的算法研发人员或者在校学生们可能大多都觉得模型部署并不是一件非常困难的事情，无非是在服务器上安装TensorFlow，然后写一套简单的加载模型与实际预测的函数，嵌入到某个Web框架里，就可以实现用户请求一张图片立即返回这个图片是猫是狗的结果了；稍微有些经验的工程师则会想到这里还涉及到高并发、负载均衡等常见的大数据问题，但这些也算是有成熟的解决方案，有很多之前的经验可以借鉴；但是真正有经验的专业深度学习系统/平台研发人员则会告诉你这里还存在更多的其他问题，移动端/专用AI芯片的框架研发（因为TF、Caffe等大多数DL框架都无法直接支持除Nvidia GPU和x86 CPU以外的处理器）、异构集群的服务部署、模型Inference加速、延时和并发的矛盾、在Nvidia GPU平台上如何做到在特定场景下比CuDNN/TensorRT速度更快。。。好吧，问题还是挺多的。

TVM作为Tianqi大神在MXNet之后的又一个大项目，在发布之后历经多个版本到如今应该也算初步实现了当初诞生时所设下的目标了，



## Graph优化

常见的深度神经网络本质上可以看成是一个计算图(computational graph)，下图给出了一个两层卷积CNN的例子作为展示

![](/images/post/2018/12/conv_2layer_graph.png)



比较早期的时候大家主要把优化精力放在了operator上（例如conv的n种实现），而选择，则慢慢的开始发现其实在graph层面就可以做非常多的改进，TVM在这个层面实现了如下几种

* operator fusion: 把多个独立的operator融合成一个；
* constant-folding: 把一些可以静态计算出来的常量提前计算出来；
* static memory planning pass: 预先把需要的存储空间申请下来，避免动态分配；
* data layout transformations: 有些特殊的计算设备可能会对不同的data layout (i.e. NCHW, NHWC, HWCN)有不同的性能，TVM可以根据实际需求在graph层面就做好这些转换。

后三点其实从定义上看大体都能理解其具体含义，但第一点op fusion对很多人而言可能还是会有些困扰，简单来说，op fusion就是“把原本n个函数调用合并到一个函数里面”，以最常见的conv-bn-relu为例，假设输入为x，如果按照正常的计算流程我们需要做如下操作

```python
x1 = conv(x)
x2 = bn(x1)
y = relu(x2)
```

可以看到，我们为了得到最终的结果y，需要进行三次函数调用，并且存储中间结果x1和x2，但如果我们进行了op fusion，整个过程则会变成如下形式

```python
y = conv_bn_relu(x)
```

此时，所有中间计算结果都被省略，计算速度和存储空间同时得到了优化。TVM的作者在论文中对各种operator做了个分类：

* injective (one-to-one map, e.g., add)
* reduction (e.g., sum)
* complex-out-fusable (can fuse element-wise map to output, e.g., conv2d)
* opaque (cannot be fused, e.g., sort)

实际在做op fusion时遵循如下原则

* 任意多个（连续）injective op都可以被合并成一个op；
* injective op如果作为一个reduction op的输入，则可以被合并；
* 如果complex-out op（如conv2d）的输出如果接的是element-wise，则可以被合并；

下图展示了一些常见op fusion给网络带来的加速效果：

![](/images/post/2018/12/op_fusion_speed_up.png)