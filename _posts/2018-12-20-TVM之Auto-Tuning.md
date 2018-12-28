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



说起神经网络模型加速，可能大部分人首先想到的会是KD、剪枝、量化这些模型压缩方法，不过如果在实际项目中算法研发人员已经尝试了所有模型压缩的方法，而性能依然和需求有差距，我们是否还有其他可以尝试的方案呢？一般来说，选择一个高效的深度学习库肯定是应该被最先考虑的，如果你用的是caffe一代、Keras、PyTorch这些本来就性能比较一般主要面向学术圈的计算库，那肯定不太合适，事实上，在英伟达GPU平台如果只考虑模型预测，业界中公认的性能标杆应该还是英伟达自家的TensorRT库，不过如果让真正用过这个库的人来做分享，可能你又会听到他的一肚子苦水，因为这个TensorRT对各种operator的支持实在有限，很多时候你还得用C++自己去实现一些需要的网络层，另外，如果你需要支持的计算设备并不仅仅是英伟达的GPU，还得支持诸如CPU甚至ARM移动端等，那么接下来针对每个平台你还得再去找其他的计算库来做支持，需要的人力和时间绝对超乎你的想象。

本文尝试介绍一个通过TVM来对模型进行硬件加速的方案，并在英伟达GPU上给出一些对比实验的结果以展示其可行性。



## TVM概览

首先看一下下方TVM的系统概览图

![](/images/post/2018/12/tvm_system_overview.png)

可以看出使用TVM进行模型部署的完整流程包括：

* TF、PyTorch、MXNet等frontend深度学习框架的模型到计算图IR的转换；
* 计算图IR的graph优化，得到Optimized Computational Graph;
* 对计算图中的每个op得到一种用Tensor Expression描述的Tensor计算表达式，并针对所需的硬件平台，生成最小的计算原语(primitives)；
* 使用某种基于机器学习的Automated Optimizer生成经过优化的Low Level Loop Program；
* 生成特定于硬件设备的二进制程序；
* 生成可以部署的module；

下面我们自上而下分别聊一聊上述过程的几个核心步骤。



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



