# gluon_GMLoss
The implementation of the paper "Rethinking Feature Distribution for Loss Functions in Image Classification"
The implementation of the paper "Rethinking Feature Distribution for Loss Functions in Image Classification"
===========================================================================================================================================
参考文献和代码链接
-------------------------------------------------------------------------------------------------------------------------------------------
* (Rethinking Feature Distribution for Loss Functions in Image Classification)[https://arxiv.org/abs/1803.02988]
* (caffe&tensorflow gmloss)[https://github.com/WeitaoVan/L-GM-loss] 
* (pytorch gmloss)[https://github.com/ChaofWang/L-GM_loss_pytorch]
* (gluon gmloss)[https://github.com/LeeJuly30/L-GM-Loss-For-Gluon]

---------------------------------------------------------------------------------------------
## 尝试过程
The first edition is improved from the [LeeJuly30](https://github.com/LeeJuly30/L-GM-Loss-For-Gluon) , I use the MSRA initialization method to replace the Xavier and add the margin
parameter increase schedule according to caffe edition,which is helpful to improve the acc performance. I am still confused about the parameter margin_add in caffe .  

The latest edition v3 is based on the caffe and gluon edition .  I use softmaxloss to replace the Gluon edition's gmloss,gmloss layer is only used to provide the logits_with_margin like caffe code. .
This margin is seemed as the logits in the softmaxloss. And I also rewrite the likelihood_reg_loss part in the gluon edition using caffe's method,which will be more understandable.
I found the validation part in the gluon edition is very wired,which still need to input the groundtruth in the gmloss to do the inference ,this is unreasonable. Therefore I rewrite 
the validation part as well ,but the acc is decredsed a lot .

The v4 edition is based on the caffe/tensorflow edition only,without the variable var parameter ,which can reach the best performance 0.85 val acc from scratch.

I already try some parameter strategy,like choosing the last globalaveragepooling's output as the gmloss's feature input,decreacing the weight decay, warm up strategy,modifying the 
number of input features ,modifying the lr decay schedule.Most of them are useless, the reason for the low acc is still from the implementation of the gmloss. 

----------------------------------------------------------------------------------------------
## 发现模仿原作者的caffe和tensorflow版本会有更好的效果
由于原作者没有在tensorflow中加入updated var的效果，参考caffe代码进行添加，但发现添加了之后val acc反而从不加的0.89下降到了0.84的水平.通过一系列的调参尝试后，如下两个方法会有较好的效果
### 1.warmup策略 
由于开始的损失会很大，如果直接用大的学习率，就梯度爆炸，但一开始就用很小的学习率，再继续decay又很容易陷入局部最小。这里用了0.01的lr预热了10个epoch，接着再用0.1的lr进行模型的训练。

### 2.decay策略
decay_rate一开始的选择有问题，如果还是用论文里说的0.1会衰减的太快，在第二次decay之后就已经陷入了局部最小，这里用了0.5来做decay，用0.1这个大的lr训练较长时间后，后面每隔50个epoch就做一次decay。

### 3.初始化的方法
不能盲目使用caffe版本的MSRA初始化去初始化全部的参数，gluon里的预训练模型res20用Xavier会有更好的效果，仅仅对gmloss中的mean矩阵使用MSRA初始化

### 目前达到的效果
![](https://github.com/yuyijie1995/gluon_GMLoss/blob/master/3_23_warmup_400epoch_decay6times.png)

