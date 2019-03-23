
from __future__ import division

import argparse, time, logging, random, math

import numpy as np
import mxnet as mx

from mxnet import gluon, nd
from mxnet import autograd as ag
from mxnet.gluon import nn
from mxnet.gluon.data.vision import transforms
from L_GM_v6 import L_GM_Loss
import pdb

from gluoncv.model_zoo import get_model
from gluoncv.utils import makedirs, TrainingHistory
from gluoncv.utils import LRScheduler

def parse_arguments():
    parser=argparse.ArgumentParser()
    parser.add_argument('--batch-size',type=int,default=128)
    parser.add_argument('--epochs',type=int,default=400)
    parser.add_argument('--workers',type=int,default=8)
    parser.add_argument('--lr-decay-epoch',type=list,default=[100,150,200,250,300,350,np.inf])
    parser.add_argument('--lr-decay',type=float,default=0.5)
    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--optimizer',type=str,default='nag')
    parser.add_argument('--momentum',type=float,default=0.9)
    parser.add_argument('--wd',type=float,default=5e-4)
    parser.add_argument('--gpus',type=str,default='0,1,2,3')
    parser.add_argument('--margin',type=float,default=0.1)
    parser.add_argument('--lamda',type=float,default=0.1)
    parser.add_argument('--mult',type=float,default=0.01)
    parser.add_argument('--warmup_epochs',type=int,default=10)
    parser.add_argument('--warmup-lr',type=float,default=0.01)
    parser.add_argument('--num-samples', type=int, default=-1,
                        help='Training images. Use -1 to automatically get the number.')

    parser.add_argument('--lr-mode', type=str, default='step',
                        help='learning rate scheduler mode. options are step, poly and cosine.')

    args=parser.parse_args()
    return args

args=parse_arguments()
# number of GPUs to use
logger=logging.getLogger()
logger.setLevel(logging.DEBUG)
file_handler=logging.FileHandler('3_23_warmup_400epoch_decay6times.log')
logger.addHandler(file_handler)
logger.info(args)
ctx=[mx.gpu(int(i)) for i in args.gpus.split(',')]
num_gpus = len(ctx)
net = get_model('cifar_resnet20_v1', classes=10)
#net.output=nn.GlobalAvgPool1D(100)
softmax_loss=gluon.loss.SoftmaxCrossEntropyLoss()
gmloss=L_GM_Loss(10, 10, args.margin, args.lamda, args.mult)
gmloss.initialize(mx.init.MSRAPrelu(),ctx=ctx)
net.initialize(mx.init.Xavier(), ctx = ctx)
params=net.collect_params()
params.update(gmloss.collect_params())
#params.update(gmloss.collect_params(select='mean'))


transform_train = transforms.Compose([
    # Randomly crop an area, and then resize it to be 32x32
    transforms.RandomResizedCrop(32),
    # Randomly flip the image horizontally
    transforms.RandomFlipLeftRight(),
    # Randomly jitter the brightness, contrast and saturation of the image
    transforms.RandomColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    # Randomly adding noise to the image
    transforms.RandomLighting(0.1),
    # Transpose the image from height*width*num_channels to num_channels*height*width
    # and map values from [0, 255] to [0,1]
    transforms.ToTensor(),
    # Normalize the image with mean and standard deviation calculated across all images
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])


transform_test = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
])

per_device_batch_size = args.batch_size
# Number of data loader workers
num_workers = args.workers
# Calculate effective total batch size
batch_size = per_device_batch_size * num_gpus

train_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10(train=True).transform_first(transform_train),
    batch_size=batch_size, shuffle=True, last_batch='discard', num_workers=num_workers)

# Set train=False for validation data
val_data = gluon.data.DataLoader(
    gluon.data.vision.CIFAR10(train=False).transform_first(transform_test),
    batch_size=batch_size, shuffle=False, num_workers=num_workers)


# Learning rate decay factor
lr_decay = args.lr_decay



# Epochs where learning rate decays
lr_decay_epoch = args.lr_decay_epoch


# Nesterov accelerated gradient descent
optimizer = args.optimizer
# Set parameters
optimizer_params = {'learning_rate': args.lr, 'wd': args.wd, 'momentum': args.momentum}

# Define our trainer for net
trainer = gluon.Trainer(params, optimizer, optimizer_params)


train_metric = mx.metric.Accuracy()
train_history = TrainingHistory(['training-error', 'validation-error'])


def test(ctx, val_data):
    metric = mx.metric.Accuracy()
    for i, batch in enumerate(val_data):
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        outputs = [net(X) for X in data]
        probabilitys=[gmloss(output,None) for output in outputs]
        metric.update(label, probabilitys)
    return metric.get()

epochs = args.epochs
lr_decay_count = 0

for epoch in range(epochs):
    tic = time.time()
    train_metric.reset()
    train_loss = 0
    # Learning rate decay
    if  epoch<args.warmup_epochs:
        trainer.set_learning_rate(args.warmup_lr)
    if epoch==args.warmup_epochs:
        trainer.set_learning_rate(trainer.learning_rate*10)

    if epoch == lr_decay_epoch[lr_decay_count]:
        trainer.set_learning_rate(trainer.learning_rate*lr_decay)
        lr_decay_count += 1

    # Loop through each batch of training data
    for i, batch in enumerate(train_data):
        # Extract data and label
        data = gluon.utils.split_and_load(batch[0], ctx_list=ctx, batch_axis=0)
        label = gluon.utils.split_and_load(batch[1], ctx_list=ctx, batch_axis=0)
        # AutoGrad
        with ag.record():
            outputs = [net(X) for X in data]
            likelihood_loss = [gmloss(output, y)[1] for output, y in zip(outputs, label)]
            margin_logits = [gmloss(output, y)[0] for output, y in zip(outputs, label)]

            loss_1=[softmax_loss(margin_logit,y) for margin_logit,y in zip(margin_logits,label)]
            loss = [loss_1[i] + likelihood_loss[i] for i in range(min(len(loss_1), len(likelihood_loss)))]
        # Backpropagation
        for l in loss:
            l.backward()

        # Optimize
        #lr_scheduler.update(i,epoch)
        trainer.step(batch_size)
        #pdb.set_trace()
        # Update metrics
        train_loss += sum([l.sum().asscalar() for l in loss])
        train_metric.update(label, margin_logits)


    name, acc = train_metric.get()
    # Evaluate on Validation data
    name, val_acc = test(ctx, val_data)

    # Update history and print metrics
    train_history.update([1-acc, 1-val_acc])
    logger.info('[Epoch %d] train=%f val=%f loss=%f time: %f' %
        (epoch, acc, val_acc, train_loss, time.time()-tic))
    print('[Epoch %d] train=%f val=%f loss=%f time: %f' %
        (epoch, acc, val_acc, train_loss, time.time()-tic))
    print(trainer.learning_rate)





