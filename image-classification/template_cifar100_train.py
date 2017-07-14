
# coding: utf-8

# In[1]:


import os
import argparse

from common import find_mxnet, data, fit
from common.util import download_file
#from tf_iterators import *
import mxnet as mx
import numpy as np
import numpy 
from mxnet import nd
#from mxnet.module.module_tf import *

# In[2]:


def cif100_iterator(data_dir,batch_size):
    train = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "train.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax1_label',
            data_shape          = (3, 28, 28),
            batch_size          = batch_size,
            pad                 = 0,
            fill_value          = 127,  # only used when pad is valid
            rand_crop           = True,
            #max_random_scale    = 1.0,  # 480 with imagnet, 32 with cifar10 448 with birds 0.93
            #min_random_scale    = 0.53,  # 256.0/480.0
            #max_aspect_ratio    =  0.25,
            random_h            = 36,  # 0.4*90
            random_s            = 50,  # 0.4*127
            random_l            = 50,  # 0.4*127
            #max_rotate_angle    = 10,
            #max_shear_ratio     = 0.1, #
            rand_mirror         = True,
            shuffle             = True)
            #num_parts           = kv.num_workers,
            #part_index          = kv.rank)
    val = mx.io.ImageRecordIter(
            path_imgrec         = os.path.join(data_dir, "test.rec"),
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax1_label',
            batch_size          = batch_size,
            max_random_scale    = 1,  # 480 with imagnet, 32 with cifar10
            min_random_scale    = 1,  # 256.0/480.
            data_shape          = (3,28, 28),
            rand_crop           = False,
            rand_mirror         = False)
            #num_parts           = kv.num_workers,
            #part_index          = kv.rank)
    return train, val


# In[3]:


class Cross_Entropy(mx.metric.EvalMetric):
    """Calculate accuracies of multi label"""

    def __init__(self):
        super(Cross_Entropy, self).__init__('cross-entropy')
    def update(self, labels, preds):
        mx.metric.check_label_shapes(labels, preds)
        label = labels[0].asnumpy()
        pred = preds[0].asnumpy()
        for i in range(label.shape[0]):
            prob = pred[i,numpy.int64(label[i])]
            if len(labels) == 1:
                self.sum_metric += (-numpy.log(prob)).sum()
        self.num_inst += label.shape[0]


# In[4]:
if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description="fine-tune a dataset",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #train = fit.add_fit_args(parser)
    #data.add_data_args(parser)
    #aug = data.add_data_aug_args(parser)
    parser.add_argument('--num_layers', type=int)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--num_branch', type=int)
    parser.add_argument('--num_step', type=int)

   # parser.add_argument('--layer-before-fullc', type=str, default='flatten0',
    #                    help='the name of the layer before the last fullc layer')
    # use less augmentations for fine-tune
    #data.set_data_aug_level(parser, 1)
    # use a small learning rate and less regularizations
    # when training comes to 10th and 20th epoch
	# see http://mxnet.io/how_to/finetune.html and Mu's thesis
    # http://www.cs.cmu.edu/~muli/file/mu-thesis.pdf 
    parser.set_defaults(image_shape='3,28,28', num_epochs=120,
                        lr=.1, lr_schedule=[40000,60000,90000], wd=0.004, mom=0.9, batch_size=128,results_prefix='/home/ubuntu/results/',data_dir='/home/ubuntu/data/cifar100',ctx=[mx.gpu(0)], num_layers=50,num_branch=1,num_step=1
)

    args = parser.parse_args()


    image_shape = args.image_shape
    batch_size = args.batch_size
    num_layers=args.num_layers    
    num_branch=args.num_branch
    num_step=args.num_step
    results_prefix=args.results_prefix
    num_epoch=args.num_epochs
    lr=args.lr
    schedule=args.lr_schedule
    wd=args.wd
    mom=args.mom
    ctx=args.ctx
    data_dir=args.data_dir
    results_prefix=args.results_prefix
    model_prefix='cif100_'+str(num_layers)+str(num_branch)+str(num_step)

    import logging
    
    import logging

    logging.basicConfig(format = '%(asctime)s %(message)s',
                    datefmt = '%m/%d/%Y %I:%M:%S %p',
                    filename = results_prefix+'/eval/'+model_prefix+'.log',
                    level=logging.DEBUG)
    # logging.basicConfig(level=logging.DEBUG)
    # logger = logging.getLogger('mxnet_train_logger')
    # hdlr = logging.FileHandler(results_prefix+'/eval/'+model_prefix+'.log')
    # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    # hdlr.setFormatter(formatter)
    # logger.addHandler(hdlr)

    
    
    train, val = cif100_iterator(data_dir,batch_size)
    label_names = [train.provide_label[0][0]]
    from importlib import import_module
    net = import_module('symbols.td_resnet')
    sym=net.get_unrolled_symbol(100, num_layers, image_shape,num_branch=num_branch,num_step=num_step, conv_workspace=256,bottle_neck=True)






    # imagenet_weights= '/home/ubuntu/models/imagenet_r50-lr05'
    # tornado_weights='/home/ubuntu/models/resnet-101'
    # #model_prefix= 'tryhard-resnet'
    # prefix = tornado_weights
    # #prefix = model_prefix
    # epoch=0
    # save_dict = nd.load('%s-%04d.params' % (prefix, epoch))
    # arg_params_imag = {}
    # aux_params_imag = {}
    # ext_check=['sc','fc1','data']
    # imagenet_par=[]
    # exact_check=['bn1_beta','bn1_gamma']
    # for k, v in save_dict.items():
    #     tp, name = k.split(':', 1)

    #     if tp == 'arg':
    #         arg_params_imag[name] = v


    #         #print name
    #         if not any(ext in name for ext in ext_check):
    #             if not any(ext == name for ext in exact_check):
    #                 imagenet_par.append(name)
    #                 if init_2=='imagenet':
    #                     arg_params_imag['_a_'+name] = v


    #     if tp == 'aux':
    #         aux_params_imag[name] = v
    #         if init_2=='imagenet':
    #             aux_params_imag['_a_'+name] = v
    # del arg_params_imag['fc1_bias']
    # del arg_params_imag['fc1_weight']



    # #arg_params_imag.list_arguments


    # #_a_bn_data_beta

    # symlist=sym.list_arguments()
    # gatelist=[s for s in symlist if 'gate' in s]


    fixed=None
    arg_params=None
    aux_params=None
    mod = mx.mod.Module(sym, label_names=label_names,fixed_param_names=fixed,context=ctx)
    #
    checkpoint_path=results_prefix+model_prefix
    mod.bind(data_shapes=train.provide_data, label_shapes=train.provide_label)
    mod.init_params(initializer=mx.initializer.Uniform(0.01), arg_params=arg_params, aux_params=aux_params,
                        allow_missing=True, force_init=False)

    checkpoint = mx.callback.module_checkpoint(mod,checkpoint_path,period=5)
    #lr_schedule it isimilar to the CIFAR100 schedule but half length of the steps
    schedule = [40000,60000,90000]
    begin_epoch=0


    # In[9]:


    mod.fit(train,
             eval_data=val,
             eval_metric=[Cross_Entropy(),mx.metric.Accuracy()],
             #eval_metric=[mx.metric.Accuracy()],

             batch_end_callback = [mx.callback.log_train_metric(50),mx.callback.Speedometer(batch_size,100)],
             epoch_end_callback=checkpoint,
             allow_missing=False,
             begin_epoch=begin_epoch,
             #log_prefix = model_prefix,
             optimizer_params={'learning_rate':lr, 'momentum': mom,'wd':wd, 'lr_scheduler': mx.lr_scheduler.MultiFactorScheduler(step=schedule,factor=0.1) },
             num_epoch=num_epoch)

