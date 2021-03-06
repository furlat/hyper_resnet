'''
Adapted from https://github.com/tornadomeet/ResNet/blob/master/symbol_resnet.py
Original author Wei Wu

Implemented the following paper:

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun. "Identity Mappings in Deep Residual Networks"
'''
import mxnet as mx
def param_dictionary():
    global conv_w_dict
    global count_dict
    global beta_dict
    global gamma_dict
    global fc_w_dict
    global fc_b_dict
    conv_w_dict={}
    count_dict={}
    gamma_dict={}
    beta_dict={}
    fc_w_dict={}
    fc_b_dict={}
    

def Activation(data,act_type,name):
    if name not in count_dict:
        count_dict[name]=0
    act=mx.sym.Activation(data=data, act_type=act_type, name=name + '_relu1_'+str(count_dict[name]))
    count_dict[name]+=1
    return act

def Convolution(data,num_filter,kernel,stride,pad,name,workspace,no_bias=True):
    
    if name not in count_dict:
        
        conv_w_dict[name]=mx.sym.Variable(name + '_weight', init=mx.init.Uniform(),dtype='float32')
        count_dict[name]=0
    conv = mx.sym.Convolution(data=data,weight=conv_w_dict[name], num_filter=num_filter, kernel=kernel, stride=stride, pad=pad,no_bias=no_bias, workspace=workspace, name=name+'_'+str(count_dict[name]))
    count_dict[name]+=1
    return conv
                                   
                                   

def BatchNorm(data,momentum,name,fix_gamma=True,eps=2e-5):
    if name not in count_dict:
        gamma_dict[name]=mx.sym.Variable(name +'_gamma', init=mx.init.Uniform(),dtype='float32')
        beta_dict[name]=mx.sym.Variable(name +'_beta', init=mx.init.Zero(),wd_mult=0,dtype='float32')
        count_dict[name]=0
    bn=mx.sym.BatchNorm(data=data,gamma=gamma_dict[name],beta=beta_dict[name], fix_gamma=False, eps=2e-5, momentum=momentum, name=name+'_'+str(count_dict[name]))
    count_dict[name]+=1
    return bn
        
def FullyConnected(data,num_hidden,name,bias_init=mx.init.Uniform()):
    if name not in count_dict:
        count_dict[name]=0
        fc_w_dict[name]=mx.sym.Variable(name + '_weight', init=mx.init.Uniform(),dtype='float32')
        fc_b_dict[name]=mx.sym.Variable(name + '_bias', init=bias_init,wd_mult=0,dtype='float32')
    fc = mx.symbol.FullyConnected(data=data,weight=fc_w_dict[name],bias=fc_b_dict[name], num_hidden=num_hidden, name=name+str(count_dict[name]))
    count_dict[name]+=1
    return fc

def dyn_hypergated_residual_unit(data,hyper, num_filter, stride, dim_match,num_branch,t, name,batch_size=128,image_shape=(3,224,224),num_gpus=1, bottle_neck=True, bn_mom=0.1, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    k=1
    def branch_bneck(data,num_filter,stride,dim_match,name,idx,bn_mom=bn_mom, workspace=256, memonger=False):
        idx=str(idx)
        bn1 = BatchNorm(data=data, momentum=bn_mom, name=name + '_bn1_'+idx)
        act1 =Activation(data=bn1, act_type='relu', name=name + '_relu1_'+idx)
        conv1 = Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0), name=name + '_conv1_'+idx, no_bias=True, workspace=workspace)
        bn2 = BatchNorm(data=conv1, momentum=bn_mom, name=name + '_bn2_'+idx)
        act2 = Activation(data=bn2, act_type='relu', name=name + '_relu2_'+idx)
        conv2 = Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1), name=name + '_conv2_'+idx,no_bias=True, workspace=workspace)
        bn3 = BatchNorm(data=conv2, momentum=bn_mom, name=name + '_bn3_'+idx)
        act3 = Activation(data=bn3, act_type='relu', name=name + '_relu3_'+idx)
        conv3 = Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),name=name + '_conv3_'+idx, no_bias=True, workspace=workspace)
        return conv3

    def branch_no_bneck(data,num_filter,stride,dim_match,name,idx,bn_mom=bn_mom, workspace=256, memonger=False):
        idx=str(idx)
        bn1 = BatchNorm(data=data, momentum=bn_mom, name=name + '_bn1_'+idx)
        act1 =Activation(data=bn1, act_type='relu', name=name + '_relu1_'+idx)
        conv1 = Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1), name=name + '_conv1_'+idx, no_bias=True, workspace=workspace)
        bn2 = BatchNorm(data=conv1, momentum=bn_mom, name=name + '_bn2_'+idx)
        act2 =Activation(data=bn2, act_type='relu', name=name + '_relu2_'+idx)
        conv2 =Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),name=name + '_conv2_'+idx, no_bias=True, workspace=workspace)
        return conv2    

    bn1 = BatchNorm(data=data, momentum=bn_mom, name=name + '_bn1')
    act1 =Activation(data=bn1, act_type='relu', name=name + '_relu1')
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        conv_3=[]
        for i in range(k):

            out=branch_bneck(data,num_filter,stride,dim_match,name,i,bn_mom=0.9, workspace=256, memonger=False)
            if t>0:
                gate=FullyConnected(data=hyper,num_hidden=num_filter, name=name+'hyper_gate_'+str(i))
                
            else: #at step 0 the gate is set to 1
                gate=mx.sym.Variable(name + 'hyper_gate_placeholder', init=mx.init.One(),shape=(batch_size,num_filter), lr_mult=0,dtype='float32')
            gate=mx.sym.reshape(data=gate,shape=(0,-1,1,1))
            gate=Activation(data=gate, act_type='relu', name=name + '_hyper_gate_relu_'+str(i))
            conv_3.append(mx.sym.broadcast_mul(out, gate))

        if dim_match:
            shortcut = data
        else:
            bn1 = BatchNorm(data=data, momentum=bn_mom, name=name + '_bn1_sc')
            act1 =Activation(data=bn1, act_type='relu', name=name + '_relu1_sc')
            shortcut = Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride,pad=(0,0),name=name+'_sc', no_bias=True, workspace=workspace)
        if memonger:
            shortcut._set_attr(mirror_stage='True')   
        return sum(conv_3) + shortcut
    
    else:
        conv_2=[]
        for i in range(k):
            out=branch_no_bneck(data,num_filter,stride,dim_match,name,i,bn_mom=0.9, workspace=256, memonger=False)
            if t>0:
                gate=FullyConnected(data=hyper,num_hidden=num_filter, name=name+'hyper_gate_'+str(i))
                
            else: #at step 0 the gate is set to 1
                gate=mx.sym.Variable(name + 'hyper_gate_placeholder', init=mx.init.One(),shape=(batch_size,num_filter), lr_mult=0,dtype='float32')
            gate=mx.sym.reshape(data=gate,shape=(0,-1,1,1))
            gate=Activation(data=gate, act_type='relu', name=name + '_hyper_gate_relu_'+str(i))
            conv_2.append(mx.sym.broadcast_mul(out, gate))
       
           
        if dim_match:
            shortcut = data
        else:
            bn1 = BatchNorm(data=data, momentum=bn_mom, name=name + '_bn1_sc')
            act1 =Activation(data=bn1, act_type='relu', name=name + '_relu1_sc')
            shortcut = Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride,pad=(0,0),name=name+'_sc', no_bias=True,workspace=workspace)
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return sum(conv_2) + shortcut

def dyn_hypergated_resnet(units, num_stages, filter_list, num_classes, image_shape,num_branch,num_step,batch_size, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    param_dictionary()
    num_unit = len(units)
    assert(num_unit == num_stages)
    label=mx.sym.Variable(name='softmax1_label')
    data = mx.sym.Variable(name='data')
    data = mx.sym.identity(data=data, name='id')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    (nchannel, height, width) = image_shape
    out=[]
    fc1=[]
    sfo=[]
    for t in range(num_step):
        if t==0:
            hyper=mx.sym.Variable(name='prior',init=mx.init.One(),shape=(16,100),dtype='float32') 
        else:
            hyper=mx.sym.identity(data=fc1[t-1], name='prior_'+str(t))
            
        hyper=mx.sym.SoftmaxActivation(hyper)
        hyper=mx.symbol.Flatten(hyper)

        if height <= 32:            # such as cifar10
            body = Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),name="conv0_"+str(t),
                                      no_bias=True,  workspace=workspace)
        else:                       # often expected to be 224 such as imagenet
            body = Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3), name="conv0_"+str(t),
                                      no_bias=True, workspace=workspace)
            body = BatchNorm(data=body, momentum=bn_mom, name='bn0')
            body = mx.sym.Activation(data=body, act_type='relu', name='relu0_'+str(t))
            body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

        for i in range(num_stages):

            body = dyn_hypergated_residual_unit(body,hyper, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,num_branch,t, name='stage%d_unit%d' % (i + 1, 1),batch_size=batch_size, bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
            for j in range(units[i]-1):
                body = dyn_hypergated_residual_unit(body,hyper, filter_list[i+1], (1,1), True,num_branch,t, name='stage%d_unit%d' % (i + 1, j + 2),batch_size=batch_size, bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
        bn1 = BatchNorm(data=body, momentum=bn_mom, name='bn1')
        relu1 = Activation(data=bn1, act_type='relu', name='relu1')
        # Although kernel is not used here when global_pool=True, we should put one
        pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1_'+str(t))
        flat = mx.symbol.Flatten(data=pool1)
        fc1.append(FullyConnected(data=flat, num_hidden=num_classes, name='fc1'))
        sfo.append(mx.symbol.SoftmaxOutput(data=fc1[t], name='softmax'+str(t),label=label))
    #return mx.sym.Group(sfo)
    return sfo[num_step-1]



def dyn_k_branch_residual_unit(data,hyper, num_filter, stride, dim_match,num_branch, name,batch_size=64,image_shape=(3,224,224),num_gpus=1, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    k=num_branch
    def branch_bneck(data,num_filter,stride,dim_match,name,idx,bn_mom=0.9, workspace=256, memonger=False):
        idx=str(idx)
        bn1 = BatchNorm(data=data, momentum=bn_mom, name=name + '_bn1_'+idx)
        act1 =Activation(data=bn1, act_type='relu', name=name + '_relu1_'+idx)
        conv1 = Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0), name=name + '_conv1_'+idx, no_bias=True, workspace=workspace)
        bn2 = BatchNorm(data=conv1, momentum=bn_mom, name=name + '_bn2_'+idx)
        act2 = Activation(data=bn2, act_type='relu', name=name + '_relu2_'+idx)
        conv2 = Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1), name=name + '_conv2_'+idx,no_bias=True, workspace=workspace)
        bn3 = BatchNorm(data=conv2, momentum=bn_mom, name=name + '_bn3_'+idx)
        act3 = Activation(data=bn3, act_type='relu', name=name + '_relu3_'+idx)
        conv3 = Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0),name=name + '_conv3_'+idx, no_bias=True, workspace=workspace)
        return conv3

    def branch_no_bneck(data,num_filter,stride,dim_match,name,idx,bn_mom=0.9, workspace=256, memonger=False):
        idx=str(idx)
        bn1 = BatchNorm(data=data, momentum=bn_mom, name=name + '_bn1_'+idx)
        act1 =Activation(data=bn1, act_type='relu', name=name + '_relu1_'+idx)
        conv1 = Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1), name=name + '_conv1_'+idx, no_bias=True, workspace=workspace)
        bn2 = BatchNorm(data=conv1, momentum=bn_mom, name=name + '_bn2_'+idx)
        act2 =Activation(data=bn2, act_type='relu', name=name + '_relu2_'+idx)
        conv2 =Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),name=name + '_conv2_'+idx, no_bias=True, workspace=workspace)
        return conv2    

    bn1 = BatchNorm(data=data, momentum=bn_mom, name=name + '_bn1')
    act1 =Activation(data=bn1, act_type='relu', name=name + '_relu1')
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        conv_3=[]
        for i in range(k):

            out=branch_bneck(data,num_filter,stride,dim_match,name,i,bn_mom=0.9, workspace=256, memonger=False)
            gate=FullyConnected(data=hyper,num_hidden=1, name=name+'_hyper_gate_'+str(i))
            gate=mx.sym.reshape(data=gate,shape=(0,-1,1,1))
            conv_3.append(mx.sym.broadcast_mul(out, gate))

        if dim_match:
            shortcut = data
        else:
            bn1 = BatchNorm(data=data, momentum=bn_mom, name=name + '_bn1_sc')
            act1 =Activation(data=bn1, act_type='relu', name=name + '_relu1_sc')
            shortcut = Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride,pad=(0,0),name=name+'_sc', no_bias=True, workspace=workspace)
        if memonger:
            shortcut._set_attr(mirror_stage='True')   
        return sum(conv_3) + shortcut
    
    else:
        conv_2=[]
        for i in range(k):
            out=branch_no_bneck(data,num_filter,stride,dim_match,name,i,bn_mom=0.9, workspace=256, memonger=False)
            gate=FullyConnected(data=hyper,num_hidden=1, name=name+'hyper_gate_'+str(i))
            gate=mx.sym.reshape(data=gate,shape=(0,-1,1,1))
            conv_2.append(mx.sym.broadcast_mul(out, gate))
       
           
        if dim_match:
            shortcut = data
        else:
            bn1 = BatchNorm(data=data, momentum=bn_mom, name=name + '_bn1_sc')
            act1 =Activation(data=bn1, act_type='relu', name=name + '_relu1_sc')
            shortcut = Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride,pad=(0,0),name=name+'_sc', no_bias=True,workspace=workspace)
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return sum(conv_2) + shortcut

def dyn_k_branch_resnet(units, num_stages, filter_list, num_classes, image_shape,num_branch,num_step, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    param_dictionary()
    num_unit = len(units)
    assert(num_unit == num_stages)
    label=mx.sym.Variable(name='softmax1_label')
    data = mx.sym.Variable(name='data')
    data = mx.sym.identity(data=data, name='id')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    (nchannel, height, width) = image_shape
    out=[]
    fc1=[]
    sfo=[]
    for t in range(num_step):
        if t==0:
            hyper=mx.sym.Variable(name='prior',init=mx.init.One(),shape=(16,100),dtype='float32') 
        else:
            hyper=mx.sym.identity(data=fc1[t-1], name='prior_'+str(t))
            
        hyper=mx.sym.SoftmaxActivation(hyper)
        hyper=mx.symbol.Flatten(hyper)

        if height <= 32:            # such as cifar10
            body = Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),name="conv0_"+str(t),
                                      no_bias=True,  workspace=workspace)
        else:                       # often expected to be 224 such as imagenet
            body = Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3), name="conv0_"+str(t),
                                      no_bias=True, workspace=workspace)
            body = BatchNorm(data=body, momentum=bn_mom, name='bn0')
            body = mx.sym.Activation(data=body, act_type='relu', name='relu0_'+str(t))
            body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

        for i in range(num_stages):

            body = dyn_k_branch_residual_unit(body,hyper, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,num_branch, name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
            for j in range(units[i]-1):
                body = dyn_k_branch_residual_unit(body,hyper, filter_list[i+1], (1,1), True,num_branch, name='stage%d_unit%d' % (i + 1, j + 2), bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
        bn1 = BatchNorm(data=body, momentum=bn_mom, name='bn1')
        relu1 = Activation(data=bn1, act_type='relu', name='relu1')
        # Although kernel is not used here when global_pool=True, we should put one
        pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1_'+str(t))
        flat = mx.symbol.Flatten(data=pool1)
        fc1.append(FullyConnected(data=flat, num_hidden=num_classes, name='fc1'))
        sfo.append(mx.symbol.SoftmaxOutput(data=fc1[t], name='softmax'+str(t),label=label))
    #return mx.sym.Group(sfo)
    return sfo[num_step-1]





























def k_branch_residual_unit(data, num_filter, stride, dim_match,num_branch, name,scalar_gate=False,elemwise_gate=False,batch_size=64,image_shape=(3,224,224),num_gpus=4, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    k=num_branch
    def branch_bneck(act1,num_filter,stride,dim_match,name,idx,bn_mom=0.9, workspace=256, memonger=False):
        idx=str(idx)
        
        conv1 = mx.sym.Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1_'+idx)
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2_'+idx)
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2_'+idx)
        conv2 = mx.sym.Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2_'+idx)
        bn3 = mx.sym.BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3_'+idx)
        act3 = mx.sym.Activation(data=bn3, act_type='relu', name=name + '_relu3_'+idx)
        conv3 = mx.sym.Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3_'+idx)
        return conv3

    def branch_no_bneck(act1,num_filter,stride,dim_match,name,idx,bn_mom=0.9, workspace=256, memonger=False):
        idx=str(idx)
        # bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1_'+idx)
        # act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1_'+idx)
        bn2 = mx.sym.BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2_'+idx)
        act2 = mx.sym.Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = mx.sym.Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2_'+idx)
        return conv2    

    bn1 = mx.sym.BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
    act1 = mx.sym.Activation(data=bn1, act_type='relu', name=name + '_relu1')
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        conv_3=[]
        for i in range(k):
       
            #scalar gate
          
            if scalar_gate:
                out=branch_bneck(act1,num_filter,stride,dim_match,name,i,bn_mom=0.9, workspace=256, memonger=False)
                gate=mx.sym.Variable(name+'gate_'+str(i), init=mx.init.One(),shape=(1),dtype='float32')
                conv_3.append(mx.sym.broadcast_mul(out, gate))
            else:
                conv_3.append(branch_bneck(act1,num_filter,stride,dim_match,name,i,bn_mom=0.9, workspace=256, memonger=False))
                #shape=conv_3[i].infer_shape(data=(batch_size/num_gpus,image_shape[0],image_shape[1],image_shape[2]))
        
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        if elemwise_gate:
                gate=mx.sym.Variable(name+'_elemwise_gate',init=mx.init.One(),dtype='float32')
                return (sum(conv_3) + shortcut)*gate
        else:
            
            return sum(conv_3) + shortcut
    else:
        conv_2=[]
        for i in range(k):
            if scalar_gate:
                print 'scalar gate'
                out=branch_no_bneck(act1,num_filter,stride,dim_match,name,i,bn_mom=0.9, workspace=256, memonger=False)
                gate=mx.sym.Variable(name+'gate_'+str(i), init=mx.init.One(),shape=(1,1,1,1),dtype='float32')
                conv_2.append(mx.sym.broadcast_mul(out, gate))
            else:    
                conv_2.append(branch_no_bneck(act1,num_filter,stride,dim_match,name,i,bn_mom=0.9, workspace=256, memonger=False))
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
            print('sono buggato')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        if elemwise_gate:
            gate=mx.sym.Variable(name+'_elemwise_gate',init=mx.init.One(),dtype='float32')
            return (sum(conv_2) + shortcut)*gate    
        return sum(conv_2) + shortcut

def k_branch_resnet(units, num_stages, filter_list, num_classes, image_shape,num_branch,scalar_gate,elemwise_gate, bottle_neck=True, bn_mom=0.9, workspace=256, memonger=False):
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    data = mx.sym.identity(data=data, name='id')
    data = mx.sym.BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    (nchannel, height, width) = image_shape
    if height <= 32:            # such as cifar10
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    else:                       # often expected to be 224 such as imagenet
        body = mx.sym.Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = mx.sym.Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    for i in range(num_stages):
        
        body = k_branch_residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,num_branch,
                             name='stage%d_unit%d' % (i + 1, 1),scalar_gate=scalar_gate,elemwise_gate=elemwise_gate, bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = k_branch_residual_unit(body, filter_list[i+1], (1,1), True,num_branch, name='stage%d_unit%d' % (i + 1, j + 2),
                                 scalar_gate=scalar_gate,elemwise_gate=elemwise_gate,bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1 = mx.sym.BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = mx.sym.Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = mx.symbol.FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    return mx.symbol.SoftmaxOutput(data=fc1, name='softmax1')


def residual_unit(data, num_filter, stride, dim_match, name, bottle_neck=True, bn_mom=0.1, workspace=256, memonger=False):
    """Return ResNet Unit symbol for building ResNet
    Parameters
    ----------
    data : str
        Input data
    num_filter : int
        Number of output channels
    bnf : int
        Bottle neck channels factor with regard to num_filter
    stride : tuple
        Stride used in convolution
    dim_match : Boolean
        True means channel number between input and output is the same, otherwise means differ
    name : str
        Base name of the operators
    workspace : int
        Workspace used in convolution operator
    """
    if bottle_neck:
        # the same as https://github.com/facebook/fb.resnet.torch#notes, a bit difference with origin paper
        bn1 = BatchNorm(data=data, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn1')
        act1 = Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = Convolution(data=act1, num_filter=int(num_filter*0.25), kernel=(1,1), stride=(1,1), pad=(0,0),
                                   no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = BatchNorm(data=conv1, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn2')
        act2 = Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = Convolution(data=act2, num_filter=int(num_filter*0.25), kernel=(3,3), stride=stride, pad=(1,1),
                                   no_bias=True, workspace=workspace, name=name + '_conv2')
        bn3 = BatchNorm(data=conv2, fix_gamma=False, eps=2e-5, momentum=bn_mom, name=name + '_bn3')
        act3 = Activation(data=bn3, act_type='relu', name=name + '_relu3')
        conv3 = Convolution(data=act3, num_filter=num_filter, kernel=(1,1), stride=(1,1), pad=(0,0), no_bias=True,
                                   workspace=workspace, name=name + '_conv3')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, no_bias=True,
                                            workspace=workspace, name=name+'_sc')
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv3 + shortcut
    else:
        bn1 = BatchNorm(data=data, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn1')
        act1 = Activation(data=bn1, act_type='relu', name=name + '_relu1')
        conv1 = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(3,3), stride=stride, pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv1')
        bn2 = BatchNorm(data=conv1, fix_gamma=False, momentum=bn_mom, eps=2e-5, name=name + '_bn2')
        act2 = Activation(data=bn2, act_type='relu', name=name + '_relu2')
        conv2 = Convolution(data=act2, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1),
                                      no_bias=True, workspace=workspace, name=name + '_conv2')
        if dim_match:
            shortcut = data
        else:
            shortcut = mx.sym.Convolution(data=act1, num_filter=num_filter, kernel=(1,1), stride=stride, name=name+'_sc', no_bias=True,workspace=workspace)
        if memonger:
            shortcut._set_attr(mirror_stage='True')
        return conv2 + shortcut

def resnet(units, num_stages, filter_list, num_classes, image_shape, bottle_neck=True, bn_mom=0.1, workspace=256, memonger=False):
    """Return ResNet symbol of
    Parameters
    ----------
    units : list
        Number of units in each stage
    num_stages : int
        Number of stage
    filter_list : list
        Channel size of each stage
    num_classes : int
        Ouput size of symbol
    dataset : str
        Dataset type, only cifar10 and imagenet supports
    workspace : int
        Workspace used in convolution operator
    """
    num_unit = len(units)
    param_dictionary()
    assert(num_unit == num_stages)
    data = mx.sym.Variable(name='data')
    data = mx.sym.identity(data=data, name='id')
    data = BatchNorm(data=data, fix_gamma=True, eps=2e-5, momentum=bn_mom, name='bn_data')
    (nchannel, height, width) = image_shape
    if height <= 32:            # such as cifar10
        body = Convolution(data=data, num_filter=filter_list[0], kernel=(3, 3), stride=(1,1), pad=(1, 1),
                                  no_bias=True, name="conv0", workspace=workspace)
    else:                       # often expected to be 224 such as imagenet
        body = Convolution(data=data, num_filter=filter_list[0], kernel=(7, 7), stride=(2,2), pad=(3, 3),
                                  no_bias=True, name="conv0", workspace=workspace)
        body = BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn0')
        body = Activation(data=body, act_type='relu', name='relu0')
        body = mx.symbol.Pooling(data=body, kernel=(3, 3), stride=(2,2), pad=(1,1), pool_type='max')

    for i in range(num_stages):
        body = residual_unit(body, filter_list[i+1], (1 if i==0 else 2, 1 if i==0 else 2), False,
                             name='stage%d_unit%d' % (i + 1, 1), bottle_neck=bottle_neck, workspace=workspace,
                             memonger=memonger)
        for j in range(units[i]-1):
            body = residual_unit(body, filter_list[i+1], (1,1), True, name='stage%d_unit%d' % (i + 1, j + 2),
                                 bottle_neck=bottle_neck, workspace=workspace, memonger=memonger)
    bn1 = BatchNorm(data=body, fix_gamma=False, eps=2e-5, momentum=bn_mom, name='bn1')
    relu1 = Activation(data=bn1, act_type='relu', name='relu1')
    # Although kernel is not used here when global_pool=True, we should put one
    pool1 = mx.symbol.Pooling(data=relu1, global_pool=True, kernel=(7, 7), pool_type='avg', name='pool1')
    flat = mx.symbol.Flatten(data=pool1)
    fc1 = FullyConnected(data=flat, num_hidden=num_classes, name='fc1')
    return mx.symbol.SoftmaxOutput(data=fc1, name='softmax1')

def get_unrolled_symbol(num_classes, num_layers, image_shape,batch_size,wide=False,num_branch=1,num_step=2, conv_workspace=256, **kwargs):
    #print scalar_gate
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    image_shape = [int(l) for l in image_shape.split(',')]
    (nchannel, height, width) = image_shape
    if height <= 32:
        num_stages = 3
        if (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            if wide:
                filter_list=[16, 160, 320, 640]
            else:    
                filter_list = [16, 16, 32, 64]
            bottle_neck = False
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
 
    if num_branch>1:
        return dyn_k_branch_resnet(units = units,
                          num_stages  = num_stages,
                          filter_list = filter_list,
                          num_classes = num_classes,
                          image_shape = image_shape,
                          num_branch = num_branch,
                             num_step=num_step,
                          bottle_neck = bottle_neck,
                          workspace   = conv_workspace)
    else: 
        return dyn_hypergated_resnet(units = units,
                          num_stages  = num_stages,
                          filter_list = filter_list,
                          num_classes = num_classes,
                          image_shape = image_shape,
                          batch_size = batch_size,           
                          num_branch = num_branch,
                             num_step=num_step,
                                     
                          bottle_neck = bottle_neck,
                          workspace   = conv_workspace)

                          
                          
                          
                          
def get_symbol(num_classes, num_layers, image_shape,wide=False,hyper=False,num_branch=1, conv_workspace=256, **kwargs):
    #print scalar_gate
    """
    Adapted from https://github.com/tornadomeet/ResNet/blob/master/train_resnet.py
    Original author Wei Wu
    """
    image_shape = [int(l) for l in image_shape.split(',')]
    (nchannel, height, width) = image_shape
    if height <= 32:
        num_stages = 3
        if (num_layers-2) % 9 == 0 and num_layers >= 164:
            per_unit = [(num_layers-2)//9]
            filter_list = [16, 64, 128, 256]
            bottle_neck = True
        elif (num_layers-2) % 6 == 0 and num_layers < 164:
            per_unit = [(num_layers-2)//6]
            if wide:
                filter_list=[16, 160, 320, 640]
            else:    
                filter_list = [16, 16, 32, 64]
            bottle_neck = False    
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
        units = per_unit * num_stages
    else:
        if num_layers >= 50:
            filter_list = [64, 256, 512, 1024, 2048]
            bottle_neck = True
        else:
            filter_list = [64, 64, 128, 256, 512]
            bottle_neck = False
        num_stages = 4
        if num_layers == 18:
            units = [2, 2, 2, 2]
        elif num_layers == 34:
            units = [3, 4, 6, 3]
        elif num_layers == 50:
            units = [3, 4, 6, 3]
        elif num_layers == 101:
            units = [3, 4, 23, 3]
        elif num_layers == 152:
            units = [3, 8, 36, 3]
        elif num_layers == 200:
            units = [3, 24, 36, 3]
        elif num_layers == 269:
            units = [3, 30, 48, 8]
        else:
            raise ValueError("no experiments done on num_layers {}, you can do it yourself".format(num_layers))
            
    if num_branch >1:
        if hyper:
            return hyper_k_branch_resnet(units = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  image_shape = image_shape,
                  scalar_gate=scalar_gate,
                  elemwise_gate=elemwise_gate,             
                            num_branch = num_branch,   
                  bottle_neck = bottle_neck,
                  workspace   = conv_workspace)
        else:    
            return k_branch_resnet(units = units,
                      num_stages  = num_stages,
                      filter_list = filter_list,
                      num_classes = num_classes,
                      image_shape = image_shape,
                      scalar_gate=scalar_gate,
                      elemwise_gate=elemwise_gate,             
                                num_branch = num_branch,   
                      bottle_neck = bottle_neck,
                      workspace   = conv_workspace)
    else:
        return resnet(units       = units,
                  num_stages  = num_stages,
                  filter_list = filter_list,
                  num_classes = num_classes,
                  image_shape = image_shape,
                  bottle_neck = bottle_neck,
                  workspace   = conv_workspace)
