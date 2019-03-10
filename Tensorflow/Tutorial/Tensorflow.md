#Eager Execution
    启用：tf.enable_eager_execution()
    借助 Eager Execution，TensorFlow 会立即评估各项操作，并返回具体的值，而不是创建稍后执行的计算图。（用于交互式）

#Tensor
    和Numpy ndarray 相似
    Tensor对象有data type 和 shape属性
    
    和Numpy arrays区别：
        1、GPU和TPU能备份tensors 
        2、tensors是不可变的
    和Numpy转换：
        1、tensorflow的operations自动转换numpy ndarrays 到tensors
        2、numpy的操作自动转换tensors 到ndarrays
        3、tensors通过调用.numpy()方法可以转换为numpy ndarrays(他们会尽可能地共享内存，这样使得转换迅速。但是因为tensors
        可能存储于gpu，numpy存储于主机内存，所以可能会导致gpu内存copy到主机内存)
#GPU使用：
    tensorflow会自动决定是否使用gpu还是cpu，若在gpu上，则该operation涉及的tensor都会在gpu上备份
    可以使用tf.test.is_gpu_available()检测是否可使用gpu
    使用tensors.device可以打印该tensors所在设备的全限定名，若在gpu上则以“GPU:<n>”结尾
    也可以通过with tf.device("GPU/CPU:N")指定运行设备
#DataSet:
    Dataset包含一个或多个元素，每个元素结构相同。一个元素又包含：
        一个或多个tensor对象，这些对象称为**组件**，每个组件都有一个tf.DType,表示张量中元素的类型
        一个tf.TensorShape,表示每个元素可能的静态形状
    tf.data.Dataset.from_tensor_slices()将参数tenseor里每一个元素加入dataset
    tf.data.Dataset.from_tensor()是将参数tensor整体作为一个元素加入dataset
    Eager Execution开启后（tf.enable_eager_execution()）可以直接迭代dataset，未开启则需要调用Dataset.make_one_shot_iterator()
#Gradient tapes:
    使用：
    
    x = tf.constant(3.0)
    with tf.GradientTape(persistent=True) as t:
      t.watch(x)
      y = x * x
      z = y * y
    dz_dx = t.gradient(z, x)  # 108.0 (4*x^3 at x = 3)
    dy_dx = t.gradient(y, x)  # 6.0
    del t  # Drop the reference to the tape
   可以自动计算范围内变量对x的求导。
#Variable:
    可变
    
    v = tf.Variable(1.0)
    assert v.numpy() == 1.0

    v.assign(3.0)
    assert v.numpy() == 3.0

    v.assign(tf.square(v))
    assert v.numpy() == 9.0
#keras.layer
    含有预定义的一些网络层结构：Dense(全连接),Conv2D,LSTM等
    自定义layer需要继承keras.layer.Layer,通过重写以下方法定制：
        __init__:初始化
        build:获得输入tensors，可以做更特殊的初始化
        call:逻辑计算
     比如：
     
     class MyDenseLayer(tf.keras.layers.Layer):
          def __init__(self, num_outputs):
            super(MyDenseLayer, self).__init__()
            self.num_outputs = num_outputs
            
          def build(self, input_shape):
            self.kernel = self.add_variable("kernel", 
                                            shape=[int(input_shape[-1]), 
                                                   self.num_outputs])
            
          def call(self, input):
            return tf.matmul(input, self.kernel)
          
        layer = MyDenseLayer(10)
        print(layer(tf.zeros([10, 5])))
        print(layer.trainable_variables)
        
#Estimator
    其中Estimators要求input_fn不接受任何参数，因此常常将可配置的输入函数封装到带预期签名的对象中。如：
        
        import functools
        train_inpf = functools.partial(census_dataset.input_fn, train_file, num_epochs=2, shuffle=True, batch_size=64)
        test_inpf = functools.partial(census_dataset.input_fn, test_file, num_epochs=1, shuffle=False, batch_size=64)