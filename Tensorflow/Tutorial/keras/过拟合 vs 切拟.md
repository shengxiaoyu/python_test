容量：模型可学习参数的数量
容量大->过拟合
容量小->切拟合

您可能熟悉奥卡姆剃刀定律：如果对于同一现象有两种解释，最可能正确的解释是“最简单”的解释，即做出最少
量假设的解释。这也适用于神经网络学习的模型：给定一些训练数据和一个网络架构，有多组权重值（多个模型）
可以解释数据，而简单模型比复杂模型更不容易过拟合。

在这种情况下，“简单模型”是一种参数值分布的熵较低的模型（或者具有较少参数的模型，如我们在上面的部分
中所见）。因此，要缓解过拟合：
    1、获取更多训练数据。
    2、降低网络容量。
    3、限制网络的复杂性，具体方法是强制要求其权重仅采用较小的值，
        使权重值的分布更“规则”。这称为“权重正则化”，通过向网络的损失函数添加与权重较大相关的代价来实现。
        这个代价分为两种类型：
            L1 正则化，其中所添加的代价与权重系数的绝对值（即所谓的权重“L1 范数”）成正比。
            L2 正则化，其中所添加的代价与权重系数值的平方（即所谓的权重“L2 范数”）成正比。
            L2 正则化在神经网络领域也称为权重衰减。不要因为名称不同而感到困惑：从数学角度来讲，
            权重衰减与 L2 正则化完全相同。
         （就是在训练时，会把0.001 * 权重系数**2添加到总损失中，这样训练时的损失就会增大）
    4、添加丢弃层
        丢弃（应用于某个层）是指在训练期间随机“丢弃”（即设置为 0）该层的多个输出特征。假设某个指定的层通
        常会在训练期间针对给定的输入样本返回一个向量 [0.2, 0.5, 1.3, 0.8, 1.1]；在应用丢弃后，此向量将随机
        分布几个 0 条目，例如 [0, 0.5, 1.3, 0, 1.1]。“丢弃率”指变为 0 的特征所占的比例，通常设置在 0.2 和
         0.5 之间。在测试时，网络不会丢弃任何单元，而是将层的输出值按等同于丢弃率的比例进行缩减，以便平衡以
         下事实：测试时的活跃单元数大于训练时的活跃单元数。