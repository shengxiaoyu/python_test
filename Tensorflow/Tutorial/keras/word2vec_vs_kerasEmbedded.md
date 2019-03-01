<h1>word2vec vs keras.layers.Embedding</h1>


做word2vec+Bi-LSTM+CRF模型，进行事件检测。NLP、tensorflow初学者~头皮发麻刚刚使用实验数据训练完gensim word2vec模型，晚上看tensorflow教程，突然发现有个Embedding层。看着作用也是在将字或者词转换为一个稠密矩阵，有点蒙~那还要专门去训练word2vec干嘛，直接都在网络前加一层embedded层，就可以了啊。网上搜了一下~
<p>参考地址：<a href="https://stats.stackexchange.com/questions/324992/how-the-embedding-layer-is-trained-in-keras-embedding-layer">1</a>和
<a href="https://stats.stackexchange.com/questions/324992/how-the-embedding-layer-is-trained-in-keras-embedding-layer"/>2</a>
</p>
没有完全弄懂embedded层的底层原理，似乎是先知道词汇表大小（所以需要先传入vocab_size），然后构造一个相同大小的向量表，像量表里向量和词汇表里词汇通过索引对应。随机初始化，之后就针对整个模型减少loss，优化参数，这个过程中就会调整向量~~~解释不清。区别是：
**embedded只是为了减少loss产生的，没有语义信息，而word2vec是携带上下文信息的向量，含有语义信息。
**
<p>但是，有推荐使用word2vec训练好的weights来初始化embedded~比如使用gensim word2vec模型初始化embedded：
embedding_model = word2vec.Word2Vec.load(model_name)#加载训练好的模型
#模型中有的单词向量就用该向量表示，没有的则随机初始化一个向量
    embedding_weights = np.array(embedding_model[word] if word in embedding_model else
    np.random.uniform(-0.25, 0.25, embedding_model.vector_size)
    for word in embedding_model.wv.vocab.items())
    weights = np.array([v for v in embedding_weights.values()])
    z = Embedding(
        len(vocabulary_inv),
        embedding_dim,
        input_length=sequence_length,
        weights=[weights]#为embedding层设置初始权重矩阵
        )(model_input)
</p>