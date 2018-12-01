from keras.models import Model
from keras.layers import Input, Dense, Embedding, concatenate
from keras.layers import CuDNNGRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D, Conv1D
from keras.layers import Add, BatchNormalization, Activation, CuDNNLSTM, Dropout
from keras.layers import *
from keras.models import *
from keras import backend as K
from keras.engine.topology import Layer, InputSpec
from keras import initializers
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import gc
from sklearn import metrics


maxlen = 70
max_features = 50000
embed_size = 300


def attention_3d_block(inputs):  # inouts shape=[?, ? , 256]
    # inputs.shape = (batch_size, time_steps, input_dim)
    TIME_STEPS = inputs.shape[1].value  # None
    SINGLE_ATTENTION_VECTOR = False

    input_dim = int(inputs.shape[2])  # 256
    a = Permute((2, 1))(inputs)  # shape=[?, 256, 70]
    a = Reshape((input_dim, TIME_STEPS))(a)  # this line is not useful. It's just to know which dimension is what.
    a = Dense(TIME_STEPS, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

class AttLayer(Layer):
    def __init__(self, attention_dim):
        self.init = initializers.get('normal')
        self.supports_masking = True
        self.attention_dim = attention_dim
        super(AttLayer, self).__init__()

    def build(self, input_shape):
        assert len(input_shape) == 3
        self.W = K.variable(self.init((input_shape[-1], self.attention_dim)))
        self.b = K.variable(self.init((self.attention_dim, )))
        self.u = K.variable(self.init((self.attention_dim, 1)))
        self.trainable_weights = [self.W, self.b, self.u]
        super(AttLayer, self).build(input_shape)

    def compute_mask(self, inputs, mask=None):
        return mask

    def call(self, x, mask=None):
        # size of x :[batch_size, sel_len, attention_dim]
        # size of u :[batch_size, attention_dim]
        # uit = tanh(xW+b)
        uit = K.tanh(K.bias_add(K.dot(x, self.W), self.b))
        ait = K.dot(uit, self.u)
        ait = K.squeeze(ait, -1)

        ait = K.exp(ait)

        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            ait *= K.cast(mask, K.floatx())
        ait /= K.cast(K.sum(ait, axis=1, keepdims=True) + K.epsilon(), K.floatx())
        ait = K.expand_dims(ait)
        weighted_input = x * ait
        output = K.sum(weighted_input, axis=1)

        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


def get_embedding_matrix(tokenizer):

    EMBEDDING_FILE = '/home/pczero/embeddings/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word,*arr):
        return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean= all_embs.mean()  # 总体的平均值，一个数值
    emb_std = all_embs.std()
    embed_size = all_embs.shape[1]

    word_index = tokenizer.word_index  # X_train与X_test中所有单词的编号,编号按频率由高到低
    nb_words = min(max_features, len(word_index))
    embedding_matrix_1 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix_1[i] = embedding_vector

    del embeddings_index; gc.collect()  # 两句结合清理内存



    EMBEDDING_FILE = '/home/pczero/embeddings/embeddings/wiki-news-300d-1M/wiki-news-300d-1M.vec'
    def get_coefs(word,*arr):
        return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE) if len(o)>100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix_2 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix_2[i] = embedding_vector
    del embeddings_index; gc.collect()



    EMBEDDING_FILE = '/home/pczero/embeddings/embeddings/paragram_300_sl999/paragram_300_sl999.txt'
    def get_coefs(word, *arr):
        return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(
        get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE, encoding="utf8", errors='ignore') if len(o) > 100)

    all_embs = np.stack(embeddings_index.values())
    emb_mean, emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix_3 = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix_3[i] = embedding_vector

    del embeddings_index; gc.collect()



    embedding_matrix = np.concatenate((embedding_matrix_1, embedding_matrix_2, embedding_matrix_3), axis=1)
    del embedding_matrix_1, embedding_matrix_2, embedding_matrix_3
    gc.collect()
    np.shape(embedding_matrix)

    return embedding_matrix



def lstm_model(tokenizer):

    inp = Input(shape=(maxlen,))  # [?, 70]
    embed = Embedding(max_features, embed_size * 3, weights=[get_embedding_matrix(tokenizer)], trainable=False)(inp)
    x = embed  # [?, 70, 900]

    # x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)  # 基于CuDNN的快速LSTM实现，只能在GPU上运行，只能使用tensoflow为后端
    x = Bidirectional(GRU(128, unroll=True, return_sequences=True))(x)  # 修改 x.shape=(?, 70, 256)
    # 可选GRU，LSTM  注意：一定要使用unroll=True
    x = attention_3d_block(x)
    # x = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x)
    x = Bidirectional(GRU(128, unroll=True, return_sequences=True))(x)  # 修改
    x = AttLayer(64)(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    outp = Dense(1, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model

