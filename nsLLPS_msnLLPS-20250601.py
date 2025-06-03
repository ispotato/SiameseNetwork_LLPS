import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import numpy as np
import random
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Conv1D, Dropout, Lambda,MaxPooling1D,BatchNormalization,Multiply,Dot #看名字就知道这是模型中的常见操作，为了简化后续程序，在这里引入，也可以直接在搭建中写keras.layers.Dense()等，注意Lambda层比较特殊，这是自定义操作层，你可随意发挥
from keras.optimizers import RMSprop
from keras import backend as K
import os
import torch
from keras.regularizers import l2,l1
from sklearn import metrics
from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score,recall_score,confusion_matrix
from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping
from keras.models import  load_model
from tensorflow.keras.layers import Layer, Multiply

class DynamicWeightLayer(Layer):
    def __init__(self, **kwargs):
        super(DynamicWeightLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # 初始化可训练权重参数
        self.w_esm = self.add_weight(name='w_esm', shape=(1,), initializer='ones', trainable=True)
        self.w_ppi = self.add_weight(name='w_ppi', shape=(1,), initializer='ones', trainable=True)
        super(DynamicWeightLayer, self).build(input_shape)

    def call(self, inputs):
        distance_esm, distance_ppi = inputs
        # 动态softmax权重（归一化）
        total = tf.abs(self.w_esm) + tf.abs(self.w_ppi) + 1e-6
        weight_esm = tf.abs(self.w_esm) / total
        weight_ppi = tf.abs(self.w_ppi) / total
        return weight_esm * distance_esm + weight_ppi * distance_ppi

def check_file_exist(file_path):
    if os.path.exists(file_path):
        return True
    else:
        return False

def mkdir_result_path(path):
    if os.path.exists(path) and os.path.isdir(path):
        print("result path exist")
    else:
        print("create result path")
        mk_comm = "mkdir " + path
        os.system(mk_comm)

    return

def load_acc2label_ht(acc2label_xls):
    data = np.loadtxt(acc2label_xls, delimiter='\t', dtype=str)
    access_label_ht = {}
    for line_idx in range(1, len(data)):
        access = data[line_idx][0]
        label = data[line_idx][1]
        if label=="Client" or label=="Regulator":
            continue
        access_label_ht[access]=label
    return access_label_ht

def load_esm_pt_file(pt_file, device, t_length):
    esm = 0
    exist_flag=False
    if check_file_exist(pt_file):
        exist_flag=True
        esm = torch.load(pt_file, map_location=device)
        if type(esm) == dict:
            esm = esm['mean_representations'][t_length]
            esm = esm.numpy()
    return esm, exist_flag

def load_esm2_dataset(esm_file_path, acc2label_xls, device, t_length):
    access_label_ht=load_acc2label_ht(acc2label_xls)

    X = []
    Y = []
    for access in access_label_ht:
        label_name = access_label_ht[access]
        label=get_label(label_name)

        each_pt_file = esm_file_path + "/" + access + ".pt"
        esm_np, esm_exist_flag = load_esm_pt_file(each_pt_file, device, t_length)

        if esm_exist_flag:
            X.append(esm_np)
            Y.append(label)

    X=np.array(X)
    Y=np.array(Y)
    return X, Y

def load_net_npy_file(npy_file):#npy
    net_feature = 0
    exist_flag=False
    if check_file_exist(npy_file):
        exist_flag=True
        net_feature = np.load(npy_file)
    return net_feature, exist_flag

def get_merge_esm2_net_dataset(esm_file_path, npy_feat_path, acc2label_xls, device, t_length):
    access_label_ht=load_acc2label_ht(acc2label_xls)

    esm_X=[]
    ppi_X=[]
    Y=[]
    for access in access_label_ht:
        label_name = access_label_ht[access]
        label=get_label(label_name)

        each_pt_file = esm_file_path + "/" + access + ".pt"
        esm_np, esm_exist_flag = load_esm_pt_file(each_pt_file, device, t_length)

        each_npy_file=npy_feat_path+ "/" + access + "_net-feature.npy"
        net_np, npy_exist_flag = load_net_npy_file(each_npy_file)

        if npy_exist_flag and esm_exist_flag:
            esm_X.append(esm_np)
            ppi_X.append(net_np)

            Y.append(label)

    esm_X=np.array(esm_X)
    ppi_X = np.array(ppi_X)
    Y=np.array(Y)
    return esm_X,ppi_X,Y

def get_ppi_dataset(ppi_feat_path, acc2label_xls):
    access_label_ht=load_acc2label_ht(acc2label_xls)

    ppi_X=[]
    Y=[]
    for access in access_label_ht:
        label_name = access_label_ht[access]
        label=get_label(label_name)

        each_npy_file=ppi_feat_path+ "/" + access + "_net-feature.npy"
        ppi_np, npy_exist_flag = load_net_npy_file(each_npy_file)

        if npy_exist_flag:
            ppi_X.append(ppi_np)
            Y.append(label)

    ppi_X = np.array(ppi_X)
    Y=np.array(Y)
    return ppi_X, Y

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    dist = K.mean(y_true * square_pred + (1 - y_true) * margin_square)
    return dist

def contrastive_loss_with_prob(y_true, y_pred_prob):
    margin = 0.3
    epsilon = 1e-7

    y_pred_distance = -K.log(y_pred_prob / (1 - y_pred_prob + epsilon))

    square_pred = K.square(y_pred_distance)
    margin_square = K.square(K.maximum(margin - y_pred_distance, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def create_pairs_enhanced(x, digit_indices, num_repeats=3, max_offset=5):
    pairs = []
    labels = []
    num_classes = len(digit_indices)

    for _ in range(num_repeats):
        n = min([len(digit_indices[d]) for d in range(num_classes)]) - max_offset

        for d in range(num_classes):
            for i in range(n):
                offset = random.randint(1, max_offset)
                z1, z2 = digit_indices[d][i], digit_indices[d][i + offset]
                pairs.append([x[z1], x[z2]])
                dn = (d + random.randint(1, num_classes - 1)) % num_classes
                z1, z2 = digit_indices[d][i], digit_indices[dn][i]
                pairs.append([x[z1], x[z2]])
                labels += [1, 0]  #
    return np.array(pairs), np.array(labels)

def get_pair_dataset(X, y, class_nums):
    digit_indices = [np.where(y == i)[0] for i in range(class_nums)]
    X_pairs, y_pairs = create_pairs_enhanced(X, digit_indices)
    return X_pairs, y_pairs

def get_mean5(array):
    mean_value = np.mean(array)
    mean_value5 = round(mean_value, 5)
    return mean_value5

def create_base_network(layer_label, input_dim, dense1, dense2):
    input = Input((input_dim, 1))
    x = Flatten()(input)
    x = Dense(dense1, activation='relu', name=layer_label+"dense1",
              activity_regularizer=l1(10e-6),
              kernel_regularizer=l1(0.001),
              bias_regularizer=l1(0.0001))(x)
    x = Dropout(0.2)(x)
    x = Dense(dense2, activation='relu', name=layer_label+"dense2",
              activity_regularizer=l1(10e-6),
              kernel_regularizer=l1(0.001),
              bias_regularizer=l1(0.0001))(x)
    return Model(input, x)

def load_base_network(layer_label, model_h5, input_dim, dense1, dense2):
    pretrain_siamese_model = load_model(model_h5, custom_objects={'contrastive_loss': contrastive_loss})
    pretrain_siamese_model.summary()
    base_net_layers=pretrain_siamese_model.get_layer(index=2)
    dense1_layer = base_net_layers.get_layer(index=2)
    dense2_layer = base_net_layers.get_layer(index=4)

    flag=True

    input = Input((input_dim, 1))
    x = Flatten(name=layer_label+"Flatten")(input)

    x = Dense(dense1, activation='relu', name=layer_label + "dense1",
              activity_regularizer=l1(10e-6),
              kernel_regularizer=l1(0.001),
              bias_regularizer=l1(0.0001),
              weights=dense1_layer.get_weights(),trainable=flag)(x)

    x = Dropout(0.2)(x)
    x = Dense(dense2, activation='relu', name=layer_label + "dense2",
              activity_regularizer=l1(10e-6),
              kernel_regularizer=l1(0.001),
              bias_regularizer=l1(0.0001),
              weights=dense2_layer.get_weights(),trainable=flag)(x)

    model=Model(input, x)
    return model

def single_siamese_network(esm_base_network, feat_dim,
                           train_pairs, train_y, test_pairs, test_y,
                           indep_pairs, indep_y, epochs, batch_size):

    input_esm_a = Input(shape=(feat_dim,), name='feat1_a')
    input_esm_b = Input(shape=(feat_dim,), name='feat1_b')

    esm_encoded_a = esm_base_network(input_esm_a)
    esm_encoded_b = esm_base_network(input_esm_b)

    distance_esm = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [esm_encoded_a, esm_encoded_b])  # 自定义层计算距离，注意此处API

    model = Model(
        inputs=[input_esm_a, input_esm_b],
        outputs=distance_esm)

    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])

    checkpoint = EarlyStopping(monitor='val_loss',
                               min_delta=0.0001,
                               patience=7,
                               verbose=1,
                               mode='min',
                               restore_best_weights=True)

    history = model.fit([train_pairs[:, 0, :esm_feat_size], train_pairs[:, 1, :esm_feat_size]], train_y,
                        validation_data=([test_pairs[:, 0, :esm_feat_size], test_pairs[:, 1, :esm_feat_size]],
                                         test_y),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[checkpoint],
                        verbose=2)

def multi_siamese_network(feat1_base_network, feat2_base_network, feat1_size, feat2_size,
                           train_pairs, train_y, test_pairs, test_y,
                          indep_pairs, indep_y, epochs, batch_size):

    input_feat1_a = Input(shape=(feat1_size,), name='feat1_a')
    input_feat1_b = Input(shape=(feat1_size,), name='feat1_b')

    feat1_encoded_a = feat1_base_network(input_feat1_a)
    feat1_encoded_b = feat1_base_network(input_feat1_b)

    input_feat2_a = Input(shape=(feat2_size,), name='feat2_a')
    input_feat2_b = Input(shape=(feat2_size,), name='feat2_b')

    feat2_encoded_a = feat2_base_network(input_feat2_a)
    feat2_encoded_b = feat2_base_network(input_feat2_b)

    distance_feat1 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [feat1_encoded_a, feat1_encoded_b])
    distance_feat2 = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)(
        [feat2_encoded_a, feat2_encoded_b])

    distance_feat1 = BatchNormalization()(distance_feat1)
    distance_feat2 = BatchNormalization()(distance_feat2)

    final_distance = DynamicWeightLayer()([distance_feat1, distance_feat2])

    model = Model(
        inputs=[input_feat1_a, input_feat1_b, input_feat2_a, input_feat2_b],
        outputs=final_distance
    )

    rms = RMSprop()
    model.compile(loss=contrastive_loss, optimizer=rms, metrics=[accuracy])  #

    checkpoint = EarlyStopping(monitor='val_loss',
                               min_delta=0.0001,
                               patience=7,
                               verbose=1,
                               mode='min',
                               restore_best_weights=True)

    history = model.fit([train_pairs[:, 0, :feat1_size], train_pairs[:, 1, :feat1_size],
                         train_pairs[:, 0, feat1_size:], train_pairs[:, 1, feat1_size:]], train_y,
                        validation_data=([test_pairs[:, 0, :feat1_size], test_pairs[:, 1, :feat1_size],
                                          test_pairs[:, 0, feat1_size:], test_pairs[:, 1, feat1_size:]],
                                         test_y),
                        epochs=epochs,
                        batch_size=batch_size,
                        callbacks=[checkpoint],
                        verbose=2)


train_acc2label_xls="/sampSet/train-data.xls"
indep_acc2label_xls="/sampSet/val-data.xls"

esm_name_arry=["uniprot_human_t6-8M_pt","uniprot_human_t12-35M_pt","uniprot_human_t30_pt",
                 "uniprot_human_t33_pt"]
t_length_arry=[6,12,30,33]
esm_feat_arry=[320,480,640,1280]
esm_index=1

esm_name=esm_name_arry[esm_index]
t_length=t_length_arry[esm_index]
esm_feat_size=esm_feat_arry[esm_index]

esm_file_path=esm_name
device="cpu"

npy_feat_path="/home/yang/ppi"
ppi_feat_arry=[64,128,256]
ppi_feat_index=1

ppi_feat_size=ppi_feat_arry[ppi_feat_index]
weight="w0.2"

num_classes=2

epochs=80
batch_size=16

esm_X,ppi_X,y=get_merge_esm2_net_dataset(esm_file_path, npy_feat_path, train_acc2label_xls, device, t_length)
X1 = esm_X.reshape(-1, esm_feat_size).astype('float32')
X2 = ppi_X.reshape(-1, ppi_feat_size).astype('float32')
X = np.concatenate((X1, X2), axis=-1)

indep_esm_X,indep_ppi_X,indep_y=get_merge_esm2_net_dataset(esm_file_path, npy_feat_path, indep_acc2label_xls, device, t_length)
indep_esm_X1 = indep_esm_X.reshape(-1, esm_feat_size).astype('float32')
indep_ppi_X2 = indep_ppi_X.reshape(-1, ppi_feat_size).astype('float32')
indep_X = np.concatenate((indep_esm_X, indep_ppi_X), axis=-1)

cnn1 = {'dense1':32, 'dense2':8}

indep_pairs, y_pairs_indep = get_pair_dataset(indep_X, indep_y, num_classes)

dense1=cnn1['dense1']
dense2 = cnn1['dense2']

esm_base_network1 = create_base_network("esm",esm_feat_size,dense1, dense2)
ppi_base_network1 = create_base_network("ppi",ppi_feat_size,dense1, dense2)

esm_base_network2 = create_base_network("esm", esm_feat_size, dense1, dense2)
ppi_base_network2 = create_base_network("ppi", ppi_feat_size, dense1, dense2)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
kf = kf.split(X)
for i, (train_fold, validate_fold) in enumerate(kf):
    train_pairs, y_pairs_train=\
        get_pair_dataset(X[train_fold], y[train_fold], num_classes)

    test_pairs, y_pairs_test = \
        get_pair_dataset(X[validate_fold], y[validate_fold], num_classes)

    single_siamese_network(esm_base_network1, esm_feat_size, train_pairs, y_pairs_train,
                    test_pairs, y_pairs_test, indep_pairs, y_pairs_indep, epochs, batch_size)

    single_siamese_network(ppi_base_network1, ppi_feat_size,train_pairs,y_pairs_train,test_pairs,
                           y_pairs_test, indep_pairs, y_pairs_indep, epochs, batch_size)

    multi_siamese_network(esm_base_network2, ppi_base_network2, esm_feat_size,ppi_feat_size,train_pairs, y_pairs_train,
                         test_pairs, y_pairs_test, indep_pairs, y_pairs_indep, epochs, batch_size)

