from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import random
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Conv1D, Dropout, Lambda,MaxPooling1D
from keras.optimizers import RMSprop
from keras import backend as K
import os
import torch
from keras.regularizers import l2,l1
from sklearn import metrics

from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping

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

def check_file_exist(file_path):
    if os.path.exists(file_path):
        return True
    else:
        return False

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

def load_net_npy_file(npy_file):
    net_feature = 0
    exist_flag=False
    if check_file_exist(npy_file):
        exist_flag=True
        net_feature = np.load(npy_file)
    return net_feature, exist_flag

def get_label(label_name):
    if label_name=="Scaffold":
        return 1
    if label_name=="no-LLPS":
        return 0

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

def load_net_embedding_dataset(npy_feat_path, acc2label_xls):
    access_label_ht=load_acc2label_ht(acc2label_xls)

    X=[]
    Y=[]
    for access in access_label_ht:
        label_name = access_label_ht[access]
        label=get_label(label_name)

        each_npy_file=npy_feat_path+ "/" + access + "_net-feature.npy"
        net_np, npy_exist_flag = load_net_npy_file(each_npy_file)

        if npy_exist_flag:
            X.append(net_np)
            Y.append(label)

    X=np.array(X)
    Y=np.array(Y)
    return X,Y

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

def static_posi_nega(Y):
    posi_count=0
    nega_count=1
    for y in Y:
        if y==1:
            posi_count+=1
        else:
            nega_count+=1
    print("posi_count=",str(posi_count)," nega_count=",str(nega_count))
    return posi_count, nega_count

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    dis= K.sqrt(K.maximum(sum_square, K.epsilon()))
    return dis

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

def create_pairs(x, digit_indices):
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 2
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 2]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]

    return np.array(pairs), np.array(labels)

def compute_acc_auc_mcc(y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    acc=np.mean(pred == y_true)
    auc = metrics.roc_auc_score(y_true, pred)
    mcc = metrics.matthews_corrcoef(y_true, pred)

    return acc,auc,mcc

def get_err_pret_set(esm_pairs_X, y_true, y_pred):
    pred = y_pred.ravel() < 0.5
    err_esm_arry=[]
    err_y_arry=[]
    err_index_arry=[]
    for index in range(0,len(y_true)):
        if pred[index]!=y_true[index]:
            err_index_arry.append(index)
            err_esm_arry.append(esm_pairs_X[index])
            #err_ppi_arry.append(ppi_pairs_X[index])
            err_y_arry.append(y_true[index])

    return err_esm_arry, err_y_arry, err_index_arry

def accuracy(y_true, y_pred):
    return K.mean(K.equal(y_true, K.cast(y_pred < 0.5, y_true.dtype)))

def get_pair_dataset(X, y, class_nums):
    digit_indices = [np.where(y == i)[0] for i in range(class_nums)]
    X_pairs, y_pairs = create_pairs(X, digit_indices)
    return X_pairs, y_pairs

def look_weights(model):
    for i, layer in enumerate(model.layers):
        print(f"Layer {i}: {layer.name}")
        weights = layer.get_weights()
        if i < 18:
            layer.trainable = False
        for j, weight in enumerate(weights):
            print(f"  Weight {j}: {weight.shape}")

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

    train_y_pred = model.predict([esm_train_pairs[:, 0], esm_train_pairs[:, 1]])
    train_acc,train_auc,train_mcc = compute_acc_auc_mcc(train_y, train_y_pred)
    test_y_pred = model.predict([esm_test_pairs[:, 0], esm_test_pairs[:, 1]])
    test_acc,test_auc,test_mcc = compute_acc_auc_mcc(test_y, test_y_pred)
    y_pred_indep = model.predict([esm_indep_pairs[:, 0], esm_indep_pairs[:, 1]])
    indep_acc,indep_auc,indep_mcc = compute_acc_auc_mcc(indep_y, y_pred_indep)

    train_err_esm_X, train_err_y, train_err_index=get_err_pret_set(esm_train_pairs, train_y, train_y_pred)
    test_err_esm_X, test_err_y, test_err_index=get_err_pret_set(esm_test_pairs, test_y, test_y_pred)
    indep_err_esm_X, indep_err_y, indep_err_index=get_err_pret_set(esm_indep_pairs, indep_y, y_pred_indep)

    print(model_name, 'Train acc:', train_acc, 'Test acc:',test_acc, 'Indep acc:',indep_acc)
    print(model_name, 'Train mcc:', train_mcc, 'Test mcc:',test_mcc, 'Indep mcc:',indep_mcc)

    return model, train_acc, train_auc, train_mcc, test_acc,test_auc,test_mcc,indep_acc,indep_auc,indep_mcc, \
           train_err_index, test_err_index, indep_err_index

root="/root/LLPS"

train_acc2label_xls=root+"/posiNega_trainTest.xls"
indep_acc2label_xls=root+"/posiNega_validation.xls"

esm_root=root+"/esm2-feature"
result_root=root+"/esm+ppi-param"
model_name_arry=["uniprot_human_t6-8M_pt","uniprot_human_t12-35M_pt","uniprot_human_t30_pt",
                 "uniprot_human_t33_pt"]
t_length_arry=[6,12,30,33]
esm_feat_arry=[320,480,640,1280]
esm_index=2

esm_name=model_name_arry[esm_index]
ppi_feat_size=esm_feat_arry[esm_index]
t_length=t_length_arry[esm_index]
esm_feat_size=esm_feat_arry[esm_index]

esm_file_path=esm_root+"/"+esm_name
device="cpu"

weight="w0.2"
embedding_path=["deepNF","node2vec"]
embedding_index=1
merge_name="channel_esm+"+embedding_path[embedding_index]+"-ppi-"+str(esm_feat_arry[esm_index])

#hpc
npy_feat_path=root+"/ppi_embedding"

num_classes = 2
epochs = 80
batch_size=16

esm_X,ppi_X,y=get_merge_esm2_net_dataset(esm_file_path, npy_feat_path, train_acc2label_xls, device, t_length)
indep_esm_X,indep_ppi_X,indep_y=get_merge_esm2_net_dataset(esm_file_path, npy_feat_path, indep_acc2label_xls, device, t_length)

esm_X = esm_X.reshape(-1, esm_feat_size, 1).astype('float32')
ppi_X = ppi_X.reshape(-1, esm_feat_size, 1).astype('float32')
X = np.concatenate((esm_X, ppi_X), axis=-1)

esm_input_shape = esm_X.shape[1:]

merge_input_shape = X.shape[1:]
indep_X1 = indep_esm_X.reshape(-1, esm_feat_size, 1).astype('float32')
indep_X2 = indep_ppi_X.reshape(-1, esm_feat_size, 1).astype('float32')
indep_X = np.concatenate((indep_X1, indep_X2), axis=-1)

cnn_param = {'conv_filters':5 , 'conv_kernel': 4, 'pool_size': 3, 'dense1':32, 'dense2':4}

conv_filters1=cnn_param['conv_filters']
conv_kernel1=cnn_param['conv_kernel']
pool_size=cnn_param['pool_size']
dense1=cnn_param['dense1']
dense2 = cnn_param['dense2']

esm_base_network = create_esm_base_network(ppi_feat_size,conv_filters1, conv_kernel1,
                                   pool_size,conv_filters1, conv_kernel1,dense1,dense2)

merge_base_network = create_merge_base_network(ppi_feat_size, conv_filters1, conv_kernel1,
                                           pool_size, conv_filters1, conv_kernel1, dense1, dense2)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
kf = kf.split(esm_X)
for i, (train_fold, validate_fold) in enumerate(kf):

    X_pairs_train, y_pairs_train=get_pair_dataset(X[train_fold], y[train_fold], num_classes)
    X_pairs_test, y_pairs_test=get_pair_dataset(X[validate_fold], y[validate_fold], num_classes)
    X_pairs_indep, y_pairs_indep=get_pair_dataset(indep_X, indep_y, num_classes)

    esm_pairs_train, y_pairs_train = get_pair_dataset(esm_X[train_fold], y[train_fold], num_classes)
    esm_pairs_test, y_pairs_test = get_pair_dataset(esm_X[validate_fold], y[validate_fold], num_classes)
    esm_pairs_indep, y_pairs_indep = get_pair_dataset(indep_X1, indep_y, num_classes)

    esm_model_h5 = result_root + "/snLLPS.h5"

    esm_model, esm_train_acc, esm_train_auc, esm_train_mcc, esm_test_acc, esm_test_auc, esm_test_mcc, \
    esm_indep_acc, esm_indep_auc, esm_indep_mcc, \
    esm_train_err_index_arry, esm_test_err_index_arry, esm_indep_err_index_arry\
        = siamese_net_LLPS("esm-only",esm_model_h5, esm_base_network, esm_input_shape,
                    esm_pairs_train, y_pairs_train, esm_pairs_test, y_pairs_test,
                    esm_pairs_indep, y_pairs_indep, epochs, batch_size)

    merge_model_h5 = result_root + "/mssnLLPS.h5"
    merge_model, train_acc, train_auc, train_mcc, test_acc, test_auc, test_mcc, indep_acc, indep_auc, indep_mcc, \
    train_err_index_arry, test_err_index_arry, indep_err_index_arry\
         = siamese_net_LLPS("multi_channel",merge_model_h5, merge_base_network, merge_input_shape,
                                  X_pairs_train, y_pairs_train, X_pairs_test, y_pairs_test,
                                  X_pairs_indep, y_pairs_indep, epochs, batch_size)
