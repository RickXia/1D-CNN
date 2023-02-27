#! -*- coding: utf-8 -*-

import numpy as np
import os, glob
import pandas as pd
import json
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import tensorflow.keras.backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

np.random.seed(2018)
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="-1"
physical_devices = tf.config.list_physical_devices('GPU')
for p in physical_devices:
        tf.config.experimental.set_memory_growth(p, True)

# READ DATA
def read_single_csv(input_path):
    import pandas as pd
    df_chunk=pd.read_csv(input_path,chunksize=1000,header=None,index_col=0)
    res_chunk=[]
    for chunk in df_chunk:
        res_chunk.append(chunk)
    res_df=pd.concat(res_chunk)
    return res_df
df = read_single_csv('COD+TUR_all_2560.csv')
X= np.expand_dims(df.values[:,0:2560].astype(float),axis=2)#plus one dimension X.shape=（20000，213,1）
Y=pd.read_csv('label_all_cod.csv')['type'].values

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1000, random_state=0)

from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


def BLOCK(seq, filters):  # 定义网络的Block
    cnn = Conv1D(filters * 2, 3, padding='SAME', dilation_rate=1, activation='relu')(seq)
    cnn = Lambda(lambda x: x[:, :, :filters] + x[:, :, filters:])(cnn)
    cnn = Conv1D(filters * 2, 3, padding='SAME', dilation_rate=2, activation='relu')(cnn)
    cnn = Lambda(lambda x: x[:, :, :filters] + x[:, :, filters:])(cnn)
    cnn = Conv1D(filters * 2, 3, padding='SAME', dilation_rate=4, activation='relu')(cnn)
    cnn = Lambda(lambda x: x[:, :, :filters] + x[:, :, filters:])(cnn)
    if int(seq.shape[-1]) != filters:
        seq = Conv1D(filters, 1, padding='SAME')(seq)
    seq = add([seq, cnn])
    return seq


# CNN MODELING
input_tensor = Input(shape=(2559, 1))
seq = input_tensor

seq = BLOCK(seq, 16)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 16)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 32)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 32)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 64)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 64)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 128)
seq = MaxPooling1D(2)(seq)
seq = BLOCK(seq, 128)
seq = Flatten()(seq)#GlobalAveragePooling1D()
seq = Dropout(0.5, (20, int(seq.shape[1])))(seq)#batch size=20
# seq = Dropout(0.5, (20, int(seq.shape[1]), 1))(seq)#batch size=20
# seq = GlobalMaxPooling1D()(seq)
#seq = Dense(128, activation='relu')(seq)

#output_tensor = Dense(10, activation='softmax')(seq)
output_tensor = Dense(1, activation='linear')(seq)#regression


model = Model(inputs=[input_tensor], outputs=[output_tensor])
plot_model(model, to_file='./model_linear.png', show_shapes=True)
model.summary()


# marco f1 score
# def score_loss(y_true, y_pred):
#     loss = K.epsilon()
#     list_loss=[]
#     for i in np.eye(20):
#         y_true_ = K.constant([list(i)]) * y_true
#         y_pred_ = K.constant([list(i)]) * y_pred
#         loss += 0.2 * K.sum(y_true_ * y_pred_) / K.sum(y_true_ + y_pred_ + K.epsilon())#2/class
#     return - K.log(loss + K.epsilon())
#
#
# # 定义marco f1 score的计算公式
# def score_metric(y_true, y_pred):
#     y_true = K.argmax(y_true)
#     y_pred = K.argmax(y_pred)
#     score = 0.
#     for i in range(20):
#         y_true_ = K.cast(K.equal(y_true, i), 'float32')
#         y_pred_ = K.cast(K.equal(y_pred, i), 'float32')
#         score += 0.2 * K.sum(y_true_ * y_pred_) / K.sum(y_true_ + y_pred_ + K.epsilon())
#     return score


from keras.optimizers import adam_v2
adam1 = adam_v2.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
adam2 = adam_v2.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
adam3 = adam_v2.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

# 自定义度量函数
def coeff_determination(y_true, y_pred):
    SS_res =  K.sum(K.square( y_true-y_pred ))
    SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )



model.compile(loss='mean_squared_error',  #mean_absolute_percentage_error Using the mean absolute percentage error as our loss function indicates
                                          # seeking to minimize the average percentage difference between predicted and actual concentrations.
              optimizer=adam1,
              metrics=[coeff_determination])
              #metrics=['mae','acc'])
              #metrics=[score_metric])

# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model3.png', show_shapes=True)




if __name__ == '__main__':
    from tensorflow.keras.callbacks import ModelCheckpoint
    # save the best model
    #print(D.for_valid())
    logdir = './callbacks'
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    output_model_file = os.path.join(logdir, "regression_best_model.h5")
    checkpoint = ModelCheckpoint(filepath=output_model_file,
                                 monitor='val_loss',
                                 mode='min',
                                 save_best_only=True,
                                 verbose=0)


    #1st train
    history = model.fit(X_train,Y_train,
                        validation_data=(X_test, Y_test),
                        steps_per_epoch=500,
                        epochs=50,
                        batch_size=20,
                        callbacks=[checkpoint])
                                  #callbacks=[evaluator])
    loss1=history.history['loss']
    acc1=history.history['coeff_determination']
    #2nd train
    model.compile(loss='mean_squared_error',
                  optimizer=adam2,
                  metrics=[coeff_determination])

    model.load_weights('./callbacks/regression_best_model.h5')


    history = model.fit(X_train,Y_train,
                        validation_data=(X_test, Y_test),
                                  steps_per_epoch=500,
                                  epochs=50,
                                  batch_size=20,
                                  callbacks=[checkpoint])
    loss1.extend(history.history['loss'])
    acc1.extend(history.history['coeff_determination'])

    # 3rd train
    model.compile(loss='mean_squared_error',
                  optimizer=adam3,
                  metrics=[coeff_determination])

    model.load_weights('./callbacks/regression_best_model.h5')


    history = model.fit(X_train,Y_train,
                        validation_data=(X_test, Y_test),
                                  steps_per_epoch=500,
                                  epochs=50,
                                  batch_size=20,
                                  callbacks=[checkpoint])
    loss1.extend(history.history['loss'])
    acc1.extend(history.history['coeff_determination'])
    #save loss&acc
    import time
    loss1=np.array(loss1).reshape(1,len(loss1))
    acc1=np.array(acc1).reshape(1,len(acc1))
    np_out=np.concatenate([acc1,loss1],axis=0)
    df=pd.DataFrame(np.transpose(np_out),columns=["coeff_determination","loss"])
    df.to_csv("./result/score&loss_%s.csv"% (int(time.time())),index=False)

    # accuracy
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print('%s: %.2f%%' % (model.metrics_names[1], scores[1] * 100))



    #plot predict result
    predicted = model.predict(X_test)
    plt.scatter(Y_test, predicted,s=10)#默认s=20
    x = np.linspace(0, 70, 100)
    y = x
    plt.plot(x, y, color='red', linewidth=1.0, linestyle='--', label='line')
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    plt.legend([ "prediction","y = x"])
    plt.title("deviation between true and prediction")
    plt.xlabel('COD_true')
    plt.ylabel('COD_prediction')
    plt.savefig('result/test_%s.png' % (int(time.time())), dpi=200, bbox_inches='tight', transparent=False)
    # plt.show()

    # MSE
    result = abs(np.mean(predicted - Y_test))
    print("The mean error of linear regression:")
    print(result)
    #csv saved
    d = pd.DataFrame()
    predicted = model.predict(X_test)
    d['TRUE'] = Y_test
    d['test'] = predicted
    print(d)
    d.to_csv(r'result/regression_result_%s.csv' % (int(time.time())), index=None)

