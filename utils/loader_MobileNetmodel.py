from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Flatten, Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras import applications, optimizers
from tensorflow.keras.applications import MobileNet


def create_modelA(base_model):
    model = Sequential()
    model.add(base_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(0.5))  # 丟棄法
    model.add(Dense(1, activation='sigmoid'))
    base_model.trainable = False
    return model


def create_modelB(base_model):
    # ---- 建立分類模型 ---- #
    model = Sequential()
    model.add(base_model)    # 將模型做為一層
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(0.5))  # 丟棄法
    model.add(Dense(1, activation='sigmoid'))
    #for layer in base_model.layers:
    #layer.trainable = False
    base_model.trainable = False     # 凍結權重
    return model


def create_modelC(base_model):
    # ---- 建立分類模型 ---- #
    model = Sequential()
    model.add(base_model)    # 將模型做為一層
    model.add(Flatten())
    model.add(Dense(1024, activation='relu'))
    #model.add(Dropout(0.5))  # 丟棄法
    model.add(Dense(1, activation='sigmoid'))
    #for layer in base_model.layers:
    #layer.trainable = False
    return model


def create_MobileNet(optimizer,
                     loss='binary_crossentropy',
                     metrics=['accuracy']):
    # 建立卷積基底
    base_model = MobileNet(include_top=False,
                           weights='imagenet',
                           input_shape=(224, 224, 3))
    model = create_modelA(base_model)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    return model


def create_MobileNetB(optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy']):
    # 建立卷積基底
    base_model = MobileNet(include_top=False,
                           weights='imagenet',
                           input_shape=(224, 224, 3))

    model = create_modelB(base_model)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    return model


def create_MobileNetC(optimizer,
                      loss='binary_crossentropy',
                      metrics=['accuracy']):
    base_model = MobileNet(include_top=False,
                           weights='imagenet',
                           input_shape=(224, 224, 3))

    # 解凍有節點層
    unfreeze = ['conv_dw_13', 'conv_dw_13_bn', 'conv_pw_13', 'conv_pw_13_bn'] # 最後 4 層的名稱

    for layer in base_model.layers:
      if layer.name in unfreeze:
        layer.trainable = True  # 解凍
      else:
        layer.trainable = False  # 其他凍結權重

    #for layer in base_model.layers[-2:]:
        #layer.trainable = True

    model = create_modelC(base_model)
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
    return model
