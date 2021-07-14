from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.imagenet_utils import preprocess_input


def load_data(train_dir,
              val_dir,
              test_dir,
              batch_size,
              target_size=(224, 224),
              class_mode='binary'):
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    validation_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input)

    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=target_size,
                                                        batch_size=batch_size,
                                                        class_mode=class_mode)

    validation_generator = validation_datagen.flow_from_directory(val_dir,
                                                                  shuffle=False,
                                                                  target_size=target_size,
                                                                  batch_size=batch_size,
                                                                  class_mode=class_mode)

    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=target_size,
                                                      batch_size=batch_size,
                                                      interpolation='bicubic',
                                                      class_mode=class_mode,
                                                      shuffle=False)
    return train_generator, validation_generator, test_generator
