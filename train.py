#!

from keras.models import Model
from keras.applications import resnet50
from keras.applications import ResNet50
from keras.applications import ResNet50V2
from keras.applications import vgg16
from keras.applications import VGG16
from keras.applications import vgg19
from keras.applications import VGG19
from keras.applications import inception_v3
from keras.applications import InceptionV3
from keras.applications import densenet
from keras.applications import DenseNet121, DenseNet201

from keras.preprocessing import image
from keras.layers import Input, GlobalAveragePooling2D, Dense, Dropout, Flatten, BatchNormalization
from keras.optimizers import SGD
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
import keras

import os
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import sys
import numpy as np
import shutil
import pandas as pd

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

class TscDataset:
    IMG_W, IMG_H, IMG_CH = (224, 224, 3)

    CLASSES = [
        '00000',
        '00001',
        '00002',
        '00003',
        '00004',
        '00005',
        '00006',
        '00007',
        '00008',
        '00009',
        '00010',
        '00011',
        '00012',
        '00013',
        '00014',
        '00015',
        '00016',
        '00017',
        '00018',
        '00019',
        '00020',
        '00021',
        '00022',
        '00023',
        '00024',
        '00025',
        '00026',
        '00027',
        '00028',
        '00029',
        '00030',
        '00031',
        '00032',
        '00033',
        '00034',
        '00035',
        '00036',
        '00037',
        '00038',
        '00039',
        '00040',
        '00041',
        '00042',
    ]

    # sh: find ./ -name "*.ppm" | wc -l
    SAMPLES = 39209

    # sh: for cate in $(ls ./); do echo "$cate: $(ll ./$cate/*.ppm | wc -l),"; done
    CLASS_WEIGHT = {
        0: SAMPLES/210.,
        1: SAMPLES/2220.,
        2: SAMPLES/2250.,
        3: SAMPLES/1410.,
        4: SAMPLES/1980.,
        5: SAMPLES/1860.,
        6: SAMPLES/420.,
        7: SAMPLES/1440.,
        8: SAMPLES/1410.,
        9: SAMPLES/1470.,
        10: SAMPLES/2010.,
        11: SAMPLES/1320.,
        12: SAMPLES/2100.,
        13: SAMPLES/2160.,
        14: SAMPLES/780.,
        15: SAMPLES/630.,
        16: SAMPLES/420.,
        17: SAMPLES/1110.,
        18: SAMPLES/1200.,
        19: SAMPLES/210.,
        20: SAMPLES/360.,
        21: SAMPLES/330.,
        22: SAMPLES/390.,
        23: SAMPLES/510.,
        24: SAMPLES/270.,
        25: SAMPLES/1500.,
        26: SAMPLES/600.,
        27: SAMPLES/240.,
        28: SAMPLES/540.,
        29: SAMPLES/270.,
        30: SAMPLES/450.,
        31: SAMPLES/780.,
        32: SAMPLES/240.,
        33: SAMPLES/689.,
        34: SAMPLES/420.,
        35: SAMPLES/1200.,
        36: SAMPLES/390.,
        37: SAMPLES/210.,
        38: SAMPLES/2070.,
        39: SAMPLES/300.,
        40: SAMPLES/360.,
        41: SAMPLES/240.,
        42: SAMPLES/240.,
    }

    def __init__(self):
        pass

    def get_train_dataset(self):
        pass

    def get_test_dataset(self):
        pass

    def combat_imbalance(self):
        '''
        To combat the imbalance of training dataset.
        refer to: https://machinelearningmastery.com/tactics-to-combat-imbalanced-classes-in-your-machine-learning-dataset/
        :return:
        '''
        pass

    def crop_img(self, src_path, dst_path, box):
        img = pil_image.open(src_path)
        img = img.crop(box)
        img.save(dst_path)

    def crop_train(self):
        train_data_dir = os.path.join(os.getcwd(), 'dataset', 'train')
        cropped_data_dir = os.path.join(os.getcwd(), 'dataset', 'cropped')

        cate_list = os.listdir(train_data_dir)

        for cate in cate_list:
            train_cate_dir = os.path.join(train_data_dir, cate)
            cropped_cate_dir = os.path.join(cropped_data_dir, cate)

            if not os.path.exists(cropped_cate_dir) or not os.path.isdir(cropped_cate_dir):
                os.makedirs(cropped_cate_dir)

            csv_file_path = os.path.join(train_cate_dir, 'GT-{}.csv'.format(cate))
            if not os.path.isfile(csv_file_path):
                print('{} not exist'.format(csv_file_path))
            else:
                imgs = pd.read_csv(csv_file_path, sep=';')
                for idx, img in imgs.iterrows():
                    src_path = os.path.join(train_cate_dir, img['Filename'])
                    dst_path = os.path.join(cropped_cate_dir, img['Filename'])
                    crop_box = (img['Roi.X1'], img['Roi.Y1'], img['Roi.X2'], img['Roi.Y2'])
                    self.crop_img(src_path, dst_path, crop_box)

class TscModels:
    def __init__(self, model_name=''):
        self.__model_name = model_name
        self.__model_weight_name = '{}_weight.h5'.format(self.__model_name)
        self.__model_weight_best_name = '{}_weight_best.h5'.format(self.__model_name)
        self.__session = '{}_{}'.format(self.__model_name, datetime.datetime.now().strftime('%Y%m%d%H%M%S'))

        self.__history = None
        self.__model = None

        # self.__rescale = None
        self.__rescale = 1. / 255

    @property
    def model(self):
        return self.__model

    def init_model(self):
        self.__model = self.get_model(self.get_weight())

    def get_model(self, weight=None):
        raise Exception('Please initiate model')

    def get_weight(self):
        weight_file_path = os.path.join(os.getcwd(), self.__model_weight_name)
        if not os.path.exists(weight_file_path):
            weight_file_path = None

        if weight_file_path:
            print('Try to load weight from {}'.format(weight_file_path))
        else:
            print('Init model without weight')

        return weight_file_path

    def show(self):
        model_picture = os.path.join(os.getcwd(), '{}_model.png'.format(self.__model_name))
        if not os.path.exists(model_picture):
            plot_model(self.__model,
                       '{}_model.png'.format(self.__model_name),
                       show_shapes=True,
                       expand_nested=True)

        self.__model.summary()
        model_file_name = '{}_model.h5'.format(self.__model_name)
        if not os.path.exists(model_file_name):
            self.__model.save(model_file_name, overwrite=False, include_optimizer=False)

    def save(self):
        self.__model.save_weights(self.__model_weight_name)
        shutil.copy(self.__model_weight_name, '{}_weight.h5'.format(self.__session))

    def load(self):
        pass

    def train(self, epocks=1, batch_size=8, train_dir='dataset/over_sampling/train', validation_dir='dataset/over_sampling/validation'):
        print('compile model')
        '''
        SGD:
            lr, 0.0001 --> 0.001 --> 0.01 --> 0.1
            momentum, 0.5 --> 0.7 --> 0.8 --> 0.9
        '''
        # sgd = SGD(lr=0.0001, momentum=0.5, nesterov=True, decay=1e-2/ epocks) # epochs = 4000
        sgd = SGD(lr=0.001, momentum=0.7, nesterov=True, decay=1e-2 / epocks) # epochs = 1000
        # sgd = SGD(lr=0.01, momentum=0.8, nesterov=True, decay=1e-2 / epocks) # 1000
        # sgd = SGD(lr=0.1, momentum=0.9, nesterov=True, decay=1e-2 / epocks)
        # sgd = SGD(lr=0.01, momentum=0.9, nesterov=True, decay=1e-2 / epocks)
        # sgd = SGD()

        self.__model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        print('load training data')
        train_datagen = image.ImageDataGenerator(rescale=self.__rescale,
                                                 # zca_whitening=True,
                                                 rotation_range=5,
                                                 width_shift_range=0.05,
                                                 height_shift_range=0.05,
                                                 zoom_range=0.1,
                                                 # featurewise_center=True,
                                                 # featurewise_std_normalization=True,
                                                 validation_split=0.25,
                                                 fill_mode='constant')
        train_generator = train_datagen.flow_from_directory(train_dir,
                                                            target_size=(TscDataset.IMG_W, TscDataset.IMG_H),
                                                            batch_size=batch_size, #32,
                                                            save_to_dir='data/train_pic',
                                                            class_mode='categorical',
                                                            subset='training')

        valid_generator = train_datagen.flow_from_directory(train_dir,
                                                            target_size=(TscDataset.IMG_W, TscDataset.IMG_H),
                                                            batch_size=batch_size,  # 32,
                                                            save_to_dir='data/valid_pic',
                                                            class_mode='categorical',
                                                            subset='validation')

        print('len(train_generator)={}'.format(len(train_generator)))
        print('len(valid_generator)={}'.format(len(valid_generator)))

        callbacks_list = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=16
            ),
            ModelCheckpoint(
                filepath=self.__model_weight_best_name,
                monitor='val_loss',
                save_best_only=True
            ),
            TensorBoard(log_dir='logs/{}'.format(self.__session),
                        histogram_freq=1,
                        write_grads=True,
                        write_images=True,
                        embeddings_freq=1)
        ]

        print('start training model, epochs={}, batch_size={}'.format(epocks, batch_size))
        self.__history = self.__model.fit_generator(train_generator,
                            steps_per_epoch=len(train_generator)/batch_size,
                            epochs=epocks,
                            validation_data=valid_generator,
                            validation_steps=len(valid_generator)/batch_size,
                            callbacks=callbacks_list,
                            # class_weight=TscDataset.CLASS_WEIGHT,
                            verbose=2
                            )

        # self.__history = self.__model.fit(x=train_generator, y=train_generator,
        #     batch_size=batch_size,
        #     epochs=epocks,
        #
        #         steps_per_epoch=len(train_generator) / batch_size,
        #
        #         validation_data=valid_generator,
        #         validation_steps=len(valid_generator) / batch_size,
        #         callbacks=callbacks_list,
        #         # class_weight=TscDataset.CLASS_WEIGHT,
        #         verbose=2
        # )

    def show_history(self):
        if not self.__history:
            print('Model not fitted by this time')
            return

        acc = self.__history.history['accuracy']
        val_acc = self.__history.history['val_accuracy']

        loss = self.__history.history['loss']
        val_loss = self.__history.history['val_loss']

        epochs_range = range(len(acc))

        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='T-Acc')
        plt.plot(epochs_range, val_acc, label='V-Acc')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='T-Loss')
        plt.plot(epochs_range, val_loss, label='V-Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')

        plt.savefig('{}.png'.format(self.__session))
        plt.show()

    def get_cate(self, img_file_path):
        img = image.load_img(img_file_path, target_size=(TscDataset.IMG_W, TscDataset.IMG_H))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = self.preprocess_input(x)
        if self.__rescale:
            x = x * self.__rescale
        result = self.__model.predict(x)
        # print('result:', result)
        result = self.decode_predictions(result, top=1)[0][0]
        print('Scale: {: >6,.3f}, Predicted: {} -> {} : {: ^6,.3f}   {}'.format(
                                              self.__rescale if self.__rescale else 0.,
                                              img_file_path,
                                              result[0],
                                              result[1],
                                              'V' if result[1] > 0.9 else ' '))
        return result

    def pred_gen_dir(self, pred_dir, batch_size=32):
        pred_datagen = image.ImageDataGenerator(rescale=self.__rescale)
        pred_generator = pred_datagen.flow_from_directory(
            pred_dir,
            target_size=(TscDataset.IMG_W, TscDataset.IMG_H),
            batch_size=batch_size,  # 32,
            save_to_dir='data/test_pic',
            shuffle=False
        )
        print('len(pred_generator)={}'.format(len(pred_generator)))
        result = self.__model.predict(
            x=pred_generator,
            batch_size=batch_size,
            steps=len(pred_generator),
            verbose=1
        )
        print(result)
        return result

    def pred_gen_csv(self, csv_file='test.csv', batch_size=32):
        pred_datagen = image.ImageDataGenerator(rescale=self.__rescale)
        pred_generator = pred_datagen.flow_from_dataframe(
            pd.read_csv(csv_file),
            os.getcwd(),
            x_col='image_path',
            target_size=(TscDataset.IMG_W, TscDataset.IMG_H),
            batch_size=batch_size,  # 32,
            save_to_dir='data/test_pic',
            class_mode=None,
            shuffle=False
        )
        print('len(pred_generator)={}'.format(len(pred_generator)))
        result = self.__model.predict(
            x=pred_generator,
            batch_size=batch_size,
            steps=len(pred_generator),
            verbose=1
        )
        print(result)

        # save prediction result to csv file.
        res = self.decode_predictions(result, top=1)
        res_list = []
        for idx, elem in enumerate(res):
            res_list.append([idx, int(elem[0][0])])

        submission = pd.DataFrame(res_list, columns=('id', 'pred'))
        submission.to_csv('submission.csv', index=False)

        return result

    def test(self):
        test = pd.read_csv('test.csv')
        submission = pd.DataFrame(columns=('id', 'pred'))

        for idx, rec in test.iterrows():
            img_file_path = os.path.join(os.getcwd(), rec['image_path'])
            if os.path.isfile(img_file_path):
                ret = self.get_cate(img_file_path)
                submission.loc[idx] = {'id': idx, 'pred': int(ret[0])}

        submission.to_csv('submission.csv', index=False)

    def predict(self, test_dir='dataset/test/test_images'):
        # TEST_DIR = 'validation/00017'

        img_test_list = os.listdir(test_dir)

        for img_file_name in img_test_list:
            img_file_path = os.path.join(os.getcwd(), test_dir, img_file_name)
            if os.path.isfile(img_file_path):
                if img_file_path.endswith('.ppm'):
                    # print('Predicted: {}'.format(img_file_name))
                    self.get_cate(img_file_path)

    def classification(self, test_dir='dataset/test/test_images'):
        img_test_list = os.listdir(test_dir)

        for img_file_name in img_test_list:
            img_file_path = os.path.join(os.getcwd(), test_dir, img_file_name)
            if os.path.isfile(img_file_path):
                ret = self.get_cate(img_file_path)
                # TODO:
                if ret[1] > 0.9:
                    shutil.move(img_file_path, os.path.join(os.getcwd(), 'dataset/test/validation', ret[0]))

    def preprocess_input(self, x):
        return x

    def decode_predictions(self, preds, top=5, **kwargs):
        """Decodes the prediction of an ImageNet model.

        # Arguments
            preds: Numpy tensor encoding a batch of predictions.
            top: Integer, how many top-guesses to return.

        # Returns
            A list of lists of top class prediction tuples
            `(class_name, class_description, score)`.
            One list of tuples per sample in batch input.

        # Raises
            ValueError: In case of invalid shape of the `pred` array
                (must be 2D).
        """

        if len(preds.shape) != 2 or preds.shape[1] != len(TscDataset.CLASSES):
            raise ValueError('`decode_predictions` expects '
                             'a batch of predictions '
                             '(i.e. a 2D array of shape (samples, 1000)). '
                             'Found array with shape: ' + str(preds.shape))

        results = []
        for pred in preds:
            top_indices = pred.argsort()[-top:][::-1]
            # result = [tuple(TSC.CLASSES[i]) + (pred[i],) for i in top_indices]
            result = [[TscDataset.CLASSES[i], pred[i]] for i in top_indices]
            result.sort(key=lambda x: x[1], reverse=True)
            # result = [TSC.CLASSES[i] for i in top_indices]
            results.append(result)
        return results

class ModelVgg16(TscModels):

    def __init__(self):
        super().__init__(model_name='Vgg16-drop')

    def get_model(self, weight_file_path):
        # return self.default_model(weight_file_path)
        return self.with_dropout_model(weight_file_path)

    def default_model(self, weight_file_path):
        shape = (TscDataset.IMG_W, TscDataset.IMG_H, TscDataset.IMG_CH)
        x = Input(shape)

        return VGG16(weights=weight_file_path, input_tensor=x, classes=len(TscDataset.CLASSES))

    def with_dropout_model(self, weight_file_path):
        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
        ])

        shape = (TscDataset.IMG_W, TscDataset.IMG_H, TscDataset.IMG_CH)

        inputs = Input(shape=shape, name='input')

        x = VGG16(include_top=False, weights=None, input_shape=shape)(inputs)
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.5, name='drp1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5, name='drp2')(x)
        y = Dense(len(TscDataset.CLASSES), activation='softmax', name='predictions')(x)

        model = Model(inputs=inputs, outputs=y, name="Vgg16_drop")
        if weight_file_path is not None:
            model.load_weights(weight_file_path)
        return model

    def preprocess_input(self, x):
        return vgg16.preprocess_input(x)

class ModelVgg19(TscModels):

    def __init__(self):
        super().__init__(model_name='Vgg19')

    def get_model(self, weight_file_path):
        shape = (TscDataset.IMG_W, TscDataset.IMG_H, TscDataset.IMG_CH)
        x = Input(shape)

        return VGG19(weights=weight_file_path, input_tensor=x, classes=len(TscDataset.CLASSES))

    def preprocess_input(self, x):
        return vgg19.preprocess_input(x)

class Resnet50Model(TscModels):

    def __init__(self):
        super().__init__(model_name='Resnet50')
        # super().__init__(model_name='Resnet50_drop')

    def get_model(self, weight_file_path):
        return self.default_model(weight_file_path)
        # return self.with_dropout_model(weight_file_path)

    def default_model(self, weight_file_path):
        shape = (TscDataset.IMG_W, TscDataset.IMG_H, TscDataset.IMG_CH)
        x = Input(shape)

        return ResNet50V2(weights=weight_file_path, input_tensor=x, classes=len(TscDataset.CLASSES))

    def with_dropout_model(self, weight_file_path):

        #---------------------------
        shape = (TscDataset.IMG_W, TscDataset.IMG_H, TscDataset.IMG_CH)

        # K.set_learning_phase(0)
        input = Input(shape, name='input')
        base_model = ResNet50(weights=None, include_top=False, input_shape=shape)
        x = base_model(input)
        # K.set_learning_phase(1)
        x = Flatten(name='flatten')(x)
        x = Dense(2048, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='fc1')(x)
        x = BatchNormalization(name='fc1_bn')(x)
        x = Dense(1024, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01), name='fc2')(x)
        x = BatchNormalization(name='fc2_bn')(x)
        y = Dense(len(TscDataset.CLASSES), activation='softmax', name='prediction')(x)
        model = Model(inputs=input, outputs=y, name='Resnet50_drop')

        if weight_file_path is not None:
            model.load_weights(weight_file_path)

        return model

    def preprocess_input(self, x):
        return resnet50.preprocess_input(x)

class InceptionModel(TscModels):
    def __init__(self):
        super().__init__(model_name='InceptionV3')

    def get_model(self, weight_file_path):
        shape = (TscDataset.IMG_W, TscDataset.IMG_H, TscDataset.IMG_CH)
        x = Input(shape)

        return InceptionV3(weights=weight_file_path, input_tensor=x, classes=len(TscDataset.CLASSES))

    def preprocess_input(self, x):
        return inception_v3.preprocess_input(x)

class DenseNet121Model(TscModels):
    def __init__(self):
        super().__init__(model_name='DenseNet121')

    def get_model(self, weight_file_path):
        shape = (TscDataset.IMG_W, TscDataset.IMG_H, TscDataset.IMG_CH)
        x = Input(shape)

        return DenseNet121(weights=weight_file_path, input_tensor=x, classes=len(TscDataset.CLASSES))

    def preprocess_input(self, x):
        return densenet.preprocess_input(x)

class DenseNet201Model(TscModels):
    def __init__(self):
        super().__init__(model_name='DenseNet201')

    def get_model(self, weight_file_path):
        shape = (TscDataset.IMG_W, TscDataset.IMG_H, TscDataset.IMG_CH)
        x = Input(shape)

        return DenseNet201(weights=weight_file_path, input_tensor=x, classes=len(TscDataset.CLASSES))

    def preprocess_input(self, x):
        return densenet.preprocess_input(x)


if __name__ == '__main__':
    # with tf.device('/cpu:0'):
    tsc = DenseNet201Model()
    if len(sys.argv) > 1 and sys.argv[1] == '-p':
        tsc.init_model()
        tsc.show()
        tsc.predict(sys.argv[2] if len(sys.argv) > 2 else 'dataset/test/test_images')
    elif len(sys.argv) > 1 and sys.argv[1] == '-csv':
        tsc.init_model()
        tsc.show()
        tsc.pred_gen_csv()
    elif len(sys.argv) > 1 and sys.argv[1] == '-tst':
        tsc.init_model()
        tsc.show()
        tsc.test()
    elif len(sys.argv) > 1 and sys.argv[1] == '-c':
        tsc.init_model()
        tsc.show()
        tsc.classification(sys.argv[2] if len(sys.argv) > 2 else 'dataset/test/test_images')
    elif len(sys.argv) > 1 and sys.argv[1] == '-t':
        tsc.init_model()
        tsc.show()
        tsc.train(epocks=int(sys.argv[2]) if len(sys.argv) > 2 else 10, batch_size=32, train_dir='dataset/over_sampling/train')
        tsc.save()
        tsc.show_history()
    else:
        print('{} -t|p|c'.format(sys.argv[0]))