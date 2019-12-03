import cv2
import numpy as np
import os
import random
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import tensorflow.keras.backend as K

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

from sacred import Ingredient

ingredient = Ingredient('dme')

@ingredient.config
def cfg():
    num_examples = 1
    input_size = 128
    batch_size = 16
    numpy_source_path = 'path_to_numpy_matrices'
    dropout_rate = 0.2
    filters = 32
    epochs = 50
    excel_path = 'path_to_excel_data_file'
    model_save_path = 'path_to_saved_models'
    history_save_path = 'path_to_history_images'
    verbose = 2
    patience = 10
    evenly_distributed = False
    # extras
    extras = ['bcva','cstb','mrtb','hba1c']

# skip layer, siehe https://arxiv.org/pdf/1512.03385.pdf, Abbildung 2
@ingredient.capture
def down(x, dropout_rate, filters):
    x1 = Dropout(dropout_rate)(x)
    x1 = Conv2D(filters, kernel_size=(3, 3), padding='same')(x1)

    x2 = Dropout(dropout_rate)(x)
    x2 = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='sigmoid')(x2)
    x2 = BatchNormalization()(x2)

    x2 = Dropout(dropout_rate)(x2)
    x2 = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='sigmoid')(x2)
    x2 = BatchNormalization()(x2)

    x2 = Dropout(dropout_rate)(x2)
    x2 = Conv2D(filters, kernel_size=(3, 3), padding='same', activation='sigmoid')(x2)
    x2 = BatchNormalization()(x2)

    x2 = Dropout(dropout_rate)(x2)
    x2 = Conv2D(filters, kernel_size=(3, 3), padding='same')(x2)

    x = add([x1, x2])
    x = Activation('sigmoid')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D()(x)

    return x

# Modell besteht aus zwei Teilen:
# - encoder, welcher Eingabebilder verarbeitet, wird für alle Bilder für alle Zeitpunkte verwendet
# - pro Zeitpunkt werden die Encodings 
# - von den Encodings werden die jweiligen Minimal- und Maximalwerte genommen, wenn es mehr als 1 Zeitpunkt ist
# - die Encodings pro Zeitpunkt werden konkateniert und in den classifier gegeben
class EyesMonthsClassifier(object) :
    @ingredient.capture
    def __init__(self, num_examples, input_size, extras):
        self.num_examples = num_examples
        self.input_size = input_size
        self.num_extra = len(extras)

    def create_model(self):
        month0 = [Input((self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        month3 = [Input((self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]

        # hier würde man dann zusätzliche Informationen erstellen:
        extra = Input((self.num_extra,))

        encoder = self._create_encoder()

        if self.num_examples > 1:
            classifier_input_size = 1024
            enc0 = [encoder(inp) for inp in month0]
            enc0_min = minimum(enc0)
            enc0_max = maximum(enc0)
            enc0 = concatenate([enc0_min, enc0_max])
            enc3 = [encoder(inp) for inp in month3]
            enc3_min = minimum(enc3)
            enc3_max = maximum(enc3)
            enc3 = concatenate([enc3_min, enc3_max])
        else:
            classifier_input_size = 512
            enc0 = encoder(month0[0])
            enc3 = encoder(month3[0])

        enc = concatenate([enc0, enc3])

        # Extra-Informationen werden dann hier konkateniert:
        enc = concatenate([enc, extra])

        classifier = self._create_classifier(input_size=classifier_input_size + self.num_extra)
        out = classifier(enc)

        model = Model(month0 + month3 + [extra], out, name='eye_sum_model')
        model.compile(loss='mse', optimizer='nadam', metrics=[ca]) # ca

        return model

    def _create_encoder(self):
        inp1 = Input((self.input_size, self.input_size, 1))
        inp2 = Lambda(lambda x: 1-x)(inp1)
        inp = concatenate([inp1, inp2])
        
        var = Dropout(0.1)(inp)

        while var.shape[1] > 2:
            var = down(var)

        print(var.shape)
        var = Flatten()(var)

        var = Dropout(0.1)(var)
        var = Dense(256, activation='sigmoid')(var)

        return Model(inp1, var, name='eye_sum_encoder')

    def _create_classifier(self, input_size):
        inp = Input((input_size,))
        var = BatchNormalization()(inp)
        var = Dropout(0.1)(var)

        for size in [128, 64, 32]:
            var = Dropout(0.1)(var)
            var = Dense(size, activation='sigmoid')(var)
            var = BatchNormalization()(var)

        var = Dense(2, activation='softmax')(var)
        return Model(inp, var, name='eye_sum_classifier')


# Klasse, welche die Numpy-Matrizen lädt
# kann insofern erweitert werden, dass nicht alle Daten sofort geladen werden:
# - zufällige Daten werden geladen
# - jedesmal, wenn ein Beispiel genommen würde ein Counter erhöht werden
# - nach einer bestimmten Anzahl Zugriffen werden neue Daten geladen
class EyesNumpySource(object):
    @ingredient.capture
    def __init__(self, numpy_source_path, input_size):
        self.path = numpy_source_path
        self.files = {}
        self.input_size = input_size
        self.ids = {} # id for targets

        for target in ['dmenr', 'dmer']:
            self._load_files(target)
            print('Loaded %i files for %s and got %i ids!' % (len(self.files[target]), target, len(self.ids[target])))

    def _load_files(self, target):
        self.files[target] = {}
        self.ids[target] = {}

        path = '%s/%s/%s' % (self.path, target, 'train')
        for file_index, file in enumerate(os.listdir(path)):
            file_name = file.split('.')[0]
            self.files[target][file_name] = path + '/' + file
        path = '%s/%s/%s' % (self.path, target, 'test')
        for file_index, file in enumerate(os.listdir(path)):
            file_name = file.split('.')[0]
            self.files[target][file_name] = path + '/' + file

    def _resize(self, mat):
        arr = [self.resize_image(mat[i]) for i in range(mat.shape[0])]
        return arr

    def resize_image(self, img):
        # reshape from (c, w, h) to (w, h, c) for cv2.resize
        reshaped = np.moveaxis(img, 0, -1)
        resized = cv2.resize(reshaped, dsize=(self.input_size, self.input_size))
        # reshape back to (c, w, h)
        return np.moveaxis(resized, -1, 0)

    def _load(self, target, id):
        mat = np.load(self.files[target][id])
        if np.max(mat) > 1:
            mat = mat / 255

        images = self._resize(mat)
        # self.examples[target][id] = images
        return images

    def get_example(self, target, id, evenly_distributed):
        if evenly_distributed:
            # random example
            if target == 'dmer':
                example = self.get_pos_example()
            else:
                example = self.get_neg_example()
            return self.parse_example(target, example)
        else:
            # example by id
            return self.parse_example(target, self._load(target, id))
        # return parse_example(self.examples[target][id])

    def get_pos_example(self):
        return self._load('dmer', random.choice(list(self.files['dmer'].keys())))

    def get_neg_example(self):
        return self._load('dmenr', random.choice(list(self.files['dmenr'].keys())))

    @ingredient.capture
    def parse_example(self, target, example):
        return example[0], example[1]

class EyesMonthsDataGenerator(Sequence):
    @ingredient.capture
    def __init__(self, num_examples, input_size, batch_size, extras, excel_path):
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.input_size = input_size
        self.data_source = EyesNumpySource() # TODO
        self.extras_csv = pd.read_excel(excel_path)
        self.shuffle = True
        self.extras = extras
        self.num_extra = len(extras)

        self.ids = [] # list of all ids
        self.labels = [] # list of labels

    # Gets batch at position index.
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.train_indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        # ids = [self.ids[k] for k in indexes]

        # Generate data
        X, y = self.data_generation(indexes)

        return X, y

    # Number of batch in the Sequence
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.ids) / self.batch_size))
        # return 100
        
    # def on_epoch_end(self):
    #   'Updates indexes after each epoch'
    #   self.ids = np.arange(len(self.ids))
    #   if self.shuffle == True:
    #     np.random.shuffle(self.ids)

    @ingredient.capture
    def data_generation(self, index_list, evenly_distributed):
        'Generates data containing batch_size samples'
        # Initialization        
        M0 = [np.zeros((self.batch_size, self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        M3 = [np.zeros((self.batch_size, self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        Y = np.zeros((self.batch_size, 2))
        EXTRA = [np.zeros((self.batch_size, self.num_extra))]

        dataset_size = len(Y)
        # Generate data
        for counter, idx in enumerate(index_list):
            if counter >= dataset_size:
                break

            id = self.ids[idx]

            # evenly distributed pos and neg examples
            if evenly_distributed:
                condition = counter < self.batch_size // 2
            else: # no distribution, use index_list
                condition = self.labels[idx] == 0

            # label
            if condition:
                Y[counter, 0] = 1
                target = 'dmer'
            else:
                Y[counter, 1] = 1
                target = 'dmenr'
            
            # sample
            p0, p3 = self.data_source.get_example(target, id, evenly_distributed)

            indexes = list(range(self.num_examples))
            random.shuffle(indexes)
            indexes = indexes[:self.num_examples]

            for i in indexes:
                M0[i][counter, :, :, 0] = p0[i]
                M3[i][counter, :, :, 0] = p3[i]
            
            # extras
            extras = []
            for i, extra in enumerate(self.extras):
                if extra == 'bcva':
                    extras.append(self._bcva(id))
                if extra == 'cstb':
                    extras.append(self._cstb(id))
                if extra == 'mrtb':
                    extras.append(self._mrtb(id))
                if extra == 'hba1c':
                    extras.append(self._hba1c(id))
                if extra == 'no-extras':
                    extras.append(0)

            EXTRA[0][counter] = extras

        return M0 + M3 + EXTRA, Y

    def set_train_indexes(self, indexes):
        self.train_indexes = indexes

    # return list of all ids and labels for k-fold split
    def get_all_data(self):
        keys_dmer = self.data_source.files['dmer'].keys()
        keys_dmenr = self.data_source.files['dmenr'].keys()
        # random.shuffle(keys_dmer)
        # random.shuffle(keys_dmenr)

        total_images = len(keys_dmer) + len(keys_dmenr)
        print('Number of total eyes: %i' % total_images)

        for id in keys_dmer:
            self.ids.append(id)
            self.labels.append(0)
        for id in keys_dmenr:
            self.ids.append(id)
            self.labels.append(1)

        return self.ids, self.labels

    # extras
    @ingredient.capture
    def _bcva(self, id):
        return self.get_extra_value('Baseline BCVA (LogMAR)', id)
    # Central subfield Thickness baseline (μm)
    @ingredient.capture
    def _cstb(self, id):
        return self.get_extra_value('Central subfield Thickness baseline (µm)', id) / 1000
    # Maximal retina thickness, baseline (µm)
    @ingredient.capture
    def _mrtb(self, id):
        return self.get_extra_value('Maximal retina thickness, baseline (µm)', id) / 1000
    # HbA1c at DME diagnosis, (%)
    @ingredient.capture
    def _hba1c(self, id):
        return self.get_extra_value('HbA1c at DME diagnosis, (%)', id) / 100
        
    def get_extra_value(self, column_name, id):
        value = self.extras_csv.loc[self.extras_csv['ID'] == id][column_name].values[0]
        if np.isnan(value):
            return 0
        else:
            return value

def ca(y_true, y_pred):
    return 1 - K.mean(K.abs(y_true - y_pred))

def plot(history, history_save_path, id, counter):
    # summarize history for accuracy
    plt.plot(history.history['ca'])
    plt.plot(history.history['val_ca'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('%s%s/accuracy-%i.png' % (history_save_path, id, counter))
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('%s%s/loss-%i.png' % (history_save_path, id, counter))
    plt.clf()

@ingredient.capture
def dme_run(_run, title, epochs, model_save_path, history_save_path, verbose, patience):
    id = _run._id
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    generator = EyesMonthsDataGenerator()
    X, Y = generator.get_all_data()
    counter = 0
    
    # callbacks
    filepath = "%s%s/weights-improvement-{epoch:02d}-{val_ca:.2f}.hdf5" % (history_save_path, id)
    checkpoint = ModelCheckpoint(filepath, monitor='val_ca', verbose=0, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=patience)
    callbacks_list = [checkpoint, es]

    for train_indexes, test_indexes in kfold.split(X, Y): # return lists of indexes
        print('### K-Fold split: ', counter)
        print('train indexes: ', train_indexes)
        print('test indexes: ', test_indexes)
        
        # set train data indexes for generator
        generator.set_train_indexes(train_indexes)

        # create model
        model = EyesMonthsClassifier().create_model()

        # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        # test data
        testX, testY = generator.data_generation(test_indexes)
        # Fit the model
        history = model.fit_generator(generator, validation_data=(testX, testY), epochs=epochs, verbose=verbose, callbacks=callbacks_list)
        # trainX, trainY = generator.create_sample(train_indexes)
        # model.fit(trainX, trainY, batch_size=32, epochs=100)

        # evaluate the model
        scores = model.evaluate(testX, testY, verbose=0)
        print('test loss, test acc:', scores)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        print("average acc: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        
        _run.log_scalar("test.loss", scores[0], counter)
        _run.log_scalar("test.accuracy", scores[1], counter)

        #plot(history, history_save_path, id, counter)
        
        # save model
        # model.save('%sdme-%s-%i.h5' % (model_save_path, id, counter))
        # print("Saved model to disk")

        counter += 1
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    _run.log_scalar("average.test.accuracy", "%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    _run.log_scalar("#experiement", title)
