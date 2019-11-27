import cv2
import numpy as np
import os
import random
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import tensorflow.keras.backend as K

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.utils import *

from sacred import Ingredient

ingredient = Ingredient('dme')

@ingredient.config
def cfg():
    num_examples = 1
    input_size = 30
    generator_batch_size = 109 # number of total eyes
    numpy_source_path = 'path_to_numpy_matrices'
    dropout_rate = 0.2
    filters = 32
    fit_batch_size = 32
    epochs = 50
    steps_per_epoch = 100
    excel_path = 'path_to_excel_data_file'
    model_save_path = 'path_to_saved_models'
    verbose = 2


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
    def __init__(self, num_examples, input_size):
        self.num_examples = num_examples
        self.input_size = input_size

    def create_model(self):
        month0 = [Input((self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        month3 = [Input((self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]

        # hier würde man dann zusätzliche Informationen erstellen:
        num_extra = 1
        extra = Input((num_extra,))

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


        classifier = self._create_classifier(input_size=classifier_input_size + num_extra)
        out = classifier(enc)

        model = Model(month0 + month3 + [extra], out, name='eye_sum_model')
        model.compile(loss='mse', optimizer='nadam', metrics=[ca])

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
        self.examples = {}
        self.input_size = input_size

        for target in ['dmenr', 'dmer']:
            self._load_files(target)
            print('Loaded %i files for %s and got %i examples!' % (len(self.files[target]), target, len(self.examples[target])))

    def _load_files(self, target):
        self.files[target] = {}
        self.examples[target] = {}

        path = '%s/%s/%s' % (self.path, target, 'train')
        for file_index, file in enumerate(os.listdir(path)):
            file_name = file.split('.')[0]
            self.files[target][file.split('.')[0]] = path + '/' + file
            self._load(target, file_name) # TODO
        path = '%s/%s/%s' % (self.path, target, 'test')
        for file_index, file in enumerate(os.listdir(path)):
            file_name = file.split('.')[0]
            self.files[target][file_name] = path + '/' + file
            self._load(target, file_name) # TODO

    def _resize(self, mat):
        return [cv2.resize(mat[i], dsize=(self.input_size, self.input_size)) for i in range(mat.shape[0])]

    def _load(self, target, id):
        mat = np.load(self.files[target][id])
        if np.max(mat) > 1:
            mat = mat / 255

        images = self._resize(mat)
        # self.examples[target][id] = images
        self.examples[target][id] = mat

    def get_example(self, target, id):
        return self.examples[target][id]

    def get_sample(self, sample):
        return self._resize(sample[0]), self._resize(sample[1])

class EyesMonthsDataGenerator(Sequence):
    @ingredient.capture
    def __init__(self, num_examples, input_size, generator_batch_size, excel_path):
        self.batch_size = generator_batch_size
        self.num_examples = num_examples
        self.input_size = input_size
        self.data_source = EyesNumpySource() # TODO
        self.extras = pd.read_excel(excel_path)
        # helper for k-fold train and test datasets
        self.X = []
        self.Y = []
        self.indexes = [] # map index to id

    # Gets batch at position index.
    def __getitem__(self, index):
        # return self.create_example()
        return self.create_sample()

    # Number of batch in the Sequence
    def __len__(self):
        return 100

    def set_train_indexes(self, indexes):
        self.train_indexes = indexes

    def load_data(self):
        X = []
        Y = []
        indexes = []
        keys_dmer = list(self.data_source.examples['dmer'].keys())
        keys_dmenr = list(self.data_source.examples['dmenr'].keys())
        random.shuffle(keys_dmer)
        random.shuffle(keys_dmenr)

        for batch_idx in range(self.batch_size):
            if batch_idx < self.batch_size // 2 and batch_idx < len(keys_dmer):
                id = keys_dmer[batch_idx]
                target = 'dmer'
                Y.append(0)
            else:
                id = keys_dmenr[batch_idx % len(keys_dmenr)]
                target = 'dmenr'
                Y.append(1)

            X.append(self.data_source.get_example(target, id))
            indexes.append(id)

        self.X = X
        self.Y = Y   
        self.indexes = indexes     
        return X, Y

    def create_sample(self, data_indexes = None):
        if data_indexes is None:
            data_indexes = self.train_indexes
        M0 = [np.zeros((self.batch_size, self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        M3 = [np.zeros((self.batch_size, self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        Y = np.zeros((self.batch_size, 2))
        EXTRA = [np.zeros((self.batch_size, 1)) for _ in range(self.num_examples)]

        counter = 0
        for idx in data_indexes:
            if self.Y[idx] == 0:
                Y[counter, 0] = 1
            else:
                Y[counter, 1] = 1
            
            p0, p3 = self.data_source.get_sample(self.X[idx])

            indexes = list(range(self.num_examples))
            random.shuffle(indexes)
            indexes = indexes[:self.num_examples]

            id = self.indexes[idx]
            baselineData = self._baseline(id)
            
            for i in indexes:
                M0[i][counter, :, :, 0] = p0[i]
                M3[i][counter, :, :, 0] = p3[i]
                EXTRA[i][counter] = baselineData
            
            counter += 1

        return M0 + M3 + EXTRA, Y

    # extras
    @ingredient.capture
    def _baseline(self, id):
        return self.extras.loc[self.extras['ID'] == id]['Baseline BCVA (LogMAR)'].values[0]
        # return 0


def ca(y_true, y_pred):
    return 1 - K.mean(K.abs(y_true - y_pred))

@ingredient.capture
def dme_run(id, fit_batch_size, steps_per_epoch, epochs, model_save_path, verbose):
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    generator = EyesMonthsDataGenerator()
    X, Y = generator.load_data()
    counter = 0
    for train_indexes, test_indexes in kfold.split(X, Y): # return lists of indexes
        print('train indexes: ', train_indexes)
        print('test indexes: ', test_indexes)
        # set train data indexes for generator
        generator.set_train_indexes(train_indexes)
        # create model
        model = EyesMonthsClassifier().create_model()
        # test data
        testX, testY = generator.create_sample(test_indexes)
        # Fit the model
        model.fit_generator(generator, validation_data=(testX, testY), epochs=epochs, verbose=verbose)
        # trainX, trainY = generator.create_sample(train_indexes)
        # model.fit(trainX, trainY, batch_size=32, epochs=100)
        print('train done')
        # evaluate the model
        scores = model.evaluate(testX, testY, verbose=0)
        print('test loss, test acc:', scores)
        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
        print("average acc: %.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        model.save('%sdme-%s-%i.h5' % (model_save_path, id, counter))
        print("Saved model to disk")
        counter += 1
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
