import cv2
import numpy as np
import os
import random
import pandas as pd
# import matplotlib.pyplot as plt
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
    use_baseline = True
    evenly_distributed = False


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

    def create_train_model(self):
        month0_pos = [Input((self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        month3_pos = [Input((self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        month0_neg = [Input((self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        month3_neg = [Input((self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        
         # hier würde man dann zusätzliche Informationen erstellen:
        num_extra = 1
        extra_pos = Input((num_extra,))
        extra_neg = Input((num_extra,))

        base_model = self.create_base_model()
        out_pos = base_model(month0_pos + month3_pos + [extra_pos])
        out_neg = base_model(month0_neg + month3_neg + [extra_neg])

        # Extra-Informationen werden dann hier konkateniert:
        # out_pos = concatenate([out_pos, extra])
        # out_neg = concatenate([out_neg, extra])

        train_model = Model(month0_pos + month3_pos + [extra_pos] + month0_neg + month3_neg + [extra_neg], [out_pos, out_neg], name='train_model')
        train_model.compile(loss='mse', optimizer='nadam', metrics=[ca, ca]) # ca
        return train_model

    def create_base_model(self):
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

        model = Model(month0 + month3 + [extra], out, name='base_model')

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

    # return p0, p3, n0, n3 
    @ingredient.capture
    def parse_example(self, target, example):
        if target == 'dmer':
            return example[0], example[1], None, None
        else:
            return None, None, example[0], example[1] 

class EyesMonthsDataGenerator(Sequence):
    @ingredient.capture
    def __init__(self, num_examples, input_size, batch_size, excel_path):
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.input_size = input_size
        self.data_source = EyesNumpySource() # TODO
        self.extras = pd.read_excel(excel_path)
        self.shuffle = True
        
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
        M0_POS = [np.zeros((self.batch_size, self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        M3_POS = [np.zeros((self.batch_size, self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        M0_NEG = [np.zeros((self.batch_size, self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        M3_NEG = [np.zeros((self.batch_size, self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        Y_POS = np.zeros((self.batch_size, 2))
        Y_NEG = np.zeros((self.batch_size, 2))
        EXTRA = [np.zeros((self.batch_size, 1))]

        # Generate data
        for counter, idx in enumerate(index_list):
            if counter >= self.batch_size:
                break

            id = self.ids[idx]

            # evenly distributed pos and neg examples
            if evenly_distributed:
                condition = counter < self.batch_size // 2
            else: # no distribution, use index_list
                condition = self.labels[idx] == 0

            # label
            if condition:
                Y_POS[counter, 0] = 1
                Y_NEG[counter, 1] = 1
                target = 'dmer'
            else:
                Y_POS[counter, 1] = 1
                Y_NEG[counter, 0] = 1
                target = 'dmenr'
            
            # sample
            p0, p3, n0, n3 = self.data_source.get_example(target, id, evenly_distributed)

            indexes = list(range(self.num_examples))
            random.shuffle(indexes)
            indexes = indexes[:self.num_examples]

            # extras
            baselineData = self._baseline(id)
            
            for i in indexes:
                if p0 is not None:
                    M0_POS[i][counter, :, :, 0] = p0[i]
                if p3 is not None:
                    M3_POS[i][counter, :, :, 0] = p3[i]
                if n0 is not None:
                    M0_NEG[i][counter, :, :, 0] = n0[i]
                if n3 is not None:
                    M3_NEG[i][counter, :, :, 0] = n3[i]
            
            EXTRA[0][counter] = baselineData

        return M0_POS + M3_POS + EXTRA + M0_NEG + M3_NEG + EXTRA, [Y_POS ,Y_NEG]

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
    def _baseline(self, id, use_baseline):
        if use_baseline:
            return self.extras.loc[self.extras['ID'] == id]['Baseline BCVA (LogMAR)'].values[0]
        else:
            return 0


def ca(y_true, y_pred):
    return 1 - K.mean(K.abs(y_true - y_pred))

def plot(history, history_save_path, id, counter):
    # summarize history for accuracy
    plt.plot(history.history['base_model_ca'])
    plt.plot(history.history['base_model_1_ca'])
    plt.plot(history.history['val_base_model_ca'])
    plt.plot(history.history['val_base_model_1_ca'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train_pos', 'train_neg', 'test_pos', 'test_neg'], loc='upper left')
    plt.savefig('%s%s/accuracy-%i.png' % (history_save_path, id, counter))
    plt.clf()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['base_model_loss'])
    plt.plot(history.history['base_model_1_loss'])
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['val_base_model_loss'])
    plt.plot(history.history['val_base_model_1_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_sum', 'train_pos', 'train_neg', 'test_sum', 'test_pos', 'test_neg'], loc='upper left')
    plt.savefig('%s%s/loss-%i.png' % (history_save_path, id, counter))
    plt.clf()

def log_metrics(keys, scores, cvscores, counter, _run):
    print('### metrics:')
    for i, key in enumerate(keys):
        print('%s: %f'  % (key, scores[i]))
        _run.log_scalar(key, scores[i], counter)

    print('### average:')
    for idx, key in enumerate(keys):
        tmp = []
        for i, score in enumerate(cvscores):
            tmp.append(score[idx])
        print('avg %s:  %.2f%% (+/- %.2f%%)'  % (key, np.mean(tmp), np.std(tmp)))
        
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
    filepath = "%s%s/weights-improvement-{epoch:02d}-{val_base_model_ca:.2f}.hdf5" % (history_save_path, id)
    checkpoint = ModelCheckpoint(filepath, monitor='val_base_model_ca', verbose=0, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=patience)
    callbacks_list = [es, checkpoint]

    for train_indexes, test_indexes in kfold.split(X, Y): # return lists of indexes
        print('### K-Fold split: ', counter)
        print('train indexes: ', train_indexes)
        print('test indexes: ', test_indexes)
        
        # set train data indexes for generator
        generator.set_train_indexes(train_indexes)

        # create model
        model = EyesMonthsClassifier().create_train_model()
        # model.summary()
        # plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        # test data
        testX, testY = generator.data_generation(test_indexes)
        # Fit the model
        history = model.fit_generator(generator, validation_data=(testX, testY), epochs=epochs, verbose=verbose, callbacks=callbacks_list)
        # trainX, trainY = generator.create_sample(train_indexes)
        # model.fit(trainX, trainY, batch_size=32, epochs=100)

        # list all data in history
        # print(history.history.keys())

        # evaluate the model
        scores = model.evaluate(testX, testY, verbose=0)      
        cvscores.append(scores)

        log_metrics(model.metrics_names, scores, cvscores, counter, _run)  

        # plot(history, history_save_path, id, counter)
        
        # save model
        # model.save('%sdme-%s-%i.h5' % (model_save_path, id, counter))
        # print("Saved model to disk")

        counter += 1
    _run.log_scalar("#experiement", title)
