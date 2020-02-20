import cv2
import csv
import numpy as np
import os
import random
import pandas as pd
import glob
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
import tensorflow.keras.backend as K

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.utils import *
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, ReduceLROnPlateau

from ingredients.extras import Extras
from ingredients.logging import log_average_scores, log_average_predictions
from ingredients.prediction import get_missing_values_prediction, predictions_summary
from ingredients.plotting import plot

from sacred import Ingredient

ingredient = Ingredient('missing_values')

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
    predictions_save_path = 'path_to_generated_predictions'
    verbose = 2
    patience = 10
    evenly_distributed = False
    test_all = False # use all data for testing (ignore kfold)
    n_splits = 10
    n_repeats = 2
    # extras
    extras = ['bcva','cstb','mrtb','hba1c','prp', 'lens', 'pdr', 'gender', 'avegf', 'age', 'duration']
    num_extra = len(extras) + 6 #(prp_yes, prp_no, lens_phakic, lens_pseudophakic, pdr_npdr, pdr_pdr, gender_male, gender_female, avegf_ranibizumab, avegf_aflibercept, avegf_bevacizumab)
    validation_ids = ['A063', 'A064', 'A065', 'A066', 'A067', 'A090', 'A091', 'A092', 'A093', 'A094', 'A095', 'A096', 'A097', 'A098', 'A099', 'A100', 'A101', 'A102', 'A103', 'A104', 'A105', 'A106', 'A107', 'A108', 'A109', 'A110', 'A111']

def block_dense(var, size):
    var = Dropout(0.1)(var)
    var = Dense(size, activation='sigmoid')(var)
    var = BatchNormalization()(var)
    return var

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
    def __init__(self, num_examples, input_size, extras, num_extra):
        self.num_examples = num_examples
        self.input_size = input_size
        self.num_extra = num_extra

    def create_model(self): 
        month0 = [Input((self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        month3 = [Input((self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
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

        meta = Input((self.num_extra,), name='meta')
        meta_msk = Input((self.num_extra,), name='meta_msk')
        meta2 = Dropout(0.25)(meta)

        enc = concatenate([enc, meta2])

        meta_reconstructer = self.create_meta_reconstructer(classifier_input_size + self.num_extra)
        
        reconstructed = meta_reconstructer(enc)
        reconstructed_sig = Activation('sigmoid')(reconstructed)
        meta_sig = Activation('sigmoid')(meta)
        loss = Lambda(lambda t: (t[0]-t[1])**2)([meta_sig, reconstructed_sig])
        loss = multiply([loss, meta_msk])
        loss = Lambda(lambda x: K.mean(x, axis=-1))(loss)
        loss = Reshape((1,))(loss)
        
        model = Model(month0 + month3 + [meta] + [meta_msk], loss, name='eye_sum_model')

        predict_model = Model(month0 + month3 + [meta] + [meta_msk], [reconstructed], name='predict_model')

        model.compile(loss='mse', optimizer='nadam')

        return model, predict_model

    def create_meta_reconstructer(self, vector_size):
        inp = Input((vector_size,))
        var = Dropout(0.1)(inp)

        var = block_dense(var, 512)
        var = block_dense(var, 256)
        var = block_dense(var, 128)
        var = Dense(self.num_extra, activation='linear')(var)

        model = Model(inp, var, name='eye_reconstructer')
        return model

    def _create_encoder(self):
        inp1 = Input((self.input_size, self.input_size, 1))
        inp2 = Lambda(lambda x: 1-x)(inp1)
        inp = concatenate([inp1, inp2])
        
        var = Dropout(0.1)(inp)

        while var.shape[1] > 2:
            var = down(var)

        # print(var.shape)
        var = Flatten()(var)

        var = Dropout(0.1)(var)
        var = Dense(256, activation='sigmoid')(var)

        return Model(inp1, var, name='eye_sum_encoder')

class EyesMonthsDataGenerator(Sequence):
    @ingredient.capture 
    def __init__(self, num_examples, input_size, batch_size, extras, num_extra, excel_path):
        self.batch_size = batch_size
        self.num_examples = num_examples
        self.input_size = input_size
        self.data_source = EyesNumpySource() # TODO
        self.shuffle = True
        self.extras = extras
        self.num_extra = num_extra

        self.ids = [] # list of all ids
        self.labels = [] # list of labels

        # extras processor for reading and procesing extras excel
        self.extras_processor = Extras(excel_path, extras, num_extra)

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

    @ingredient.capture
    def data_generation(self, index_list, evenly_distributed, batch_size, no_missing_value = False):
        'Generates data containing batch_size samples'
        if batch_size is None:
            batch_size = len(index_list)
        else:
            batch_size = self.batch_size

        # Initialization        
        M0 = [np.zeros((batch_size, self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        M3 = [np.zeros((batch_size, self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        Y = np.zeros((batch_size, 1))
        EXTRA = np.zeros((batch_size, self.num_extra, ))
        EXTRA_MSK = np.zeros((batch_size, self.num_extra, ))
        
        dataset_size = len(Y)
        ids = []
        # Generate data
        for counter, idx in enumerate(index_list):
            if counter >= dataset_size:
                break

            id = self.ids[idx]

            # evenly distributed pos and neg examples
            if evenly_distributed:
                condition = counter < batch_size // 2
                # Find list of IDs
                ids = [self.ids[k] for k in self.train_indexes]
            else: # no distribution, use index_list
                condition = self.labels[idx] == 0

            # label
            if condition:
                # Y[counter, 0] = 1
                target = 'dmer'
            else:
                # Y[counter, 1] = 1
                target = 'dmenr'
            
            # sample
            p0, p3, p_id = self.data_source.get_example(target, id, evenly_distributed, ids)

            indexes = list(range(self.num_examples))
            random.shuffle(indexes)
            indexes = indexes[:self.num_examples]

            for i in indexes:
                M0[i][counter, :, :, 0] = p0[i]
                M3[i][counter, :, :, 0] = p3[i]
            
            # extras
            extras_val, extras_msk = self.extras_processor.get_extras(p_id)
            EXTRA[counter] = extras_val
            EXTRA_MSK[counter] = extras_msk
            if no_missing_value == True:
                EXTRA_MSK[counter] = 1
            

        return M0 + M3 + [EXTRA] + [EXTRA_MSK], [Y]

    def set_train_indexes(self, indexes):
        self.train_indexes = indexes

    # return list of all ids and labels for k-fold split (dmer, dmenr, dmev)
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

    def get_index(self, id):
        return self.ids.index(id)

    def get_ids(self, indexes=None):
        if indexes is not None:
            ids = []
            for index in indexes:
                ids.append(self.ids[index])
            return ids
        else:
            return self.ids

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

        path = '%s/%s' % (self.path, target)
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

    def get_example(self, target, id, evenly_distributed, ids):
        if evenly_distributed:
            # random example
            if target == 'dmer':
                example, example_id = self.get_pos_example(ids)
            else:
                example, example_id = self.get_neg_example(ids)
            example_m0, example_m3 = self.parse_example(target, example)
            return example_m0, example_m3, example_id
        else:
            # example by id
            example_m0, example_m3 = self.parse_example(target, self._load(target, id))
            return  example_m0, example_m3, id
        # return parse_example(self.examples[target][id])

    def get_pos_example(self, training_ids):
        all_positives = list(self.files['dmer'].keys())
        positives = [x for x in training_ids if x in all_positives]
        id = random.choice(positives)
        return self._load('dmer', id), id

    def get_neg_example(self, training_ids):
        all_negatives = list(self.files['dmenr'].keys())
        negatives = [x for x in training_ids if x in all_negatives]
        id = random.choice(negatives)
        return self._load('dmenr', id), id

    @ingredient.capture
    def parse_example(self, target, example):
        return example[0], example[1]

def ca(y_true, y_pred):
    return 1 - K.mean(K.abs(y_true - y_pred))

def log_metrics(history, cvscores, epoch, _run):
    """
    log metrics to cout.txt and metrics.txt 
    :param dict history: history from all epochs {'loss': [0.1, ..], ..} 
    :param dict cvscores: Avg scores from all cv-runs [{'loss': 0.1, ..}, ..] 
    :param int epoch: current epoch
    """
    print('### metrics for #%i cv-run (avg of last 10 metrics of the run):' % epoch)
    avg_metric = {}
    for key, value in history.items():
        avg_metric[key] = np.mean(value[-10:]) 
        print('%s: %f'  % (key, avg_metric[key]))
        _run.log_scalar(key, avg_metric[key], epoch)
    cvscores.append(avg_metric)
    log_average_scores([*history], cvscores)

def static_test_data(generator, validation_ids = []):
    train_ids = []
    test_ids = []
    train_indexes = []
    test_indexes = []

    # test
    test_ids = validation_ids
    # for file in os.listdir('/home/q1/Python/dl/data/uniklinik_augen/dme-data/dmev_2'):
    #     test_id = file.split('_')[0]
    #     if test_id not in test_ids:
    #         test_ids.append(test_id)
    for test_id in test_ids:
        test_indexes.append(generator.get_index(test_id))

    # train
    train_ids = [i for i in generator.get_ids() if i not in test_ids]
    for train_id in train_ids:
        train_indexes.append(generator.get_index(train_id))

    return train_indexes, test_indexes

@ingredient.capture
def missing_values_run(_run, title, epochs, model_save_path, history_save_path, verbose, patience, test_all, validation_ids, use_validation, n_splits, n_repeats, predictions_save_path):
    id = _run._id
    # fix random seed for reproducibility
    seed = 7
    np.random.seed(seed)

    seed = random.randint(1, 100)
    print('Seed:', seed)

    cvscores = []
    generator = EyesMonthsDataGenerator()
    X, Y = generator.get_all_data()
    counter = 0
    
    extras_processor = generator.extras_processor
    # extra_column = 'HbA1c at DME diagnosis, (%)'
    # avg = extras_processor.get_column_average(extra_column)
    # std = extras_processor.get_column_std(extra_column)
    # print('AVG', avg);
    # print('STD', std);
    # print(extras.get_extras('A001', False));
    # print(extras.get_extras('A002', False));
    # print(extras.get_extras('A003', False));
    # quit()

    predictions = {} # {eye1: [prediction1, ..], ..}
    for item_id in sorted(generator.get_ids()):
        predictions[item_id] = []

    # callbacks
    history_id_path = "%s%s/" % (history_save_path, id)
    total_path = history_id_path + 'weights-improvement-{val_ca:.2f}.hdf5'
    # total_path = history_id_path + 'weights-improvement-{epoch:02d}-{val_ca:.2f}.hdf5'
    checkpoint = ModelCheckpoint(total_path, monitor='val_ca', verbose=0, save_best_only=True, mode='max')
    es = EarlyStopping(monitor='val_ca', mode='max', verbose=2, patience=patience)
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, min_lr=0.001)
    
    history = None

    trainX, trainY = generator.data_generation(list(range(len(X))), False, None)

    for run in range(n_repeats): 

        # callback
        callbacks_list = []

        # set train data indexes for generator
        generator.set_train_indexes(len(X))

        # create model
        model, predict_model = EyesMonthsClassifier().create_model()
        print(model.summary())

        history = model.fit(trainX, trainY, validation_data=None, epochs=epochs, verbose=verbose, callbacks=callbacks_list)

        # predictions
        predictions, misses, validations = predict(predict_model, generator, predictions)

        # log_metrics(history.history, cvscores, counter, _run)
        
        # plot(history, history_save_path, id, counter)
        
        # save model
        # model.save('%sdme-%s-%i.h5' % (model_save_path, id, counter))
        # print("Saved model to disk")

        counter += 1

    # prediction_accuracy = log_average_predictions(predictions, predictions_save_path, id, validations)

    # log_average_scores([*(history.history)], cvscores, id, True, prediction_accuracy)
    
    # print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    # _run.log_scalar("average.test.accuracy", "%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    _run.log_scalar("#experiement", title)

###############
# Predictions #
###############
def predict(model, generator, predictions = {}, validation_ids = None, shouldPrint=False):
    if shouldPrint:
        print("[INFO] classifying images...")
    # predictions_responder_values = []
    # predictions_non_responder_values = [] 
    # predictions_miss = {'x': [], 'y_r': [], 'y_nr': []}
    predictions_ids = validation_ids if validation_ids is not None else sorted(generator.get_ids())
    misses = []
    validations = {}
    
    mv_index = generator.extras.index('bcva_delta_m0_m12')

    extras_tool = generator.extras_processor
    # extra_column = 'HbA1c at DME diagnosis, (%)'
    extra_column = 'Visual acuity change from Month 0 to Month 12 (letters)'
    avg = extras_tool.get_column_average(extra_column)
    std = extras_tool.get_column_std(extra_column)
    print('AVG', avg);
    print('STD', std);

    for item in predictions_ids:
        imageX, imageY = generator.data_generation([generator.get_index(item)], False, None, True)
        prob = model.predict(imageX)
        prediction = get_missing_values_prediction(prob[0,mv_index], avg, std)

        extras_encoded, msk = extras_tool.get_extras(item)
        extras_decoded, msk = extras_tool.get_extras(item, False)
        print('PRED %s: %.3f' % (item, prediction), extras_decoded[mv_index], 'encoded: %.3f' % prob[0,mv_index], extras_encoded[mv_index], 'masked ', msk)
        if shouldPrint:
            print('[INFO] Predict %s [ %s ] : %s => %s' % (item, validation, prediction, prob))
        # idxs = np.argsort(proba)[::-1][:2]

        # predictions[item].append(prob)
        # validations[item] = validation

    # plot
    # plot(predictions_ids, predictions_responder_values, predictions_non_responder_values, predictions_miss)

    return predictions, misses, validations



