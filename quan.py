import cv2
import numpy as np

from keras.layers import *


# skip layer, siehe https://arxiv.org/pdf/1512.03385.pdf, Abbildung 2
def down(x):
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
class EyesMonthsClassifier(object):
    def create_model(self):
        month0 = [Input((self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        month3 = [Input((self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]

        # hier würde man dann zusätzliche Informationen erstellen:
        # extra = Input((num_extra,))

        encoder = self._create_encoder()
        classifier = self._create_classifier(input_size=512)

        if self.num_examples > 1:
            enc0 = [encoder(inp) for inp in month0]
            enc0_min = minimum(enc0)
            enc0_max = maximum(enc0)
            enc0 = concatenate([enc0_min, enc0_max])
            enc3 = [encoder(inp) for inp in month3]
            enc3_min = minimum(enc3)
            enc3_max = maximum(enc3)
            enc3 = concatenate([enc3_min, enc3_max])
        else:
            enc0 = encoder(month0[0])
            enc3 = encoder(month3[0])

        enc = concatenate([enc0, enc3])

        # Extra-Informationen werden dann hier konkateniert:
        # enc = concatenate([enc, extra])

        out = classifier(enc)
        
        model = Model(month0 + month3, out, name='eye_sum_model')
        model.compile(loss='mse', optimizer='nadam', metrics=['ca'])

        self.inputs_val = month0 + month3
        self.val_output = [out]
        self.vector_model = Model(month0 + month3, out)
        # bzw: Model(month0 + month3 + [extra], out)

        return model

    def _create_encoder(self):
        inp1 = Input((self.input_size, self.input_size, 1))
        inp2 = Lambda(lambda x: 1-x)(inp1)
        inp = concatenate([inp1, inp2])
        
        var = Dropout(0.1)(inp)

        while var._keras_shape[1] > 2:
            var = down(var)

        write(var._keras_shape)
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
    def __init__(self):
        self.path = '/scratch/ws/bryan-eyes/'
        self.files = {}
        self.examples = {}

        for target in ['dmenr', 'dmer']:
            self._load_files(target)
            write('Loaded %i files for %s and got %i examples!' % (len(self.files[target]), target, len(self.examples[target])))

    def _load_files(self, target):
        self.files[target] = []
        self.examples[target] = []

        path = '%s%s/%s/' % (self.path, target, 'train')
        for file in os.listdir(path):
            self.files[target] += [path + '/' + file]
        path = '%s%s/%s/' % (self.path, target, 'test')
        for file in os.listdir(path):
            self.files[target] += [path + '/' + file]

    def _resize(self, mat):
        return [cv2.resize(mat[i], dsize=(self.input_size, self.input_size)) for i in range(mat.shape[0])]

    def _load(self, target):
        mat = np.load(self.files[target][file_index])

        if np.max(mat) > 1:
            mat = mat / 255

        images = self._resize(mat)
        self.examples[target] += images

    def get_pos_examples(self):
        return self._get_examples(self.data_source.examples['dmer'])

    def get_neg_examples(self):
        return self._get_examples(self.data_source.examples['dmenr'])


class EyesMonthsDataGenerator(object):
    def __init__(self):
        self.batch_size = 32
        self.num_examples = 2

    def __next__(self):
        return self.create_example(self.batch_size)


    # zur Hälfte positive und negative Beispiele
    # num_examples = wieviele Bilder pro Zeitpunkt, werden zufällig ausgewählt
    def create_example(self):
        M0 = [np.zeros((self.batch_size, self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        M3 = [np.zeros((self.batch_size, self.input_size, self.input_size, 1)) for _ in range(self.num_examples)]
        Y = np.zeros((self.batch_size, 2))

        for batch_idx in range(self.batch_size):
            if batch_idx < self.batch_size // 2:
                Y[batch_idx, 0] = 1
                p0, p3 = self.data_source.get_pos_example()

                indexes = list(range(self.num_examples))
                random.shuffle(indexes)
                indexes = indexes[:self.num_examples]

                for i in indexes:
                    M0[i][batch_idx, :, :, 0] = p0[i]
                    M3[i][batch_idx, :, :, 0] = p3[i]
            else:
                Y[batch_idx, 1] = 1
                n0, n3 = self.data_source.get_neg_example()
                
                indexes = list(range(self.num_examples))
                random.shuffle(indexes)
                indexes = indexes[:self.num_examples]

                for i in indexes:
                    M0[i][batch_idx, :, :, 0] = n0[i]
                    M3[i][batch_idx, :, :, 0] = n3[i]

        return M0 + M3, Y
