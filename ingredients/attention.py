class MNISTGanMaskDataGenerator(object):
    def __init__(self, ti):
        self.data_source = ti['data_source_constructor'](ti)
        self.batch_size = ti.get('batch_size', 32)

    def __next__(self):
        XP = np.zeros((self.batch_size, 32, 32, 1))
        XN = np.zeros((self.batch_size, 32, 32, 1))
        YP = np.zeros((self.batch_size, 10))
        YN = np.zeros((self.batch_size, 10))
        Y = np.zeros((self.batch_size, 1))

        for i in range(self.batch_size):
            xp, yp = self.data_source._get_data()
            xn, yn = self.data_source._get_data()

            XP[i, :28, :28, :] = xp
            XN[i, :28, :28, :] = xn
            YP[i, yp] = 1
            YN[i, yn] = 1

        return [XP, XN, YP, YN], [Y, Y]

class MNISTGanMaskNetwork(NetworkStructure):
    def __init__(self, ti):
        super(MNISTGanMaskNetwork, self).__init__(ti)

    def create_vector_model(self):
        self.vector_model

    def create_model(self):
        msk_encoder = self.create_msk_encoder()
        msk_decoder = self.create_decoder()
        clz_encoder = self.create_clz_encoder()
        classifier = self.create_classifier()

        P = Input((32, 32, 1))
        N = Input((32, 32, 1))

        YP = Input((10,))
        YN = Input((10,))

        msk = msk_encoder(P)
        msk = msk_decoder(msk)

        p = multiply([P, msk])
        n = multiply([N, msk])

        clz_p = classifier(clz_encoder(P))
        clz_n = classifier(clz_encoder(N))

        loss_p = Lambda(lambda t: K.mean((t[0]-t[1])**2, axis=-1))([YP, clz_p])
        loss_n = Lambda(lambda t: K.mean((t[0]-t[1])**2, axis=-1))([YN, clz_n])
        loss_n = Lambda(lambda x: 1-x)(loss_n)

        loss_p = Reshape((1,))(loss_p)
        loss_n = Reshape((1,))(loss_n)

        model = Model([P, N, YP, YN], [loss_p, loss_n])
        model.compile(optimizer='nadam', loss='mse')
        self.vector_model = model
        return model

    @model_definition(name='decoder')
    def create_decoder(self):
        inp = Input((2, 2, 64))
        write('Decode!')

        var = UpSampling2D()(inp)
        var = block_conv(var, 64)
        write(str(var._keras_shape))

        var = UpSampling2D()(inp)
        var = block_conv(var, 64)
        write(str(var._keras_shape))

        var = UpSampling2D()(var)
        var = block_conv(var, 32)
        write(str(var._keras_shape))

        var = UpSampling2D()(var)
        var = block_conv(var, 32)
        write(str(var._keras_shape))

        var = UpSampling2D()(var)
        var = block_conv(var, 1, use_bn=False)
        write(str(var._keras_shape))

        return Model(inp, var)

    @model_definition(name='msk_encoder')
    def create_msk_encoder(self):
        return self._create_encoder()

    @model_definition(name='clz_encoder')
    def create_clz_encoder(self):
        return self._create_encoder()

    def _create_encoder(self):
        inp = Input((32, 32, 1))
        write('Encode!')

        var = block_conv(inp, 32)
        var = MaxPooling2D()(var)
        write(str(var._keras_shape))

        var = block_conv(var, 32)
        var = MaxPooling2D()(var)
        write(str(var._keras_shape))

        var = block_conv(var, 64)
        var = MaxPooling2D()(var)
        write(str(var._keras_shape))

        var = block_conv(var, 64)
        var = MaxPooling2D()(var)
        write(str(var._keras_shape))

        return Model(inp, var)

    @model_definition(name='classifier')
    def create_classifier(self):
        inp = Input((2, 2, 64))
        var = Flatten()(inp)
        var = Dropout(0.1)(var)
        var = Dense(128, activation='sigmoid')(var)
        var = BatchNormalization()(var)
        var = Dropout(0.1)(var)
        var = Dense(64, activation='sigmoid')(var)
        var = BatchNormalization()(var)
        var = Dropout(0.1)(var)
        var = Dense(10, activation='softmax')(var)

        return Model(inp, var)