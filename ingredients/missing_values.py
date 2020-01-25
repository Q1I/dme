def block_dense(var, size):
    var = Dropout(0.1)(var)
    var = Dense(size, activation='sigmoid')(var)
    var = BatchNormalization()(var)
    return var


class EyesCompletionStructure(NetworkStructure):
        def __init__(self, ti):
        super(EyesCompletionStructure, self).__init__(ti)

    def create_model(self):
        encoder = self.create_encoder()
        meta_reconstructer = self.create_meta_reconstructer()

        month0 = [Input((128, 128, 1)) for _ in range(3)]
        month3 = [Input((128, 128, 1)) for _ in range(3)]

        enc0 = [encoder(inp) for inp in month0]
        enc0_min = minimum(enc0)
        enc0_max = maximum(enc0)
        enc3 = [encoder(inp) for inp in month3]
        enc3_min = minimum(enc3)
        enc3_max = maximum(enc3)
        enc = concatenate([enc0_min, enc0_max, enc3_min, enc3_max])

        meta = Input((21,))
        meta_msk = Input((21,))
        meta = Dropout(0.25)(meta)

        vec = concatenate([enc, meta])

        reconstructed = meta_decoder()
        loss = Lambda(lambda t: (t[0]-t[1])**2)([meta, reconstructed])
        loss = multiply([loss, meta_msk])
        loss = Lambda(lambda x: K.mean(x, axis=-1))(loss)

        model = Model([image, meta, meta_msk], loss)
        model.compile(optimizer='nadam', loss='mse')

        return model

    def create_meta_reconstructer(self, vector_size):
        inp = Input((vector_size,))
        var = Dropout(0.1)(inp)

        var = block_dense(var, 512)
        var = block_dense(var, 256)
        var = block_dense(var, 128)
        var = Dense(21, activation='linear')(var)

        model = Model(inp, var)
        return model