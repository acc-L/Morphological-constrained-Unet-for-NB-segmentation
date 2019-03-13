import tensorflow as tf
from tensorflow.keras import Model, layers, optimizers, losses, regularizers
#from clayer import SelectiveMerge_v2

reg = None

def dice_loss(y_true, y_pred):
    r = tf.reduce_sum(tf.abs(y_pred * y_true), axis=[1,2,3]) + 1
    m = tf.reduce_sum((y_true ** 2) + (y_pred ** 2), axis=[1,2,3]) + 2
    #return tf.log(tf.reduce_mean(m / (2 * r)))
    return tf.reduce_mean(tf.log(m / (2 * r)))

def dice(y_true, y_pred):
    y_pred = tf.round(y_pred)
    r = tf.reduce_sum(tf.abs(y_pred * y_true), axis=[1, 2, 3]) + 1
    m = tf.reduce_sum((y_true ** 2) + (y_pred ** 2), axis=[1, 2, 3]) + 2
    return tf.reduce_mean(r*2/m)

def m_dice(y_true, y_pred):
    y_pred = tf.round(y_pred)
    r = tf.reduce_sum(tf.abs(y_pred * y_true), axis=[1, 2]) + 1
    m = tf.reduce_sum((y_true ** 2) + (y_pred ** 2), axis=[1, 2]) + 2
    return tf.reduce_mean(r*2/m, axis=0)[1:]

def non_local_layer(input, layer_index, inter_dim=None):
    batchsize, dim1, dim2, channels = input.get_shape().as_list()
    if inter_dim==None:
        inter_dim = channels//2
        if inter_dim < 1:
            inter_dim = 1

    N = dim1 * dim2

    theta = layers.Conv2D(inter_dim, (1, 1), use_bias=False, kernel_initializer='he_normal',
                          name='nl_theta_'+str(layer_index))(input)
    theta = layers.Reshape((-1, inter_dim))(theta)

    phi = layers.Conv2D(inter_dim, (1, 1), use_bias=False, kernel_initializer='he_normal',
                        name='nl_phi_' + str(layer_index))(input)
    phi = layers.Reshape((-1, inter_dim))(phi)

    def matmul(inputs):
        x, y = inputs
        return tf.matmul(x,y)

    #f = tf.matmul(theta, phi, transpose_b=True)
    phi = layers.Permute((2,1))(phi)
    f = layers.Lambda(matmul)([theta, phi])

    f = layers.Lambda(lambda z: (1. / float(N)) * z)(f)

    #f = layers.Softmax()(f)

    g = layers.Conv2D(inter_dim, (1, 1), use_bias=False, kernel_initializer='he_normal',
                      name='nl_g_' + str(layer_index))(input)
    g = layers.Reshape((-1, inter_dim))(g)

    #y = tf.matmul(f, g)
    y = layers.Lambda(matmul)([f,g])
    y = layers.Reshape(( dim1, dim2, inter_dim))(y)

    y = layers.Conv2D(channels, (1, 1),  use_bias=False, kernel_initializer='he_normal',
                      name='nl_y_' + str(layer_index))(y)
    return y


def U_block_3D(inputs, block_index, out_dim, ker_size=[3,3] ):
    net = layers.Conv3D(out_dim, ker_size[0], padding='same', kernel_initializer='he_normal', name='conv'+str(block_index)+'_1')(inputs)
    net = layers.BatchNormalization(axis=4, name='bn'+str(block_index)+'_1')(net)
    net = layers.Activation('relu')(net)
    net = layers.Conv3D(out_dim, ker_size[1], padding='same', kernel_initializer='he_normal', name='conv' + str(block_index) + '_2')(
        net)
    net = layers.BatchNormalization(axis=4, name='bn' + str(block_index) + '_2')(net)
    net = layers.Activation('relu')(net)
    return net

def U_block_2D(inputs, block_index, out_dim, ker_size=[3,3], reg=reg, msign=''):
    if isinstance(inputs, list):
        net = layers.Concatenate()(inputs)
    else:
        net = inputs
    net = layers.BatchNormalization(axis=3, name='bn' + str(block_index) + '_2')(net)
    net = layers.Activation('relu')(net)
    #net = layers.SpatialDropout2D(0.2)(net)
    net = layers.SeparableConv2D(out_dim, ker_size[0], padding='same', kernel_initializer='he_normal', 
                                 kernel_regularizer=reg, use_bias=False, 
                                 name='conv'+msign+str(block_index)+'_1')(net)
    net = layers.BatchNormalization(axis=3, name='bn'+str(block_index)+'_1')(net)
    net = layers.Activation('relu')(net)
    net = layers.SeparableConv2D(out_dim, ker_size[1], padding='same', kernel_initializer='he_normal', 
                                 kernel_regularizer=reg, use_bias=False, 
                                 name='conv' + str(block_index) + '_2')(net)
    return net

def U_block_2D_n(inputs, block_index, out_dim, ker_size=[3,3], reg=reg, msign=''):
    if isinstance(inputs, list):
        net = layers.Concatenate()(inputs)
    else:
        net = inputs
    net = layers.BatchNormalization(axis=3, name='bn' + str(block_index) + '_2')(net)
    net = layers.Activation('relu')(net)
    #net = layers.SpatialDropout2D(0.2)(net)
    net = layers.Conv2D(out_dim, ker_size[0], padding='same', kernel_initializer='he_normal', 
                        kernel_regularizer=reg, use_bias=False, 
                        name='conv'+msign+str(block_index)+'_1')(net)
    net = layers.BatchNormalization(axis=3, name='bn'+str(block_index)+'_1')(net)
    net = layers.Activation('relu')(net)
    net = layers.Conv2D(out_dim, ker_size[1], padding='same', kernel_initializer='he_normal', 
                        kernel_regularizer=reg, use_bias=False, 
                        name='conv' + str(block_index) + '_2')(net)
    return net

def U_block_2D_g(inputs, block_index, out_dim, ker_size=[3,3], reg=reg, msign=''):
    if isinstance(inputs, list):
        net = layers.Concatenate()(inputs)
    else:
        net = inputs
        
    net = layers.Conv2D(out_dim, 1, kernel_initializer='he_normal',
                            kernel_regularizer=reg, use_bias=False, 
                            name = 'merge'+str(block_index))(net)
    
    left = layers.BatchNormalization(axis=3, name='bn' + str(block_index) + '_2')(net)
    left = layers.Activation('relu')(left)
    left = layers.SeparableConv2D(out_dim, ker_size[0], padding='same', kernel_initializer='he_normal', 
                                 kernel_regularizer=reg, use_bias=False, 
                                 name='conv'+msign+str(block_index)+'_1')(left)
    left = layers.BatchNormalization(axis=3, name='bn'+str(block_index)+'_1')(left)
    left = layers.Activation('relu')(left)
    #left = layers.SpatialDropout2D(0.5)(left)
    left = layers.SeparableConv2D(out_dim, ker_size[1], padding='same', kernel_initializer='he_normal', 
                                 kernel_regularizer=reg, use_bias=False, 
                                 name='conv' + str(block_index) + '_2')(left)
    net = layers.Add()([net, left])
    
    left = layers.BatchNormalization(axis=3, name='bn' + str(block_index) + '_3')(net)
    left = layers.Activation('relu')(left)
    left = layers.SeparableConv2D(out_dim, ker_size[0], padding='same', kernel_initializer='he_normal', 
                                 kernel_regularizer=reg, use_bias=False, 
                                 name='conv'+msign+str(block_index)+'_3')(left)
    left = layers.BatchNormalization(axis=3, name='bn'+str(block_index)+'_4')(left)
    left = layers.Activation('relu')(left)
    #left = layers.SpatialDropout2D(0.5)(left)
    left = layers.SeparableConv2D(out_dim, ker_size[1], padding='same', kernel_initializer='he_normal', 
                                 kernel_regularizer=reg, use_bias=False, 
                                 name='conv' + str(block_index) + '_4')(left)
    net = layers.Add()([net, left])
    return net

def dense_block(input, block_index, conv_dim):
    conv1 = layers.Conv2D(conv_dim, 3, padding='same', kernel_initializer='he_normal',
                        name='conv_'+str(block_index)+'_1')(input)
    conv1 = layers.BatchNormalization(axis=3, name='bn' + str(block_index) + '_1')(conv1)
    conv1 = layers.Activation('relu')(conv1)
    conv2 = layers.Conv2D(conv_dim, 3, padding='same', kernel_initializer='he_normal',
                          name='conv_' + str(block_index) + '_2')(conv1)
    conv2 = layers.BatchNormalization(axis=3, name='bn' + str(block_index) + '_2')(conv2)
    conv2 = layers.Activation('relu')(conv2)
    merge2 = layers.Concatenate(axis=3)([conv2,conv1])
    conv3 = layers.Conv2D(conv_dim, 3, padding='same', kernel_initializer='he_normal',
                          name='conv_' + str(block_index) + '_3')(merge2)
    conv3 = layers.BatchNormalization(axis=3, name='bn' + str(block_index) + '_3')(conv3)
    conv3 = layers.Activation('relu')(conv3)
    merge3 = layers.Concatenate(axis=3)([conv3,merge2])
    conv4 = layers.Conv2D(conv_dim, 3, padding='same', kernel_initializer='he_normal',
                          name='conv_' + str(block_index) + '_4')(merge3)
    conv4 = layers.BatchNormalization(axis=3, name='bn' + str(block_index) + '_4')(conv4)
    conv4 = layers.Activation('relu')(conv4)
    return conv4


def nl_block(inputs, block_index, out_dim, ker_size=3, reg=reg):
    net = layers.Conv2D(out_dim, ker_size, padding='same', kernel_regularizer=reg,
                        kernel_initializer='he_normal', name='conv'+str(block_index)+'_1')(inputs)
    net = layers.BatchNormalization(name='bn'+str(block_index)+'_1')(net)
    net = layers.Activation('relu')(net)
    net = non_local_layer(net, layer_index=block_index)
    net = layers.BatchNormalization(name='bn' + str(block_index) + '_2')(net)
    net = layers.Activation('relu')(net)
    return net

def cnl_block(inputs, block_index, out_dim, ker_size=[3, 3]):
    net = layers.Conv2D(out_dim, ker_size[0], padding='same', kernel_initializer='he_normal',
                        name='conv' + str(block_index) + '_1')(inputs)
    net = layers.BatchNormalization(name='bn' + str(block_index) + '_1')(net)
    net = layers.Activation('relu')(net)
    non_local = non_local_layer(net, block_index, inter_dim=out_dim // 2)
    local = layers.Conv2D(out_dim, ker_size[1], padding='same', kernel_initializer='he_normal',
                        name='conv' + str(block_index) + '_2')(
        net)
    net = layers.Add()([local, non_local])
    net = layers.BatchNormalization(name='bn' + str(block_index) + '_2')(net)
    net = layers.Activation('relu')(net)
    return net

def down_block_3D(inputs, block_index, out_dim, pool_size=2, ker_size=[3,3] ):
    net = U_block_3D(inputs, block_index, out_dim, ker_size)
    net = layers.MaxPooling3D(pool_size=pool_size)(net)
    return net

def up_block_3D(inputs, block_index, out_dim, pool_size=2, ker_size=[3,3] ):
    net = layers.Conv3D(out_dim, pool_size, padding='same', kernel_initializer='he_normal', name='deconv'+str(block_index))(
        layers.UpSampling3D(size=pool_size)(inputs))
    net = U_block_3D(net, block_index, out_dim, ker_size)
    return net


def unet_3d_bn(pretrained_weights=None, input_size=(64, 64, 64, 1)):
    inputs = layers.Input(input_size)
    down1 = down_block_3D(inputs, block_index=1, out_dim=64)
    down2 = down_block_3D(down1, block_index=2, out_dim=128)
    down3 = down_block_3D(down2, block_index=3, out_dim=256)
    bottom = U_block_3D(down3, block_index=4, out_dim=512)
    up3 = up_block_3D(bottom, block_index=5, out_dim=256)
    up2 = up_block_3D(up3, block_index=6, out_dim=128)
    up1 = up_block_3D(up2, block_index=7, out_dim=64)
    conv = layers.Conv3D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv_final')(
        up1)
    conv = layers.Conv3D(1, 1, activation='sigmoid', name='predict')(conv)
    model = Model(inputs=inputs, outputs=conv)

    model.compile(optimizer=optimizers.Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=0.01), loss=dice_loss, metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_2d_bn(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)
    #prep = Prepro()(inputs)
    #prep = layers.BatchNormalization(name='prep')(inputs)
    #prep = layers.Activation('tanh')(prep)
    down1 = U_block_2D(inputs, block_index=1, out_dim=64)
    pool1 = layers.MaxPooling2D(pool_size=2)(down1)
    down2 = U_block_2D(pool1, block_index=2, out_dim=128)
    pool2 = layers.MaxPooling2D(pool_size=2)(down2)
    down3 = U_block_2D(pool2, block_index=3, out_dim=256)
    pool3 = layers.MaxPooling2D(pool_size=2)(down3)
    down4 = U_block_2D(pool3, block_index=4, out_dim=512)
    pool4 = layers.MaxPooling2D(pool_size=2)(down4)
    bottom = U_block_2D(pool4, block_index=5, out_dim=1024)

    up4 = layers.Conv2D(512, 2, padding='same', kernel_initializer='he_normal',
                        name='deconv6')(
        layers.UpSampling2D(size=2)(bottom))
    up4 = layers.BatchNormalization(name='bn_u4')(up4)
    merge4 = layers.Concatenate(axis=3)([down4, up4])
    merge4 = U_block_2D(merge4, block_index=6, out_dim=512)

    up3 = layers.Conv2D(256, 2, padding='same', kernel_initializer='he_normal',
                        name='deconv7')(
        layers.UpSampling2D(size=2)(merge4))
    up3 = layers.BatchNormalization(name='bn_u3')(up3)
    merge3 = layers.Concatenate(axis=3)([down3, up3])
    merge3 = U_block_2D(merge3, block_index=7, out_dim=256)
    
    up2 = layers.Conv2D(128, 2, padding='same', kernel_initializer='he_normal',
                        name='deconv8')(
        layers.UpSampling2D(size=2)(merge3))
    up2 = layers.BatchNormalization(name='bn_u2')(up2)
    merge2 = layers.Concatenate(axis=3)([down2, up2])
    merge2 = U_block_2D(merge2, block_index=8, out_dim=128)

    up1 = layers.Conv2D(64, 2, padding='same', kernel_initializer='he_normal',
                        name='deconv9')(
        layers.UpSampling2D(size=2)(merge2))
    up1 = layers.BatchNormalization(name='bn_u1')(up1)

    merge1 = layers.Concatenate(axis=3)([down1, up1])
    merge1 = U_block_2D(merge1, block_index=9, out_dim=64)
    conv = layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv_final')(
        merge1)
    conv = layers.Conv2D(1, 1, activation='sigmoid', name='predict')(conv)
    model = Model(inputs=inputs, outputs=conv)

    model.compile(optimizer=optimizers.Adam(lr=2e-3), loss='binary_crossentropy',
                  metrics=[dice])
    #model.compile(optimizer=optimizers.Adam(lr=0.01), loss=dice_loss, metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

def unet_2d_a(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)
    pre = layers.Conv2D(64, 3, kernel_initializer='he_normal', 
                        padding='same', use_bias=False,
                        name='pre')(inputs)
    #pre = layers.BatchNormalization(name='bnpre')(pre)
    #pre = layers.Activation('relu')(pre)
    down1 = U_block_2D(pre, block_index=1, out_dim=64)
    pool1 = layers.MaxPooling2D(pool_size=2)(down1)
    down2 = U_block_2D(pool1, block_index=2, out_dim=128)
    pool2 = layers.MaxPooling2D(pool_size=2)(down2)
    down3 = U_block_2D(pool2, block_index=3, out_dim=256)
    pool3 = layers.MaxPooling2D(pool_size=2)(down3)
    down4 = U_block_2D(pool3, block_index=4, out_dim=512)
    pool4 = layers.MaxPooling2D(pool_size=2)(down4)
    bottom = U_block_2D(pool4, block_index=5, out_dim=1024)

    '''
    bota = layers.BatchNormalization(name='bnb')(bottom)
    bota = layers.Activation('relu')(bota)
    bota = layers.MaxPooling2D(pool_size=4)(bota)
    
    avep = layers.Conv2D(16, 1, padding='same', activation='relu',
                         name='amp')(bota)
    avep = layers.Flatten()(avep)
    #avep = layers.GlobalAveragePooling2D()(bottom)
    area = layers.Dense(256, kernel_initializer='he_normal', 
                        activation='relu', name='area')(avep)
    #area = layers.Dropout(0.5)(area)
    
    arp = layers.Dense(4, kernel_initializer='he_normal', 
                        name='areap')(area)
    crp = layers.Dense(4, kernel_initializer='he_normal', 
                        name='crp')(area)
    '''
    
    up4 = layers.Conv2DTranspose(512, 2, padding='same', kernel_initializer='he_normal',
                                 strides=2, name='deconv6')(bottom)
    #merge4 = layers.Concatenate(axis=3)([down4, up4])
    merge4 = U_block_2D([down4, up4], block_index=6, out_dim=512)

    up3 = layers.Conv2DTranspose(256, 2, padding='same', kernel_initializer='he_normal',
                                 strides=2, name='deconv7')(merge4)
    #merge3 = layers.Concatenate(axis=3)([down3, up3])
    merge3 = U_block_2D([down3, up3], block_index=7, out_dim=256)
    
    up2 = layers.Conv2DTranspose(128, 2, padding='same', kernel_initializer='he_normal',
                                 strides=2, name='deconv8')(merge3)
    
    #merge2 = layers.Concatenate(axis=3)([down2, up2])
    merge2 = U_block_2D([down2, up2], block_index=8, out_dim=128)

    up1 = layers.Conv2DTranspose(64, 2, padding='same', kernel_initializer='he_normal',
                                 strides=2, name='deconv9')(merge2)

    #merge1 = layers.Concatenate(axis=3)([down1, up1])
    merge1 = U_block_2D([down1, up1], block_index=9, out_dim=64)
    conv = layers.BatchNormalization(axis=3, name='bnf_2')(merge1)
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv2D(4, 1, activation='softmax', name='predict')(conv)
    
    '''
    pa = layers.GlobalAveragePooling2D()(conv)
    ap_loss = layers.Lambda(lambda x: (x[0] + tf.log(x[1]+1e-6))**2)([arp, pa])
    #ap_loss = layers.Lambda(lambda x: -tf.log(x+1e-6))(pa)
    
    dal = layers.MaxPooling2D(pool_size=3, strides=1, padding='same')(conv)
    pb = layers.GlobalAveragePooling2D()(dal)
    bp_loss = layers.Lambda(lambda x: (x[0] + tf.log((x[1]-x[2]+1e-6)/(x[1]+1e-6)))**2)([crp, pb, pa])
    #bp_loss = layers.Lambda(lambda x: -tf.log((x[1]-x[0]+1e-6)/(x[1]+1e-6)))([pa, pb])
    
    mmodel = Model(inputs=inputs, outputs=[conv, arp, crp, ap_loss, bp_loss])
    mmodel.compile(optimizer=optimizers.Adam(1e-3), 
                   loss=['categorical_crossentropy', 'mse', 'mse', lambda y_true,y_pred: y_pred,
                         lambda y_true,y_pred: y_pred],
                   metrics={'predict':m_dice}, loss_weights=[1, 0.1, 0.1, 0.1, 0.1])
    '''
    mmodel = Model(inputs=inputs, outputs=conv)

    mmodel.compile(optimizer=optimizers.Adam(lr=1e-3), loss='binary_crossentropy',
                  metrics=[m_dice])
    '''
    mmodel = Model(inputs=inputs, outputs=[conv, ap_loss, bp_loss])
    mmodel.compile(optimizer=optimizers.Adam(1e-3), 
                   loss=['binary_crossentropy', 'mse', 'mse'],
                   metrics={'predict':dice}, loss_weights=[1, 0.1, 0.1])
    
    '''
    if (pretrained_weights):
        mmodel.load_weights(pretrained_weights)

    return mmodel

def unet_2d_b(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)
    pre = layers.Conv2D(64, 3, kernel_initializer='he_normal', 
                        padding='same', use_bias=False,
                        name='pre')(inputs)
    
    down1 = U_block_2D(pre, block_index=1, out_dim=64)
    pool1 = layers.MaxPooling2D(pool_size=2)(down1)
    down2 = U_block_2D(pool1, block_index=2, out_dim=128)
    pool2 = layers.MaxPooling2D(pool_size=2)(down2)
    down3 = U_block_2D(pool2, block_index=3, out_dim=256)
    pool3 = layers.MaxPooling2D(pool_size=2)(down3)
    down4 = U_block_2D(pool3, block_index=4, out_dim=512)
    pool4 = layers.MaxPooling2D(pool_size=2)(down4)
    bottom = U_block_2D(pool4, block_index=5, out_dim=1024)

    
    bota = layers.BatchNormalization(name='bnb')(bottom)
    bota = layers.Activation('relu')(bota)
    bota = layers.MaxPooling2D(pool_size=4)(bota)
    
    avep = layers.Conv2D(16, 1, padding='same', activation='relu',
                         name='amp')(bota)
    avep = layers.Flatten()(avep)
    #avep = layers.GlobalAveragePooling2D()(bottom)
    area = layers.Dense(256, kernel_initializer='he_normal', 
                        activation='relu', name='area')(avep)
    #area = layers.Dropout(0.5)(area)
    
    arp = layers.Dense(1, kernel_initializer='he_normal', 
                        name='areap')(area)
    crp = layers.Dense(1, kernel_initializer='he_normal', 
                        name='crp')(area)
    
    up4 = layers.Conv2DTranspose(512, 2, padding='same', kernel_initializer='he_normal',
                                 strides=2, name='deconv6')(bottom)
    #merge4 = layers.Concatenate(axis=3)([down4, up4])
    merge4 = U_block_2D([down4, up4], block_index=6, out_dim=512)

    up3 = layers.Conv2DTranspose(256, 2, padding='same', kernel_initializer='he_normal',
                                 strides=2, name='deconv7')(merge4)
    #merge3 = layers.Concatenate(axis=3)([down3, up3])
    merge3 = U_block_2D([down3, up3], block_index=7, out_dim=256)
    
    up2 = layers.Conv2DTranspose(128, 2, padding='same', kernel_initializer='he_normal',
                                 strides=2, name='deconv8')(merge3)
    
    #merge2 = layers.Concatenate(axis=3)([down2, up2])
    merge2 = U_block_2D([down2, up2], block_index=8, out_dim=128)

    up1 = layers.Conv2DTranspose(64, 2, padding='same', kernel_initializer='he_normal',
                                 strides=2, name='deconv9')(merge2)

    #merge1 = layers.Concatenate(axis=3)([down1, up1])
    merge1 = U_block_2D([down1, up1], block_index=9, out_dim=64)
    conv = layers.BatchNormalization(axis=3, name='bnf_2')(merge1)
    conv = layers.Activation('relu')(conv)
    conv = layers.Conv2D(1, 1, activation='sigmoid', name='predict')(conv)
    
    
    pa = layers.GlobalAveragePooling2D()(conv)
    ap_loss = layers.Lambda(lambda x: (x[0] + tf.log(x[1]+1e-6))**2)([arp, pa])
    #ap_loss = layers.Lambda(lambda x: -tf.log(x+1e-6))(pa)
    
    dal = layers.MaxPooling2D(pool_size=3, strides=1, padding='same')(conv)
    pb = layers.GlobalAveragePooling2D()(dal)
    bp_loss = layers.Lambda(lambda x: (x[0] + tf.log((x[1]-x[2]+1e-6)/(x[1]+1e-6)))**2)([crp, pb, pa])
    #bp_loss = layers.Lambda(lambda x: -tf.log((x[1]-x[0]+1e-6)/(x[1]+1e-6)))([pa, pb])
    
    mmodel = Model(inputs=inputs, outputs=[conv, arp, crp, ap_loss, bp_loss])
    mmodel.compile(optimizer=optimizers.Adam(1e-3), 
                   loss=['binary_crossentropy', 'mse', 'mse', lambda y_true,y_pred: y_pred,
                         lambda y_true,y_pred: y_pred],
                   metrics={'predict':dice}, loss_weights=[1, 0.1, 0.1, 0.1, 0.1])
    '''
    mmodel = Model(inputs=inputs, outputs=conv)

    mmodel.compile(optimizer=optimizers.Adam(lr=1e-3), loss='binary_crossentropy',
                  metrics=[dice])
    
    mmodel = Model(inputs=inputs, outputs=[conv, ap_loss, bp_loss])
    mmodel.compile(optimizer=optimizers.Adam(1e-3), 
                   loss=['binary_crossentropy', 'mse', 'mse'],
                   metrics={'predict':dice}, loss_weights=[1, 0.1, 0.1])
    
    '''
    if (pretrained_weights):
        mmodel.load_weights(pretrained_weights)

    return mmodel


def dense_unet(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = layers.Input(input_size)
    down1 = dense_block(inputs, block_index=1, conv_dim=32)
    pool1 = layers.MaxPooling2D(pool_size=2)(down1)
    down2 = dense_block(pool1, block_index=2, conv_dim=32)
    pool2 = layers.MaxPooling2D(pool_size=2)(down2)
    down3 = dense_block(pool2, block_index=3, conv_dim=32)
    pool3 = layers.MaxPooling2D(pool_size=2)(down3)
    down4 = dense_block(pool3, block_index=4, conv_dim=64)
    pool4 = layers.MaxPooling2D(pool_size=2)(down4)
    bottom = dense_block(pool4, block_index=5, conv_dim=128)

    up4 = layers.Conv2D(64, 2, padding='same', kernel_initializer='he_normal',
                        name='deconv6')(
        layers.UpSampling2D(size=2)(bottom))
    merge4 = layers.Concatenate(axis=3)([down4, up4])
    merge4 = dense_block(merge4, block_index=6, conv_dim=128)

    up3 = layers.Conv2D(32, 2, padding='same', kernel_initializer='he_normal',
                        name='deconv7')(
        layers.UpSampling2D(size=2)(merge4))
    merge3 = layers.Concatenate(axis=3)([down3, up3])
    merge3 = dense_block(merge3, block_index=7, conv_dim=128)

    up2 = layers.Conv2D(32, 2, padding='same', kernel_initializer='he_normal',
                        name='deconv8')(
        layers.UpSampling2D(size=2)(merge3))
    merge2 = layers.Concatenate(axis=3)([down2, up2])
    merge2 = dense_block(merge2, block_index=8, conv_dim=64)

    up1 = layers.Conv2D(32, 2, padding='same', kernel_initializer='he_normal',
                        name='deconv9')(
        layers.UpSampling2D(size=2)(merge2))

    merge1 = layers.Concatenate(axis=3)([down1, up1])
    merge1 = dense_block(merge1, block_index=9, conv_dim=32)
    conv = layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal', name='conv_final')(
        merge1)
    conv = layers.Conv2D(1, 1, activation='sigmoid', name='predict')(conv)
    model = Model(inputs=inputs, outputs=conv)

    model.compile(optimizer=optimizers.Adam(lr=1e-2), loss='binary_crossentropy',
                  metrics=['accuracy', dice])
    #model.compile(optimizer=optimizers.Adam(lr=0.01), loss=dice_loss, metrics=['accuracy'])

    # model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model



def ud_block(inputs, c_dim, out=['u', 'c', 'd'], index='', reg=reg, hidden=True):
    if isinstance(inputs, list):
        net = layers.Concatenate()(inputs)
    else:
        net = inputs
    net = layers.BatchNormalization(name='bn'+index)(net)
    net = layers.Activation('relu')(net)
    net = layers.Conv2D(c_dim, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='node'+index)(net)
    net = layers.BatchNormalization(name='bnn'+index)(net)
    net = layers.Activation('relu')(net)
    if hidden:
        outputs = []
    else:
        outputs = [net]
    if 'u' in out:
        up = layers.Conv2D(c_dim//2, 2, kernel_initializer='he_normal', padding='same',
                          use_bias=False, kernel_regularizer=reg,
                          name='up'+index)(layers.UpSampling2D()(net))
        outputs.append(up)
    if 'c' in out:
        conv = layers.Conv2D(c_dim, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='conv'+index)(net)
        outputs.append(conv)
    if 'd' in out:
        down = layers.Conv2D(c_dim*2, 2, padding='valid', kernel_initializer='he_normal',
                            use_bias=False, kernel_regularizer=reg,
                            strides=2, name='down'+index)(net)
        outputs.append(down)

    return outputs

def ud_block_v2(inputs, c_dim, out=['u', 'c', 'd'], index='', reg=reg):
    if isinstance(inputs, list):
        short = layers.Add()(inputs)
    else:
        short = inputs
    net = layers.Activation('relu')(short)
    net = layers.Conv2D(c_dim, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='node'+index)(net)
    net = layers.BatchNormalization(name='bnn'+index)(net)
    net = layers.Activation('relu')(net)
    net = layers.Conv2D(c_dim, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='nodep'+index)(net)
    net = layers.BatchNormalization(name='bnp'+index)(net)
    net = layers.Add()([net, short])
    net = layers.Activation('relu')(net)
    outputs = [net]
    
    if 'u' in out:
        up = layers.Conv2D(c_dim//2, 2, kernel_initializer='he_normal', padding='same',
                          use_bias=False, kernel_regularizer=reg,
                          name='up'+index)(layers.UpSampling2D()(net))
        up = layers.BatchNormalization(name='bup'+index)(up)
        outputs.append(up)
    if 'c' in out:
        conv = layers.Conv2D(c_dim, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='conv'+index)(net)
        conv = layers.BatchNormalization(name='bconv'+index)(conv)
        outputs.append(conv)
    if 'd' in out:
        down = layers.Conv2D(c_dim*2, 2, padding='valid', kernel_initializer='he_normal',
                            use_bias=False, kernel_regularizer=reg,
                            strides=2, name='down'+index)(net)
        down = layers.BatchNormalization(name='bdown'+index)(down)
        outputs.append(down)

    return outputs

def ud_block_v3(inputs, c_dim, out=['u', 'c', 'd'], index='', reg=reg):
    if isinstance(inputs, list):
        net = layers.Concatenate()(inputs)
        net = layers.Conv2D(c_dim, 1, padding='same', kernel_initializer='he_normal',
                            use_bias=False, kernel_regularizer=reg,
                            name='pre_'+index)(net)
    else:
        net = inputs
    left = layers.Activation('relu')(net)
    left = layers.Conv2D(c_dim//4, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='node1_'+index)(left)
    left = layers.BatchNormalization(name='bn1_'+index)(left)
    left = layers.Activation('relu')(left)
    left = layers.SpatialDropout2D(0.5)(left)
    left = layers.Conv2D(c_dim, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='node2_'+index)(left)
    left = layers.BatchNormalization(name='bn2_'+index)(left)
    net = layers.Add()([net, left])
    '''
    left = layers.Activation('relu')(net)
    left = layers.SpatialDropout2D(0.5)(left)
    left = layers.Conv2D(c_dim//4, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='node3_'+index)(left)
    left = layers.BatchNormalization(name='bn3_'+index)(left)
    left = layers.Activation('relu')(left)
    left = layers.Conv2D(c_dim, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='node4_'+index)(left)
    left = layers.BatchNormalization(name='bn4_'+index)(left)
    net = layers.Add()([net, left])
    '''
    net = layers.Activation('relu')(net)
    outputs = [net]
    
    if 'u' in out:
        #up = layers.Conv2D(c_dim//2, 2, kernel_initializer='he_normal', padding='same',
        #                  use_bias=False, kernel_regularizer=reg,
        #                  name='up'+index)(layers.UpSampling2D()(net))
        up = layers.Conv2DTranspose(c_dim//2, 2, kernel_initializer='he_normal', 
                                    padding='same', strides=2,
                                    use_bias=False, kernel_regularizer=reg,
                                    name='up'+index)(net)
        up = layers.BatchNormalization(name='bup'+index)(up)
        outputs.append(up)
    if 'c' in out:
        conv = layers.Conv2D(c_dim, 1, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='conv'+index)(net)
        conv = layers.BatchNormalization(name='bconv'+index)(conv)
        outputs.append(conv)
    if 'd' in out:
        down = layers.Conv2D(c_dim*2, 2, padding='valid', kernel_initializer='he_normal',
                            use_bias=False, kernel_regularizer=reg,
                            strides=2, name='down'+index)(net)
        down = layers.BatchNormalization(name='bdown'+index)(down)
        outputs.append(down)

    return outputs

def denseud_byb(pretrianed=None, input_shape=(256,256,1)):
    inputs = layers.Input(input_shape)
    conv_pre = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='conv_pre')(inputs)
    conv1_1, down1_1 = ud_block(conv_pre, 32, out=['c', 'd'], index='1_1')
    up2_1, conv2_1, down2_1 = ud_block(down1_1, 64, index='2_1')
    up3_1, conv3_1, down3_1 = ud_block(down2_1, 128, index='3_1')
    up4_1, conv4_1, down4_1 = ud_block(down3_1, 256, index='4_1')
    up5_1 = ud_block(down4_1, 512, out=['u'], index='5_1')[0]

    conv1_2, down1_2 = ud_block([conv1_1, up2_1], 32, out=['c', 'd'], index='1_2')
    up2_2, conv2_2, down2_2 = ud_block([down1_2, conv2_1, up3_1], 64, out=['u', 'c', 'd'], index='2_2')
    up3_2, conv3_2, down3_2 = ud_block([down2_2, conv3_1, up4_1], 128, out=['u', 'c', 'd'], index='3_2')
    up4_2 = ud_block([down3_2, conv4_1, up5_1], 256, out=['u'], index='4_2')[0]

    conv1_3, down1_3 = ud_block([conv1_2, up2_2], 32, out=['c', 'd'], index='1_3')
    up2_3, conv2_3, down2_3 = ud_block([down1_3, conv2_2, up3_2], 64, out=['u', 'c', 'd'], index='2_3')
    up3_3 = ud_block([down2_3, conv3_2, up4_2], 128, out=['u'], index='3_3')[0]

    conv1_4, down1_4 = ud_block([conv1_3, up2_3], 32, out=['c', 'd'], index='1_4')
    up2_4= ud_block([down1_4, conv2_3, up3_3], 64, out=['u'], index='2_4')[0]

    node1_5 = ud_block([conv1_4, up2_4], 32, out=[], index='1_5', hidden=False)[0]

    predict = layers.Conv2D(1, 1, kernel_initializer='he_normal', activation='sigmoid',
                            padding='same', name='predict')(node1_5)

    mmodel = Model(inputs=inputs, outputs=predict)
    mmodel.compile(optimizer=optimizers.Adam(1e-3), loss='binary_crossentropy',
                   metrics=[dice])

    if pretrianed:
        mmodel.load_weights(pretrianed)

    return mmodel

def denseud_byb_v2(pretrianed=None, input_shape=(256,256,1)):
    inputs = layers.Input(input_shape)
    conv_pre = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='conv_pre')(inputs)
    conv_pre = layers.BatchNormalization(name='bn_pre')(conv_pre)
    _, conv1_1, down1_1 = ud_block_v3(conv_pre, 32, out=['c', 'd'], index='1_1')
    _, up2_1, conv2_1, down2_1 = ud_block_v3(down1_1, 64, index='2_1')
    _, up3_1, conv3_1, down3_1 = ud_block_v3(down2_1, 128, index='3_1')
    _, up4_1, conv4_1, down4_1 = ud_block_v3(down3_1, 256, index='4_1')
    node5_1, up5_1 = ud_block_v3(down4_1, 512, out=['u'], index='5_1')

    _, conv1_2, down1_2 = ud_block_v3([conv1_1, up2_1], 32, out=['c', 'd'], index='1_2')
    _, up2_2, conv2_2, down2_2 = ud_block_v3([down1_2, conv2_1, up3_1], 64, out=['u', 'c', 'd'], index='2_2')
    _, up3_2, conv3_2, down3_2 = ud_block_v3([down2_2, conv3_1, up4_1], 128, out=['u', 'c', 'd'], index='3_2')
    _, up4_2 = ud_block_v3([down3_2, conv4_1, up5_1], 256, out=['u'], index='4_2')

    _, conv1_3, down1_3 = ud_block_v3([conv1_2, up2_2], 32, out=['c', 'd'], index='1_3')
    _, up2_3, conv2_3, down2_3 = ud_block_v3([down1_3, conv2_2, up3_2], 64, out=['u', 'c', 'd'], index='2_3')
    _, up3_3 = ud_block_v3([down2_3, conv3_2, up4_2], 128, out=['u'], index='3_3')

    _, conv1_4, down1_4 = ud_block_v3([conv1_3, up2_3], 32, out=['c', 'd'], index='1_4')
    _, up2_4= ud_block_v3([down1_4, conv2_3, up3_3], 64, out=['u'], index='2_4')

    node1_5 = ud_block_v3([conv1_4, up2_4], 32, out=[], index='1_5')[0]
    
    avep = layers.GlobalAveragePooling2D()(node5_1)
    area = layers.Dense(1, kernel_initializer='he_normal', 
                        activation='sigmoid', name='areap')(avep)
    #apre = layers.Lambda(lambda x: -tf.log(x))(area)

    predict = layers.Conv2D(1, 1, kernel_initializer='he_normal', activation='sigmoid',
                            padding='same', name='predict')(node1_5)
    
    pa = layers.GlobalAveragePooling2D()(predict)
    ap_loss = layers.Lambda(lambda x: (x[0] - x[1])**2)([area, pa])

    mmodel = Model(inputs=inputs, outputs=[predict, area, ap_loss])
    mmodel.compile(optimizer=optimizers.Adam(1e-3), 
                   loss=['binary_crossentropy', 'mse', lambda y_true,y_pred: y_pred],
                   metrics={'predict':dice}, loss_weights=[1, 10, 10])

    if pretrianed:
        mmodel.load_weights(pretrianed)

    return mmodel

def denseud_byb_mo(pretrianed=None, input_shape=(256,256,1)):
    inputs = layers.Input(input_shape)
    conv_pre = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='conv_pre')(inputs)
    conv1_1, down1_1 = ud_block(conv_pre, 32, out=['c', 'd'], index='1_1')
    up2_1, conv2_1, down2_1 = ud_block(down1_1, 64, index='2_1')
    up3_1, conv3_1, down3_1 = ud_block(down2_1, 128, index='3_1')
    up4_1, conv4_1, down4_1 = ud_block(down3_1, 256, index='4_1')
    up5_1 = ud_block(down4_1, 512, out=['u'], index='5_1')[0]

    node1_2, conv1_2, down1_2 = ud_block([conv1_1, up2_1], 32, out=['c', 'd'], index='1_2', hidden=False)
    up2_2, conv2_2, down2_2 = ud_block([down1_2, conv2_1, up3_1], 64, out=['u', 'c', 'd'], index='2_2')
    up3_2, conv3_2, down3_2 = ud_block([down2_2, conv3_1, up4_1], 128, out=['u', 'c', 'd'], index='3_2')
    up4_2 = ud_block([down3_2, conv4_1, up5_1], 256, out=['u'], index='4_2')[0]

    node1_3, conv1_3, down1_3 = ud_block([conv1_2, up2_2], 32, out=['c', 'd'], index='1_3', hidden=False)
    up2_3, conv2_3, down2_3 = ud_block([down1_3, conv2_2, up3_2], 64, out=['u', 'c', 'd'], index='2_3')
    up3_3 = ud_block([down2_3, conv3_2, up4_2], 128, out=['u'], index='3_3')[0]

    node1_4, conv1_4, down1_4 = ud_block([conv1_3, up2_3], 32, out=['c', 'd'], index='1_4', hidden=False)
    up2_4= ud_block([down1_4, conv2_3, up3_3], 64, out=['u'], index='2_4')[0]

    node1_5 = ud_block([conv1_4, up2_4], 32, out=[], index='1_5', hidden=False)[0]

    predict5 = layers.Conv2D(1, 1, kernel_initializer='he_normal', activation='sigmoid',
                            padding='same', name='predict5')(node1_5)
    predict4 = layers.Conv2D(1, 1, kernel_initializer='he_normal', activation='sigmoid',
                            padding='same', name='predict4')(node1_4)
    predict3 = layers.Conv2D(1, 1, kernel_initializer='he_normal', activation='sigmoid',
                            padding='same', name='predict3')(node1_3)

    mmodel = Model(inputs=inputs, outputs=[predict5, predict4, predict3])
    mmodel.compile(optimizer=optimizers.Adam(1e-3), loss='binary_crossentropy',
                   metrics=[dice], loss_weights=[1.0, 0.5, 0.25])

    if pretrianed:
        mmodel.load_weights(pretrianed)

    return mmodel

def denseud_u(pretrianed=None, input_shape=(128,128,1)):
    inputs = layers.Input(input_shape)
    conv_pre = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='conv_pre')(inputs)
    up1_1, conv1_1, down1_1 = ud_block(conv_pre, 32, index='1_1')
    up2_1, conv2_1, down2_1 = ud_block(down1_1, 64, index='2_1')
    up3_1, conv3_1, down3_1 = ud_block(down2_1, 128, index='3_1')
    up4_1, conv4_1, down4_1 = ud_block(down3_1, 256, index='4_1')
    up5_1 = ud_block(down4_1, 512, out=['u'], index='5_1')[0]
    convu1_1, downu1_1 = ud_block(up1_1, 16, out=['c', 'd'], index='u1_1')

    up1_2, conv1_2, down1_2 = ud_block([downu1_1, conv1_1, up2_1], 32, index='1_2')
    up2_2, conv2_2, down2_2 = ud_block([down1_2, conv2_1, up3_1], 64, out=['u', 'c', 'd'], index='2_2')
    up3_2, conv3_2, down3_2 = ud_block([down2_2, conv3_1, up4_1], 128, out=['u', 'c', 'd'], index='3_2')
    up4_2 = ud_block([down3_2, conv4_1, up5_1], 256, out=['u'], index='4_2')[0]
    convu1_2, downu1_2 = ud_block([convu1_1, up1_2], 16, out=['c', 'd'], index='u1_2')

    up1_3, conv1_3, down1_3 = ud_block([downu1_2, conv1_2, up2_2], 32, index='1_3')
    up2_3, conv2_3, down2_3 = ud_block([down1_3, conv2_2, up3_2], 64, out=['u', 'c', 'd'], index='2_3')
    up3_3 = ud_block([down2_3, conv3_2, up4_2], 128, out=['u'], index='3_3')[0]
    convu1_3, downu1_3 = ud_block([convu1_2, up1_3], 16, out=['c', 'd'], index='u1_3')

    up1_4, conv1_4, down1_4 = ud_block([downu1_3, conv1_3, up2_3], 32, index='1_4')
    up2_4= ud_block([down1_4, conv2_3, up3_3], 64, out=['u'], index='2_4')[0]
    downu1_4 = ud_block([convu1_3, up1_4], 16, out=['d'], index='u1_4')[0]

    node1_5 = ud_block([downu1_4, conv1_4, up2_4], 32, out=[], index='1_5', hidden=False)[0]

    predict = layers.Conv2D(1, 1, kernel_initializer='he_normal', activation='sigmoid',
                            padding='same', name='predict')(node1_5)

    mmodel = Model(inputs=inputs, outputs=predict)
    mmodel.compile(optimizer=optimizers.Adam(1e-3), loss='binary_crossentropy',
                   metrics=[dice])

    if pretrianed:
        mmodel.load_weights(pretrianed)

    return mmodel


def denseud_byb_mo_v2(pretrianed=None, input_shape=(256,256,1)):
    inputs = layers.Input(input_shape)
    conv_pre = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='conv_pre')(inputs)
    conv_pre = layers.BatchNormalization(name='bn_pre')(conv_pre)
    _, up1_1, conv1_1, down1_1 = ud_block_v3(conv_pre, 32, index='1_1')
    _, up2_1, conv2_1, down2_1 = ud_block_v3(down1_1, 64, index='2_1')
    _, up3_1, conv3_1, down3_1 = ud_block_v3(down2_1, 128, index='3_1')
    _, up4_1, conv4_1, down4_1 = ud_block_v3(down3_1, 256, index='4_1')
    _, up5_1 = ud_block_v3(down4_1, 512, out=['u'], index='5_1')

    node1_2, conv1_2, down1_2 = ud_block_v3([conv1_1, up2_1], 32, out=['c', 'd'], index='1_2')
    _, up2_2, conv2_2, down2_2 = ud_block_v3([down1_2, conv2_1, up3_1], 64, out=['u', 'c', 'd'], index='2_2')
    _, up3_2, conv3_2, down3_2 = ud_block_v3([down2_2, conv3_1, up4_1], 128, out=['u', 'c', 'd'], index='3_2')
    _, up4_2 = ud_block_v3([down3_2, conv4_1, up5_1], 256, out=['u'], index='4_2')

    node1_3, conv1_3, down1_3 = ud_block_v3([conv1_2, up2_2], 32, out=['c', 'd'], index='1_3')
    _, up2_3, conv2_3, down2_3 = ud_block_v3([down1_3, conv2_2, up3_2], 64, out=['u', 'c', 'd'], index='2_3')
    _, up3_3 = ud_block_v3([down2_3, conv3_2, up4_2], 128, out=['u'], index='3_3')

    node1_4, conv1_4, down1_4 = ud_block_v3([conv1_3, up2_3], 32, out=['c', 'd'], index='1_4')
    _, up2_4= ud_block_v3([down1_4, conv2_3, up3_3], 64, out=['u'], index='2_4')

    node1_5 = ud_block_v3([conv1_4, up2_4], 32, out=[], index='1_5')[0]

    predict2 = layers.Conv2D(1, 1, kernel_initializer='he_normal', activation='sigmoid',
                            padding='same', name='predict2')(node1_2)
    predict5 = layers.Conv2D(1, 1, kernel_initializer='he_normal', activation='sigmoid',
                            padding='same', name='predict5')(node1_5)
    predict4 = layers.Conv2D(1, 1, kernel_initializer='he_normal', activation='sigmoid',
                            padding='same', name='predict4')(node1_4)
    predict3 = layers.Conv2D(1, 1, kernel_initializer='he_normal', activation='sigmoid',
                            padding='same', name='predict3')(node1_3)
    

    mmodel = Model(inputs=inputs, outputs=[predict5, predict4, predict3, predict2])
    mmodel.compile(optimizer=optimizers.Adam(1e-3), loss='binary_crossentropy',
                   metrics=[dice], loss_weights=[1,0.5,0.25,0.125])

    if pretrianed:
        mmodel.load_weights(pretrianed)

    return mmodel

def denseud_u_mo_v2(pretrianed=None, input_shape=(256,256,1)):
    inputs = layers.Input(input_shape)
    conv_pre = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='conv_pre')(inputs)
    conv_pre = layers.BatchNormalization(name='bn_pre')(conv_pre)
    _, up1_1, conv1_1, down1_1 = ud_block_v3(conv_pre, 32, index='1_1')
    _, up2_1, conv2_1, down2_1 = ud_block_v3(down1_1, 64, index='2_1')
    _, up3_1, conv3_1, down3_1 = ud_block_v3(down2_1, 128, index='3_1')
    _, up4_1 = ud_block_v3(down3_1, 256,out=['u'], index='4_1')
    _, convu1_1, downu1_1 = ud_block_v3(up1_1, 16, out=['c', 'd'], index='u1_1')

    node1_2, up1_2, conv1_2, down1_2 = ud_block_v3([downu1_1, conv1_1, up2_1], 32, index='1_2')
    _, up2_2, conv2_2, down2_2 = ud_block_v3([down1_2, conv2_1, up3_1], 64, out=['u', 'c', 'd'], index='2_2')
    _, up3_2 = ud_block_v3([down2_2, conv3_1, up4_1], 128, out=['u'], index='3_2')
    _, convu1_2, downu1_2 = ud_block_v3([convu1_1, up1_2], 16, out=['c', 'd'], index='u1_2')

    node1_3, up1_3, conv1_3, down1_3 = ud_block_v3([downu1_2, conv1_2, up2_2], 32, index='1_3')
    _, up2_3 = ud_block_v3([down1_3, conv2_2, up3_2], 64, out=['u'], index='2_3')
    _, downu1_3 = ud_block_v3([convu1_2, up1_3], 16, out=['d'], index='u1_3')
    

    node1_4 = ud_block_v3([downu1_3, conv1_3, up2_3], 32, out=[], index='1_4')[0]

    predict2 = layers.Conv2D(1, 1, kernel_initializer='he_normal', activation='sigmoid',
                            padding='same', name='predict2')(node1_2)
    predict4 = layers.Conv2D(1, 1, kernel_initializer='he_normal', activation='sigmoid',
                            padding='same', name='predict4')(node1_4)
    predict3 = layers.Conv2D(1, 1, kernel_initializer='he_normal', activation='sigmoid',
                            padding='same', name='predict3')(node1_3)
    

    mmodel = Model(inputs=inputs, outputs=[predict4, predict3, predict2])
    mmodel.compile(optimizer=optimizers.Adam(1e-3), loss='binary_crossentropy',
                   metrics=[dice], loss_weights=[1,0.5,0.25])

    if pretrianed:
        mmodel.load_weights(pretrianed)

    return mmodel


def res_down(inputs, out_dim, stride, index):
    short = layers.Conv2D(out_dim, 1, kernel_initializer='he_normal',
                            use_bias=False, kernel_regularizer=reg,
                            strides=stride, name='short'+index)(inputs)
    left = layers.BatchNormalization(name='bn0_'+index)(inputs)
    left = layers.Activation('relu')(left)
    left = layers.Conv2D(out_dim//4, 3, padding='same', kernel_initializer='he_normal',
                            use_bias=False, kernel_regularizer=reg,
                            strides=stride, name='leftu'+index)(left)
    left = layers.BatchNormalization(name='bn1_'+index)(left)
    left = layers.Activation('relu')(left)
    left = layers.Conv2D(out_dim, 3, padding='same', kernel_initializer='he_normal',
                            use_bias=False, kernel_regularizer=reg,
                            name='leftd'+index)(left)
    return layers.Add()([short, left])

def res_v2(inputs, dim,index, reg, dilation_rate=1):
    net = layers.BatchNormalization(name='bn1_'+index)(inputs)
    net = layers.Activation('relu')(net)
    net = layers.Conv2D(dim//4, 1, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='conv1_'+index)(net)
    net = layers.BatchNormalization(name='bn2_'+index)(net)
    net = layers.Activation('relu')(net)
    net = layers.Conv2D(dim//4, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='conv2_'+index, dilation_rate=dilation_rate)(net)
    net = layers.BatchNormalization(name='bn3_'+index)(net)
    net = layers.Activation('relu')(net)
    net = layers.Conv2D(dim, 1, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='conv3_'+index)(net)
    return net

def ms_res_b(f1, f2, f3, dim, index, dilation_rate=1, reg=reg):
    res1 = res_v2(f1, dim, index+'_1', reg=reg, dilation_rate=dilation_rate)
    res2 = res_v2(f2, dim, index+'_2', reg=reg, dilation_rate=dilation_rate)
    res3 = res_v2(f3, dim, index+'_3', reg=reg, dilation_rate=dilation_rate)
    f1 = layers.Add()([f1, res1])
    f2 = layers.Add()([f2, res2])
    f3 = layers.Add()([f3, res3])
    
    res1 = res_v2(f1, dim, index+'_4', reg=reg)
    res2 = res_v2(f2, dim, index+'_5', reg=reg)
    res3 = res_v2(f3, dim, index+'_6', reg=reg)
    res1_2 = layers.AveragePooling2D()(res1)
    res1_3 = layers.AveragePooling2D(4)(res1)
    res2_3 = layers.AveragePooling2D()(res2)
    res2_1 = layers.UpSampling2D()(res2)
    res3_1 = layers.UpSampling2D(4)(res3)
    res3_2 = layers.UpSampling2D()(res3)
    
    f1 = layers.Add()([f1, res1, res2_1, res3_1])
    f2 = layers.Add()([f2, res2, res1_2, res3_2])
    f3 = layers.Add()([f3, res3, res1_3, res2_3])
    return f1, f2, f3

def aspp(inputs, dim, index, reg):
    net = layers.BatchNormalization(name='bn1_'+index)(inputs)
    net = layers.Activation('relu')(net)
    block1 = layers.Conv2D(dim, 1, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg,
                             name='conv1_'+index)(net)
    block2 = layers.Conv2D(dim, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg, dilation_rate=6,
                             name='conv2_'+index)(net)
    block3 = layers.Conv2D(dim, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg, dilation_rate=12,
                             name='conv3_'+index)(net)
    block4 = layers.Conv2D(dim, 3, padding='same', kernel_initializer='he_normal',
                             use_bias=False, kernel_regularizer=reg, dilation_rate=18,
                             name='conv4_'+index)(net)
    
    pool = layers.AveragePooling2D(32)(net)
    pool = layers.Conv2D(dim, 1, padding='same', kernel_initializer='he_normal',
                         use_bias=False, kernel_regularizer=reg,
                         name='pool_'+index)(pool)
    pool = layers.UpSampling2D(32)(pool)
    
    output = layers.Concatenate()([block1, block2, block3, block4, pool])
    output = layers.BatchNormalization(name='bn2_'+index)(output)
    output = layers.Activation('relu')(output)
    resb = layers.Conv2D(dim, 1, padding='same', kernel_initializer='he_normal',
                         use_bias=False, kernel_regularizer=reg,
                         name='final_'+index)(output)
    return resb

def ms_res(pretrianed=None, input_shape=(256,256,1)):
    inputs = layers.Input(input_shape)
    fea1 = layers.Conv2D(32, 3, padding='same', kernel_initializer='he_normal',
                         use_bias=False, kernel_regularizer=reg,
                         name='conv_pre')(inputs)
    fea2 = res_down(fea1, 32, 2, '_d2')
    fea3 = res_down(fea2, 32, 2, '_d3')
    #fea4 = res_down(fea3, 128, 2, '_d4')
    #fea5 = res_down(fea4, 128, 2, '_d5')
    #fea6 = res_down(fea5, 128, 2, '_d6')
    fea1, fea2, fea3 = ms_res_b(fea1, fea2, fea3, 32, '_1')
    fea1, fea2, fea3 = ms_res_b(fea1, fea2, fea3, 32, '_2', dilation_rate=2)
    fea1, fea2, fea3 = ms_res_b(fea1, fea2, fea3, 32, '_3', dilation_rate=4)
    fea1, fea2, fea3 = ms_res_b(fea1, fea2, fea3, 32, '_4', dilation_rate=8)
    fea1, fea2, fea3 = ms_res_b(fea1, fea2, fea3, 32, '_5')
    fea1, fea2, fea3 = ms_res_b(fea1, fea2, fea3, 32, '_6', dilation_rate=2)
    fea1, fea2, fea3 = ms_res_b(fea1, fea2, fea3, 32, '_7', dilation_rate=4)
    fea1, fea2, fea3 = ms_res_b(fea1, fea2, fea3, 32, '_8', dilation_rate=8)
    fea1, fea2, fea3 = ms_res_b(fea1, fea2, fea3, 32, '_15')
    fea1, fea2, fea3 = ms_res_b(fea1, fea2, fea3, 32, '_16', dilation_rate=2)
    fea1, fea2, fea3 = ms_res_b(fea1, fea2, fea3, 32, '_17', dilation_rate=4)
    fea1, fea2, fea3 = ms_res_b(fea1, fea2, fea3, 32, '_18', dilation_rate=8)
    fea1, _, _ = ms_res_b(fea1, fea2, fea3, 32, '_9')
    #fea4, fea5, fea6 = ms_res_b(fea4, fea5, fea6, 256, '_5')
    #fea4, fea5, fea6 = ms_res_b(fea4, fea5, fea6, 256, '_6', dilation_rate=2)
    #fea4, fea5, fea6 = ms_res_b(fea4, fea5, fea6, 256, '_7', dilation_rate=4)
    #fea4, fea5, fea6 = ms_res_b(fea4, fea5, fea6, 128, '_8')
    #fea4, fea5, fea6 = ms_res_b(fea4, fea5, fea6, 128, '_9', dilation_rate=2)
    #fea4, fea5, fea6 = ms_res_b(fea4, fea5, fea6, 128, '_10', dilation_rate=4)
    #fea4, fea5, fea6 = ms_res_b(fea4, fea5, fea6, 128, '_11')
    #fea4, fea5, fea6 = ms_res_b(fea4, fea5, fea6, 128, '_12', dilation_rate=2)
    #fea4, fea5, fea6 = ms_res_b(fea4, fea5, fea6, 128, '_13', dilation_rate=4)
    #fea4, _, _ = ms_res_b(fea4, fea5, fea6, 128, '_14')
    
    #fea4 = aspp(fea4, 64, 'aspp', reg)
    #fea4_u = layers.UpSampling2D(8)(fea4)
    #fea = layers.Concatenate()([fea1, fea4_u])
    fea = layers.BatchNormalization(name='bn_f')(fea1)
    fea = layers.Activation('relu')(fea)
    
    predict = layers.Conv2D(1, 1, kernel_initializer='he_normal', activation='sigmoid',
                            padding='same', name='predict')(fea)
    
    mmodel = Model(inputs=inputs, outputs=predict)
    mmodel.compile(optimizer=optimizers.Adam(1e-3), loss='binary_crossentropy',
                   metrics=[dice])
    if pretrianed:
        mmodel.load_weights(pretrianed)

    return mmodel

