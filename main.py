from model import ms_res, denseud_byb_v2, unet_2d_b, unet_2d_a
from model import dice, dice_loss
from data import generate2d_from_npy, saveResult, generate_edge, generate_brats
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope
#from clayer import SelectiveMerge_v2
import pickle

#train_gen = generate_edge('../sample','TrainData', 'TrainLabel', batch_size=4)
#                                rotate=30, scale=0.2, shift=0)
val_gen = generate_edge('../sample', 'ValData', 'ValLabel', shuffle=False)
tval_gen = generate2d_from_npy('../sample', 'TrainData', 'TrainLabel', shuffle=False)
test_gen = generate2d_from_npy('../sample', 'TestData', 'TestLabel', shuffle=False)
model_checkpoint = ModelCheckpoint('models/base_bn2.hdf5',verbose=1, save_best_only=True)

#train_gen = generate_edge('../crop','traindata', 'trainlabel', batch_size=4,
#                          rotate=30, scale=0.2, shift=0)
train_gen = generate_brats('/media/public/新加卷1/brats','t2', 'seg', batch_size=4)
#                          rotate=30, scale=0.2, shift=0)

base_lr = 1e-2
def lr_sch(epoch):
    if epoch<20:
        return 1e-3
    elif epoch<30:
        return 1e-4
    else:
        return 1e-5


es = EarlyStopping(monitor='val_loss', patience=10)
#reduce_lr = ReduceLROnPlateau(factor=0.5, patience=5)
reduce_lr = LearningRateScheduler(lr_sch)

#baseline
model_checkpoint = ModelCheckpoint('models/brats_n.h5',verbose=1, save_weights_only=True)
#mymodel = unet_2d_bn()
mymodel = unet_2d_a()
#history = mymodel.fit_generator(train_gen,steps_per_epoch=2970,epochs=40,callbacks=[model_checkpoint, reduce_lr],
#                      validation_data=val_gen, validation_steps=1064, verbose=1)
history = mymodel.fit_generator(train_gen,steps_per_epoch=3520,epochs=40,callbacks=[model_checkpoint, reduce_lr],
                                verbose=1)

with open('records/brats_n', 'wb') as dicfile:
    pickle.dump(history.history, dicfile)

'''

with CustomObjectScope({'dice': dice, 'dice_loss':dice_loss}):
    mymodel = load_model('models/dense_ud_mo.hdf5')
    #history = mymodel.fit_generator(train_gen, steps_per_epoch=2970, epochs=35, callbacks=[model_checkpoint],
    #                                validation_data=val_gen, validation_steps=1064, verbose=2)

    #with open('records/base_bn.dic', 'wb') as dicfile:
    #    pickle.dump(history.history, dicfile)
    loss = mymodel.evaluate_generator(test_gen, steps=3180)
    print(loss)
    #results = mymodel.predict_generator(tval_gen,438,verbose=1)
    #saveResult("cropdata/ValPredict", 'cropdata/ValData',results)
    #print(mymodel.layers[1].get_weights())
'''
