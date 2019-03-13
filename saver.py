# -*- coding: utf-8 -*-

import os
import numpy as np
from skimage import io, transform, morphology
#from tensorflow.keras import Model, layers
from model import dice, dice_loss, unet_2d_b, denseud_byb_v2
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import CustomObjectScope

#TestDir = '../sample/TestData'
#LabelDir = '../sample/TestLabel'
TestDir = '../crop/testdata'
LabelDir = '../crop/testlabel'

samples = os.listdir(TestDir)
pre_list = ['ac2', 'p4', 'p3', 'p2']

with CustomObjectScope({'dice': dice, 'dice_loss':dice_loss}):
    #mymodel = load_model('models/denseud_sum_v3.hdf5')
    mymodel = unet_2d_b(pretrained_weights='models/unet_spac_crop.h5')
    
    #n5 = mymodel.get_layer('')
    
    for p in pre_list:
        if not os.path.exists(p):
            os.mkdir(p)
    if not os.path.exists('gt'):
        os.mkdir('gt')
    if not os.path.exists('edge'):
        os.mkdir('edge')
    
    for sample in samples:
        print(sample)
        arr = np.load(os.path.join(TestDir, sample))
        lab = np.load(os.path.join(LabelDir, sample))
        outname = sample.split('.')[0]+'.png'
        arr = arr/1000
        arr[arr>1]=1
        arr[arr<-1]=-1
        
        arr = transform.resize(arr, [256,256], order=0)
        lab = transform.resize(lab, [256,256], order=0)
        #io.imsave(os.path.join('img', outname),(arr+1)/2)
        '''
        lab_d = morphology.dilation(lab, selem=morphology.square(7))
        lab_e = morphology.erosion(lab, selem=morphology.square(7))
        edge = lab_d - lab_e
        
        io.imsave(os.path.join('edge', outname),edge)
        '''
        arr = arr[np.newaxis,:,:,np.newaxis]
        io.imsave(os.path.join('crop_gt', outname),lab)
        res = mymodel.predict_on_batch(arr)
        #for i in range(3):
        #    io.imsave(os.path.join(pre_list[i], outname), res[i][0, :, :, 0])
        io.imsave(os.path.join('crop', outname), res[0][0, :, :, 0])
    #results = mymodel.predict_generator(tval_gen,438,verbose=1)
    #saveResult("cropdata/ValPredict", 'cropdata/ValData',results)
    #print(mymodel.layers[1].get_weights())