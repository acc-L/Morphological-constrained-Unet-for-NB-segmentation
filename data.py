from scipy.io import loadmat
from skimage import io, transform, morphology
import os
import numpy as np

def prescale(x):
    x = x/1000
    x[x>1]=1
    x[x<-1]=-1
    return x

def read_img(file):
    image = loadmat(file)
    for i in range(90,100):
        pimage = image['value'][:,:,i]
        pimage = (pimage+1)/2
        io.imshow(pimage)
        io.show()

def generate_from_npy(path, data, label, batch_size=1, shuffle=True, round=True, depth = 16):
    datapath = os.path.join(path,data)
    labelpath = os.path.join(path, label)
    data_dir_s = os.listdir(datapath)
    label_dir_s = os.listdir(labelpath)
    data_dir = []
    label_dir = []
    for name in data_dir_s:
        if name.endswith('.npy'):
            data_dir.append(name)
    for name in label_dir_s:
        if name.endswith('.npy'):
            label_dir.append(name)

    data_dir.sort()
    label_dir.sort()
    sample = list(zip(data_dir,label_dir))
    slenth = len(data_dir)
    index = 0

    if(shuffle):
        np.random.shuffle(sample)

    while(True):
        if index + batch_size > slenth:
            if round:
                files = sample[index:slenth] + sample[:index + batch_size -slenth]
                index = index + batch_size - slenth
            else:
                break
        else:
            files = sample[index:index+batch_size]
            index += batch_size

        data_lst = []
        label_lst = []
        for i in range(batch_size):
            idnum = files[i][0]
            assert idnum == files[i][1]

            data_arr = np.load(os.path.join(datapath, idnum))
            label_arr = np.load(os.path.join(labelpath, idnum))
            slice_num = np.random.randint(low=0,high=data_arr.shape[2]-depth+1)
            data_arr = data_arr[:, :, slice_num:slice_num+depth]
            label_arr = label_arr[:, :, slice_num:slice_num+depth]

            data_arr = transform.resize(data_arr, output_shape=(128,128), order=0, mode='edge')
            label_arr = transform.resize(label_arr, output_shape=(128,128), order=0, mode='edge')

            data_lst.append(data_arr[np.newaxis,:,:,:,np.newaxis])
            label_lst.append(label_arr[np.newaxis,:,:,:,np.newaxis])
        data_lst = np.concatenate(data_lst)
        label_lst = np.concatenate(label_lst)
        yield data_lst, label_lst

def generate3d_from_npy(path, data, label, batch_size=1, shuffle=True, round=True,
                        out_size=(64,64,64), test=False, rotate = 0):
    datapath = os.path.join(path,data)
    labelpath = os.path.join(path, label)
    data_dir_s = os.listdir(datapath)
    label_dir_s = os.listdir(labelpath)
    data_dir = []
    label_dir = []
    for name in data_dir_s:
        if name.endswith('.npy'):
            data_dir.append(name)
    for name in label_dir_s:
        if name.endswith('.npy'):
            label_dir.append(name)

    data_dir.sort()
    label_dir.sort()
    sample = list(zip(data_dir,label_dir))
    slenth = len(data_dir)
    index = 0

    if(shuffle):
        np.random.shuffle(sample)

    while(True):
        if index + batch_size > slenth:
            if round:
                files = sample[index:slenth] + sample[:index + batch_size -slenth]
                index = index + batch_size - slenth
            else:
                break
        else:
            files = sample[index:index+batch_size]
            index += batch_size

        data_lst = []
        label_lst = []
        for i in range(batch_size):
            idnum = files[i][0]
            assert idnum == files[i][1]

            data_arr = np.load(os.path.join(datapath, idnum))
            label_arr = np.load(os.path.join(labelpath, idnum))

            if rotate>0:
                angel = np.random.random() * rotate
                data_arr = transform.rotate(data_arr, angel, mode='edge')
                label_arr = transform.rotate(label_arr, angel, mode='edge')

            data_arr = transform.resize(data_arr, output_shape=out_size, order=0, mode='edge')
            label_arr = transform.resize(label_arr, output_shape=out_size, order=0, mode='edge')

            data_lst.append(data_arr[np.newaxis,:,:,:,np.newaxis])
            label_lst.append(label_arr[np.newaxis,:,:,:,np.newaxis])
        data_lst = np.concatenate(data_lst)
        label_lst = np.concatenate(label_lst)
        if test:
            yield data_lst
        else:
            yield data_lst, label_lst

def generate2d_from_npy(path, data, label, batch_size=1, shuffle=True, round=True,
                        out_size=(256,256), test=False, rotate = 0, scale=0, shift=0):
    datapath = os.path.join(path,data)
    labelpath = os.path.join(path, label)
    data_dir_s = os.listdir(datapath)
    label_dir_s = os.listdir(labelpath)
    data_dir = []
    label_dir = []
    for name in data_dir_s:
        if name.endswith('.npy'):
            data_dir.append(name)
    for name in label_dir_s:
        if name.endswith('.npy'):
            label_dir.append(name)

    data_dir.sort()
    label_dir.sort()
    sample = list(zip(data_dir,label_dir))
    slenth = len(data_dir)
    index = 0

    if(shuffle):
        np.random.shuffle(sample)

    while(True):
        if index + batch_size > slenth:
            if round:
                files = sample[index:slenth] + sample[:index + batch_size -slenth]
                index = index + batch_size - slenth
            else:
                break
        else:
            files = sample[index:index+batch_size]
            index += batch_size

        data_lst = []
        label_lst = []
        for i in range(batch_size):
            idnum = files[i][0]
            assert idnum == files[i][1]

            data_arr = np.load(os.path.join(datapath, idnum))
            #data_arr = data_arr/3000
            data_arr = prescale(data_arr)
            label_arr = np.load(os.path.join(labelpath, idnum))

            if scale>0:
                rscale = np.random.rand()*scale*2 + 1 - scale
                d_shape = data_arr.shape
                shape = [int(i*rscale) for i in d_shape]
                if rscale > 1:
                    new_arr = np.zeros(shape)
                    nlab_arr = np.zeros(shape)
                    diffx = int((shape[0] - d_shape[0])//2)
                    diffy = int((shape[1] - d_shape[1])//2)
                    new_arr[diffx:diffx+d_shape[0], diffy:diffy+d_shape[1]] = data_arr
                    nlab_arr[diffx:diffx + d_shape[0], diffy:diffy + d_shape[1]] = label_arr
                    data_arr = new_arr
                    label_arr = nlab_arr
                else:
                    diffx = int((d_shape[0] - shape[0]) // 2)
                    diffy = int((d_shape[1] - shape[1]) // 2)
                    data_arr = data_arr[diffx:diffx+d_shape[0], diffy:diffy+d_shape[1]]
                    label_arr = label_arr[diffx:diffx + d_shape[0], diffy:diffy + d_shape[1]]

            if shift>0:
                srange = np.random.rand(2)*2*shift - shift
                new_arr = np.zeros_like(data_arr)
                nlab_arr = np.zeros_like(label_arr)
                shape = data_arr.shape
                if srange[0] > 0:
                    n_left = int(srange[0]*shape[0])
                    d_left = 0
                    n_right = shape[0]
                    d_right = shape[0] - n_left
                else:
                    d_left = int(-srange[0] * shape[0])
                    n_left = 0
                    d_right = shape[0]
                    n_right = shape[0] - d_left

                if srange[1] > 0:
                    n_up = int(srange[1]*shape[1])
                    d_up = 0
                    n_down = shape[1]
                    d_down = shape[1] - n_up
                else:
                    d_up = int(-srange[1] * shape[1])
                    n_up = 0
                    d_down = shape[1]
                    n_down = shape[1] - d_up

                new_arr[n_left:n_right, n_up:n_down] = data_arr[d_left:d_right, d_up:d_down]
                nlab_arr[n_left:n_right, n_up:n_down] = label_arr[d_left:d_right, d_up:d_down]
                data_arr = new_arr
                label_arr = nlab_arr

            if rotate>0:
                angel = np.random.rand() * rotate
                data_arr = transform.rotate(data_arr, angel, mode='edge')
                label_arr = transform.rotate(label_arr, angel, mode='edge')

            data_arr = transform.resize(data_arr, output_shape=out_size, order=0, mode='edge')
            label_arr = transform.resize(label_arr, output_shape=out_size, order=0, mode='edge')

            data_lst.append(data_arr[np.newaxis,:,:,np.newaxis])
            label_lst.append(label_arr[np.newaxis,:,:,np.newaxis])
        data_lst = np.concatenate(data_lst)
        label_lst = np.concatenate(label_lst)
        if test:
            yield data_lst
        else:
            yield data_lst, label_lst
            

def generate_edge(path, data, label, batch_size=1, shuffle=True, round=True,
                        out_size=(256,256), test=False, rotate = 0, scale=0, shift=0):
    datapath = os.path.join(path,data)
    labelpath = os.path.join(path, label)
    data_dir_s = os.listdir(datapath)
    label_dir_s = os.listdir(labelpath)
    data_dir = []
    label_dir = []
    for name in data_dir_s:
        if name.endswith('.npy'):
            data_dir.append(name)
    for name in label_dir_s:
        if name.endswith('.npy'):
            label_dir.append(name)

    data_dir.sort()
    label_dir.sort()
    sample = list(zip(data_dir,label_dir))
    slenth = len(data_dir)
    index = 0

    if(shuffle):
        np.random.shuffle(sample)

    while(True):
        if index + batch_size > slenth:
            if round:
                files = sample[index:slenth] + sample[:index + batch_size -slenth]
                index = index + batch_size - slenth
            else:
                break
        else:
            files = sample[index:index+batch_size]
            index += batch_size

        data_lst = []
        label_lst = []
        area_lst = []
        cgt_lst = []
        for i in range(batch_size):
            idnum = files[i][0]
            assert idnum == files[i][1]

            data_arr = np.load(os.path.join(datapath, idnum))
            #data_arr = data_arr/3000
            data_arr = prescale(data_arr)
            label_arr = np.load(os.path.join(labelpath, idnum))

            if scale>0:
                rscale = np.random.rand()*scale*2 + 1 - scale
                d_shape = data_arr.shape
                shape = [int(i*rscale) for i in d_shape]
                if rscale > 1:
                    new_arr = np.zeros(shape)
                    nlab_arr = np.zeros(shape)
                    diffx = int((shape[0] - d_shape[0])//2)
                    diffy = int((shape[1] - d_shape[1])//2)
                    new_arr[diffx:diffx+d_shape[0], diffy:diffy+d_shape[1]] = data_arr
                    nlab_arr[diffx:diffx + d_shape[0], diffy:diffy + d_shape[1]] = label_arr
                    data_arr = new_arr
                    label_arr = nlab_arr
                else:
                    diffx = int((d_shape[0] - shape[0]) // 2)
                    diffy = int((d_shape[1] - shape[1]) // 2)
                    data_arr = data_arr[diffx:diffx+d_shape[0], diffy:diffy+d_shape[1]]
                    label_arr = label_arr[diffx:diffx + d_shape[0], diffy:diffy + d_shape[1]]

            if shift>0:
                srange = np.random.rand(2)*2*shift - shift
                new_arr = np.zeros_like(data_arr)
                nlab_arr = np.zeros_like(label_arr)
                shape = data_arr.shape
                if srange[0] > 0:
                    n_left = int(srange[0]*shape[0])
                    d_left = 0
                    n_right = shape[0]
                    d_right = shape[0] - n_left
                else:
                    d_left = int(-srange[0] * shape[0])
                    n_left = 0
                    d_right = shape[0]
                    n_right = shape[0] - d_left

                if srange[1] > 0:
                    n_up = int(srange[1]*shape[1])
                    d_up = 0
                    n_down = shape[1]
                    d_down = shape[1] - n_up
                else:
                    d_up = int(-srange[1] * shape[1])
                    n_up = 0
                    d_down = shape[1]
                    n_down = shape[1] - d_up

                new_arr[n_left:n_right, n_up:n_down] = data_arr[d_left:d_right, d_up:d_down]
                nlab_arr[n_left:n_right, n_up:n_down] = label_arr[d_left:d_right, d_up:d_down]
                data_arr = new_arr
                label_arr = nlab_arr

            if rotate>0:
                angel = np.random.rand() * rotate
                data_arr = transform.rotate(data_arr, angel, mode='edge')
                label_arr = transform.rotate(label_arr, angel, mode='edge')

            data_arr = transform.resize(data_arr, output_shape=out_size, order=0, mode='edge')
            label_arr = transform.resize(label_arr, output_shape=out_size, order=0, mode='edge')
            area = -np.log(label_arr.mean()+1e-6)
            dal = morphology.dilation(label_arr, selem=morphology.square(3))
            cgt = -np.log((dal.mean()-label_arr.mean()+1e-6)/(dal.mean()+1e-6))

            data_lst.append(data_arr[np.newaxis,:,:,np.newaxis])
            label_lst.append(label_arr[np.newaxis,:,:,np.newaxis])
            area_lst.append(area)
            cgt_lst.append(cgt)
        data_lst = np.concatenate(data_lst)
        label_lst = np.concatenate(label_lst)
        area_lst = np.array(area_lst)
        cgt_lst = np.array(cgt_lst)
        if test:
            yield data_lst
        else:
            yield data_lst, [label_lst, area_lst, cgt_lst, area_lst, cgt_lst]
            
            
def generate_brats(path, data, label, batch_size=1, shuffle=True, round=True,
                        out_size=(256,256), test=False, rotate = 0, scale=0, shift=0):
    datapath = os.path.join(path,data)
    labelpath = os.path.join(path, label)
    data_dir_s = os.listdir(datapath)
    label_dir_s = os.listdir(labelpath)
    data_dir = []
    label_dir = []
    for name in data_dir_s:
        if name.endswith('.npy'):
            data_dir.append(name)
    for name in label_dir_s:
        if name.endswith('.npy'):
            label_dir.append(name)

    data_dir.sort()
    label_dir.sort()
    sample = list(zip(data_dir,label_dir))
    slenth = len(data_dir)
    index = 0

    if(shuffle):
        np.random.shuffle(sample)

    while(True):
        if index + batch_size > slenth:
            if round:
                files = sample[index:slenth] + sample[:index + batch_size -slenth]
                index = index + batch_size - slenth
            else:
                break
        else:
            files = sample[index:index+batch_size]
            index += batch_size

        data_lst = []
        label_lst = []
        area_lst = []
        cgt_lst = []
        for i in range(batch_size):
            idnum = files[i][0]
            assert idnum == files[i][1]

            data_arr = np.load(os.path.join(datapath, idnum))
            #data_arr = data_arr/3000
            data_arr = prescale(data_arr)
            label_arr = np.load(os.path.join(labelpath, idnum))

            if scale>0:
                rscale = np.random.rand()*scale*2 + 1 - scale
                d_shape = data_arr.shape
                shape = [int(i*rscale) for i in d_shape]
                if rscale > 1:
                    new_arr = np.zeros(shape)
                    nlab_arr = np.zeros(shape)
                    diffx = int((shape[0] - d_shape[0])//2)
                    diffy = int((shape[1] - d_shape[1])//2)
                    new_arr[diffx:diffx+d_shape[0], diffy:diffy+d_shape[1]] = data_arr
                    nlab_arr[diffx:diffx + d_shape[0], diffy:diffy + d_shape[1]] = label_arr
                    data_arr = new_arr
                    label_arr = nlab_arr
                else:
                    diffx = int((d_shape[0] - shape[0]) // 2)
                    diffy = int((d_shape[1] - shape[1]) // 2)
                    data_arr = data_arr[diffx:diffx+d_shape[0], diffy:diffy+d_shape[1]]
                    label_arr = label_arr[diffx:diffx + d_shape[0], diffy:diffy + d_shape[1]]

            if shift>0:
                srange = np.random.rand(2)*2*shift - shift
                new_arr = np.zeros_like(data_arr)
                nlab_arr = np.zeros_like(label_arr)
                shape = data_arr.shape
                if srange[0] > 0:
                    n_left = int(srange[0]*shape[0])
                    d_left = 0
                    n_right = shape[0]
                    d_right = shape[0] - n_left
                else:
                    d_left = int(-srange[0] * shape[0])
                    n_left = 0
                    d_right = shape[0]
                    n_right = shape[0] - d_left

                if srange[1] > 0:
                    n_up = int(srange[1]*shape[1])
                    d_up = 0
                    n_down = shape[1]
                    d_down = shape[1] - n_up
                else:
                    d_up = int(-srange[1] * shape[1])
                    n_up = 0
                    d_down = shape[1]
                    n_down = shape[1] - d_up

                new_arr[n_left:n_right, n_up:n_down] = data_arr[d_left:d_right, d_up:d_down]
                nlab_arr[n_left:n_right, n_up:n_down] = label_arr[d_left:d_right, d_up:d_down]
                data_arr = new_arr
                label_arr = nlab_arr

            if rotate>0:
                angel = np.random.rand() * rotate
                data_arr = transform.rotate(data_arr, angel, mode='edge')
                label_arr = transform.rotate(label_arr, angel, mode='edge')

            data_arr = transform.resize(data_arr, output_shape=out_size, order=0, mode='edge')
            label_arr = transform.resize(label_arr, output_shape=out_size, order=0, mode='edge')
            
            #one-hot
            maps = [label_arr==oh_ind +0 for oh_ind in [0,1,2,4]]
            mapss = [m[:,:,np.newaxis] for m in maps]
            label_arr = np.concatenate(mapss, axis=2)
            
            area = -np.log(label_arr.mean(axis=(0,1))+1e-6)
            
            dal = [morphology.dilation(m, selem=morphology.square(3))[:,:,np.newaxis] for m in maps]
            dal = np.concatenate(dal, axis=2)
            cgt = -np.log((dal.mean(axis=(0,1))-label_arr.mean(axis=(0,1))+1e-6)/(dal.mean(axis=(0,1))+1e-6))

            data_lst.append(data_arr[np.newaxis,:,:,np.newaxis])
            label_lst.append(label_arr[np.newaxis,:,:,:])
            area_lst.append(area)
            cgt_lst.append(cgt)
        data_lst = np.concatenate(data_lst)
        label_lst = np.concatenate(label_lst)
        area_lst = np.array(area_lst).T
        cgt_lst = np.array(cgt_lst).T
        if test:
            yield data_lst
        else:
            #yield data_lst, [label_lst, area_lst, cgt_lst, area_lst, cgt_lst]
            yield data_lst, label_lst


def generate2d_withrp_from_npy(path, data, label, batch_size=1, shuffle=True, round=True,
                        out_size=(256,256), test=False, rotate = 0, scale=0, shift=0, sample=1024):
    datapath = os.path.join(path,data)
    labelpath = os.path.join(path, label)
    data_dir_s = os.listdir(datapath)
    label_dir_s = os.listdir(labelpath)
    data_dir = []
    label_dir = []
    for name in data_dir_s:
        if name.endswith('.npy'):
            data_dir.append(name)
    for name in label_dir_s:
        if name.endswith('.npy'):
            label_dir.append(name)

    data_dir.sort()
    label_dir.sort()
    sample = list(zip(data_dir,label_dir))
    slenth = len(data_dir)
    index = 0

    if(shuffle):
        np.random.shuffle(sample)

    while(True):
        if index + batch_size > slenth:
            if round:
                files = sample[index:slenth] + sample[:index + batch_size -slenth]
                index = index + batch_size - slenth
            else:
                break
        else:
            files = sample[index:index+batch_size]
            index += batch_size

        data_lst = []
        label_lst = []

        for i in range(batch_size):
            idnum = files[i][0]
            assert idnum == files[i][1]

            data_arr = np.load(os.path.join(datapath, idnum))
            label_arr = np.load(os.path.join(labelpath, idnum))

            if scale>0:
                rscale = np.random.rand()*scale*2 + 1 - scale
                d_shape = data_arr.shape
                shape = [int(i*rscale) for i in d_shape]
                if rscale > 1:
                    new_arr = np.zeros(shape)
                    nlab_arr = np.zeros(shape)
                    diffx = int((shape[0] - d_shape[0])//2)
                    diffy = int((shape[1] - d_shape[1])//2)
                    new_arr[diffx:diffx+d_shape[0], diffy:diffy+d_shape[1]] = data_arr
                    nlab_arr[diffx:diffx + d_shape[0], diffy:diffy + d_shape[1]] = label_arr
                    data_arr = new_arr
                    label_arr = nlab_arr
                else:
                    diffx = int((d_shape[0] - shape[0]) // 2)
                    diffy = int((d_shape[1] - shape[1]) // 2)
                    data_arr = data_arr[diffx:diffx+d_shape[0], diffy:diffy+d_shape[1]]
                    label_arr = label_arr[diffx:diffx + d_shape[0], diffy:diffy + d_shape[1]]

            if shift>0:
                srange = np.random.rand(2)*2*shift - shift
                new_arr = np.zeros_like(data_arr)
                nlab_arr = np.zeros_like(label_arr)
                shape = data_arr.shape
                if srange[0] > 0:
                    n_left = int(srange[0]*shape[0])
                    d_left = 0
                    n_right = shape[0]
                    d_right = shape[0] - n_left
                else:
                    d_left = int(-srange[0] * shape[0])
                    n_left = 0
                    d_right = shape[0]
                    n_right = shape[0] - d_left

                if srange[1] > 0:
                    n_up = int(srange[1]*shape[1])
                    d_up = 0
                    n_down = shape[1]
                    d_down = shape[1] - n_up
                else:
                    d_up = int(-srange[1] * shape[1])
                    n_up = 0
                    d_down = shape[1]
                    n_down = shape[1] - d_up

                new_arr[n_left:n_right, n_up:n_down] = data_arr[d_left:d_right, d_up:d_down]
                nlab_arr[n_left:n_right, n_up:n_down] = label_arr[d_left:d_right, d_up:d_down]
                data_arr = new_arr
                label_arr = nlab_arr

            if rotate>0:
                angel = np.random.rand() * rotate
                data_arr = transform.rotate(data_arr, angel, mode='edge')
                label_arr = transform.rotate(label_arr, angel, mode='edge')

            data_arr = transform.resize(data_arr, output_shape=out_size, order=0, mode='edge')
            label_arr = transform.resize(label_arr, output_shape=out_size, order=0, mode='edge')

            if label_arr.max() > 0:
                ones = np.where(label_arr>0)
                ones = np.array(ones)
                center = ones.mean(axis=1)
                std = ones.std(axis=1)*2+10
            else:
                center = (128,128)
                std = (40, 40)
            points = np.random.normal(loc=center, scale=std, size=(1024,2))
            points = points.astype(np.int32)[np.newaxis,:]
            points[points<0] = np.random.randint(out_size[0])
            points[points>=out_size[0]] = np.random.randint(out_size[0])
            mask = np.zeros_like(label_arr, dtype=np.int32)

            for point in points:
                mask[point] -= 1

            mask[label_arr>0] *= (-1)

            data_lst.append(data_arr[np.newaxis,:,:,np.newaxis])
            label_lst.append(mask[np.newaxis,:,:,np.newaxis])

        data_lst = np.concatenate(data_lst)
        label_lst = np.concatenate(label_lst)

        if test:
            yield data_lst
        else:
            yield data_lst, label_lst

def saveResult(save_path,test_path,npyfile):
    image_list=os.listdir(test_path)
    for i,item in enumerate(npyfile):
        img = item[:,:,0]
        np.save(os.path.join(save_path,image_list[i][:-4]+"_predict"),img)

#read_img('data/TrainData/40427664.mat')
