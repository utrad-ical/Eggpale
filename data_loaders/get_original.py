import numpy as np
import os
from skimage import io, color
from io import BytesIO
import json

def load_data(rootpath, resolution, image_color_num=3, verbose=False):
    
    jsonpath = rootpath
    with open(jsonpath, "rt") as f:
        json_dict = json.load(f)
        
        # training datasets
        print ( "reading training datasets..." )
        pngrootpath = json_dict["training"]["image"].strip()
        train_x = []
        loaded_img_cnt = 0
        for trainfilename in open(json_dict["training"]["list"], 'rt'):
            trainfilename = trainfilename.strip()
            if verbose or (loaded_img_cnt % 1000 == 0):
                print ( "loading %s ... " % trainfilename )
            img = io.imread(os.path.join(pngrootpath, trainfilename))
            if len(img.shape)==2:
                img = img[:,:,np.newaxis] # enforce ndims=3
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2) # enforce RGB
            if image_color_num==1:
                img = img[:,:,0:1] # enforce monochrome
            if len(img.shape)==2:
                img = img[:,:,np.newaxis] # enforce ndims=3
            train_x.append(img)
            loaded_img_cnt += 1
        train_x = np.array(train_x)
        
        # test datasets
        print ( "reading test datasets..." )
        pngrootpath = json_dict["test"]["image"].strip()
        test_x = []
        loaded_img_cnt = 0
        for testfilename in open(json_dict["test"]["list"], 'rt'):
            testfilename = testfilename.strip()
            if verbose or (loaded_img_cnt % 1000 == 0):
                print ( "loading %s ... " % testfilename )
            img = io.imread(os.path.join(pngrootpath, testfilename))
            if len(img.shape)==2:
                img = img[:,:,np.newaxis] # enforce ndims=3
            if img.shape[2] == 1:
                img = np.repeat(img, 3, axis=2) # enforce RGB
            if image_color_num==1:
                img = img[:,:,0:1] # enforce monochrome
            if len(img.shape)==2:
                img = img[:,:,np.newaxis] # enforce ndims=3
            test_x.append(img)
            loaded_img_cnt += 1
        test_x = np.array(test_x)
    
    train_y = np.zeros(train_x.shape[0])
    test_y = np.zeros(test_x.shape[0])
    
    print ("train_x.shape = ", train_x.shape)
    print ("test_x.shape = ", test_x.shape)
    
    return ((train_x, train_y), (test_x, test_y))


def downsample(x, resolution):
    assert x.dtype == np.float32
    assert x.shape[1] % resolution == 0
    assert x.shape[2] % resolution == 0
    if x.shape[1] == x.shape[2] == resolution:
        return x
    s = x.shape
    x = np.reshape(x, [s[0], resolution, s[1] // resolution,
                       resolution, s[2] // resolution, s[3]])
    x = np.mean(x, (2, 4))
    return x


def x_to_uint8(x):
    x = np.clip(np.floor(x), 0, 255)
    return x.astype(np.uint8)


def shard(data, shards, rank):
    # Determinisitc shards
    x, y = data
    assert x.shape[0] == y.shape[0]
    assert x.shape[0] % shards == 0
    assert 0 <= rank < shards
    size = x.shape[0] // shards
    ind = rank*size
    return x[ind:ind+size], y[ind:ind+size]


def get_data(rootpath, shards, rank, n_batch_train, n_batch_test, n_batch_init, resolution,
             image_color_num, verbose):
    (x_train, y_train), (x_test, y_test) = load_data(rootpath, resolution, image_color_num, verbose)
    y_train = np.reshape(y_train, [-1])
    y_test = np.reshape(y_test, [-1])

    print('n_train:', x_train.shape[0], 'n_test:', x_test.shape[0])

    n_test = x_test.shape[0]

    # Shard before any shuffling
    x_train, y_train = shard((x_train, y_train), shards, rank)
    x_test, y_test = shard((x_test, y_test), shards, rank)

    print('n_shard_train:', x_train.shape[0], 'n_shard_test:', x_test.shape[0])

    from keras.preprocessing.image import ImageDataGenerator
    datagen_test = ImageDataGenerator()
    datagen_train = ImageDataGenerator()

    datagen_train.fit(x_train)
    datagen_test.fit(x_test)
    train_flow = datagen_train.flow(x_train, y_train, n_batch_train)
    test_flow = datagen_test.flow(x_test, y_test, n_batch_test, shuffle=False)

    def make_iterator(flow, resolution):
        def iterator():
            x_full, y = flow.next()
            x_full = x_full.astype(np.float32)
            x = downsample(x_full, resolution)
            x = x_to_uint8(x)
            return x, y

        return iterator

    #init_iterator = make_iterator(train_flow, resolution)
    train_iterator = make_iterator(train_flow, resolution)
    test_iterator = make_iterator(test_flow, resolution)

    # Get data for initialization
    data_init = make_batch(train_iterator, n_batch_train, n_batch_init)

    return train_iterator, test_iterator, data_init, n_test


def make_batch(iterator, iterator_batch_size, required_batch_size):
    ib, rb = iterator_batch_size, required_batch_size
    #assert rb % ib == 0
    k = int(np.ceil(rb / ib))
    xs, ys = [], []
    for i in range(k):
        x, y = iterator()
        xs.append(x)
        ys.append(y)
    x, y = np.concatenate(xs)[:rb], np.concatenate(ys)[:rb]
    return {'x': x, 'y': y}
