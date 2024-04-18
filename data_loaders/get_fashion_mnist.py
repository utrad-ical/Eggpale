import numpy as np
from PIL import Image
import sys

def png_generator(data, label, path_dir):
    import os
    
    img = Image.new("L", (28,28))
    pix = img.load()
    
    for i in range (data.shape[0]):
        label_name = str(label[i])
        file_dir = path_dir+label_name

        if os.path.isdir(file_dir) is False:
            os.makedirs(file_dir)

        filename = str(i)
        
        for m in range(28):
            for n in range(28):
                pix[m,n] = int(data[i][m+n*28])
        #img2 = img.resize((256, 256))
        img.save(file_dir+'/'+filename+'.png')

##################################################################


def load_png(path, kind= 'train'):
    import os
    import glob
   
    data = []
    label = []
    path_dir=path+kind
    for i in os.listdir(path_dir):
        folder_dir = path_dir+'/'+str(i)
        # for j in os.listdir(folder_dir):
        #     file_index = os.path.splitext(j)
        #     file_name = folder_dir+'/'+j
            # print("index:", int(file_index[0]))
            # data.append((60000,256,256))
            # label.append((60000))
        files = glob.glob(folder_dir+'/*.png')
       
        for f in files:
            imgs = np.array(Image.open(f))
            data.append(imgs)
            label.append(i)

    data = np.array(data)
    label = np.array(label)
    print(data.shape, label.shape)
    return data, label
    #  data = np.array(Image.open(file_name))
    #         label[int(file_index[0])] = int(i)

    # print("TTTTTTTTTT",data.shape)
    # print("AAAAAAAAAAAA",label.shape)
            
def load_fashion_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load FASHION_MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)
    return images, labels

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


def get_data(shards, rank, data_augmentation_level, n_batch_train, n_batch_test, n_batch_init, resolution):
    #from keras.datasets import fashion_mnist
    #(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # x_train_temp, y_train_temp = load_fashion_mnist('data/fashion', kind='train')
    # x_test, y_test = load_fashion_mnist('data/fashion', kind='t10k')

    # train_dir = ('fashion_data/train/')
    # test_dir = ('fashion_data/test/')
    
    # png_generator(x_train_temp, y_train_temp, train_dir);
    # png_generator(x_test, y_test, test_dir);
    
    
    
    # x_train = x_train_temp#[y_train_temp != 5]
    # y_train = y_train_temp#[y_train_temp != 5]
    # x_train = x_train.reshape(x_train.shape[0], 28, 28)
    # x_test = x_test.reshape(x_test.shape[0], 28, 28)
    # print(x_train.shape)

    x_train, y_train = load_png('fashion_data1/', kind = 'train')
    x_test, y_test = load_png('fashion_data1/', kind = 'test')
    x_train = x_train.reshape(60000, 256, 256)
    x_test = x_test.reshape(10000, 256, 256)
    #sys.exit()
    print("after+++", x_train.shape)    
    
    y_train = np.reshape(y_train, [-1])
    y_test = np.reshape(y_test, [-1])
    # Pad with zeros to make 32x32
    #x_train = np.lib.pad(x_train, ((0, 0), (114, 114), (114, 114)), 'minimum')

    # Pad with zeros to make 32x32
    #x_test = np.lib.pad(x_test, ((0, 0), (114, 114), (114, 114)), 'minimum')
    x_train = np.tile(np.reshape(x_train, (-1, 256, 256, 1)), (1, 1, 1, 3))
    x_test = np.tile(np.reshape(x_test, (-1, 256, 256, 1)), (1, 1, 1, 3))

    
    print('n_train:', x_train.shape[0], 'n_train_y:', y_train.shape[0],'n_test:', x_test.shape[0])

    
    # Shard before any shuffling
    x_train, y_train = shard((x_train, y_train), shards, rank)
    x_test, y_test = shard((x_test, y_test), shards, rank)

    print('n_shard_train:', x_train.shape[0], 'n_shard_test:', x_test.shape[0])

    from keras.preprocessing.image import ImageDataGenerator
    datagen_test = ImageDataGenerator()
    if data_augmentation_level == 0:
        datagen_train = ImageDataGenerator()
    else:
        datagen_train = ImageDataGenerator(
            width_shift_range=0.1,
            height_shift_range=0.1
        )
        
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

    return train_iterator, test_iterator, data_init


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
