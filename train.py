#!/usr/bin/env python

# Modified Horovod MNIST example

import os
import sys
import time
import json

import numpy as np
import horovod.tensorflow as hvd
import tensorflow as tf
import graphics
from utils import ResultLogger
import imageio
import PIL



learn = tf.contrib.learn

# Surpress verbose warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def _print(*args, **kwargs):
    if hvd.rank() == 0:
        print(*args, **kwargs)


def init_visualizations(hps, model, logdir):

    def sample_batch(y, eps):
        n_batch = hps.local_batch_train
        xs = []
        for i in range(int(np.ceil(len(eps) / n_batch))):
            xs.append(model.sample(
                y[i*n_batch:i*n_batch + n_batch], eps[i*n_batch:i*n_batch + n_batch]))
        return np.concatenate(xs)

    def draw_samples(epoch):
        if hvd.rank() != 0:
            return
        
        rows = 10 if hps.image_size <= 64 else 4
        cols = rows
        n_batch = rows*cols
        y = np.asarray([_y % hps.n_y for _y in (
            list(range(cols)) * rows)], dtype='int32')

        # temperatures = [0., .25, .5, .626, .75, .875, 1.] #previously
        temperatures = [0., .25, .5, .6, .7, .8, .9, 1.]

        x_samples = []
        x_samples.append(sample_batch(y, [.0]*n_batch))
        x_samples.append(sample_batch(y, [.25]*n_batch))
        x_samples.append(sample_batch(y, [.5]*n_batch))
        x_samples.append(sample_batch(y, [.6]*n_batch))
        x_samples.append(sample_batch(y, [.7]*n_batch))
        x_samples.append(sample_batch(y, [.8]*n_batch))
        x_samples.append(sample_batch(y, [.9] * n_batch))
        x_samples.append(sample_batch(y, [1.]*n_batch))
        # previously: 0, .25, .5, .625, .75, .875, 1.

        for i in range(len(x_samples)):
            x_sample = np.reshape(
                x_samples[i], (n_batch, hps.image_size, hps.image_size, hps.image_color_num))
            graphics.save_raster(x_sample, logdir +
                                 'epoch_{}_sample_{}.png'.format(epoch, i))
            
    return draw_samples

# ===
# Code for getting data
# ===
def get_data(hps, sess):
    if hps.image_size == -1:
        hps.image_size = {'mnist': 32, 'fashion_mnist':256, 'cifar10': 32, 'imagenet-oord': 64,
                          'imagenet': 256, 'celeba': 256, 'lsun_realnvp': 64, 'lsun': 256}[hps.problem]
    if hps.n_test == -1:
        hps.n_test = {'mnist': 10000, 'fashion_mnist':10000, 'cifar10': 10000, 'imagenet-oord': 50000, 'imagenet': 50000,
                      'celeba': 3000, 'lsun_realnvp': 300*hvd.size(), 'lsun': 300*hvd.size(), 'original': -1}[hps.problem]
    hps.n_y = {'mnist': 10, 'fashion_mnist': 10, 'cifar10': 10, 'imagenet-oord': 1000,
               'imagenet': 1000, 'celeba': 1, 'lsun_realnvp': 1, 'lsun': 1, 'original': 1}[hps.problem]
    if hps.data_dir == "":
        hps.data_dir = {'mnist': None, 'fashion_mnist': None, 'cifar10': None, 'imagenet-oord': '/mnt/host/imagenet-oord-tfr', 'imagenet': '/mnt/host/imagenet-tfr',
                        'celeba': '/mnt/host/celeba-reshard-tfr', 'lsun_realnvp': '/mnt/host/lsun_realnvp', 'lsun': '/mnt/host/lsun'}[hps.problem]

    if hps.problem == 'lsun_realnvp':
        hps.rnd_crop = True
    else:
        hps.rnd_crop = False

    if hps.category:
        hps.data_dir += ('/%s' % hps.category)

    # Use anchor_size to rescale batch size based on image_size
    s = hps.anchor_size
    hps.local_batch_train = hps.n_batch_train * \
        s * s // (hps.image_size * hps.image_size)
    if hps.local_batch_train == 0:
        hps.local_batch_train = 1
#    hps.local_batch_test = {128:100, 64: 50, 32: 25, 16: 10, 8: 5, 4: 2, 2: 2, 1: 1}[
#        hps.local_batch_train]  # round down to closest divisor of 50
    hps.local_batch_test = hps.local_batch_train
    hps.local_batch_init = hps.n_batch_init * \
        s * s // (hps.image_size * hps.image_size)
    if hps.local_batch_init == 0:
        hps.local_batch_init = 1

    print("Rank {} Batch sizes Train {} Test {} Init {}".format(
        hvd.rank(), hps.local_batch_train, hps.local_batch_test, hps.local_batch_init))

    if hps.problem in ['imagenet-oord', 'imagenet', 'celeba', 'lsun_realnvp', 'lsun']:
        hps.direct_iterator = True
        import data_loaders.get_data as v
        train_iterator, test_iterator, data_init = \
            v.get_data(sess, hps.data_dir, hvd.size(), hvd.rank(), hps.pmap, hps.fmap, hps.local_batch_train,
                       hps.local_batch_test, hps.local_batch_init, hps.image_size, hps.rnd_crop)

    elif hps.problem in ['mnist', 'cifar10']:
        hps.direct_iterator = False
        import data_loaders.get_mnist_cifar as v
        train_iterator, test_iterator, data_init = \
            v.get_data(hps.problem, hvd.size(), hvd.rank(), hps.dal, hps.local_batch_train,
                       hps.local_batch_test, hps.local_batch_init, hps.image_size)

    elif hps.problem in ['fashion_mnist']:
        hps.direct_iterator = False
        import data_loaders.get_fashion_mnist as v
        train_iterator, test_iterator, data_init = \
            v.get_data(hvd.size(), hvd.rank(), hps.dal, hps.local_batch_train,
                       hps.local_batch_test, hps.local_batch_init, hps.image_size)
    
    elif hps.problem in ['original']:
        hps.direct_iterator = False
        import data_loaders.get_original as v
        train_iterator, test_iterator, data_init, n_test_ = \
            v.get_data(hps.data_dir, hvd.size(), hvd.rank(), hps.local_batch_train,
                       hps.local_batch_test, hps.local_batch_init, hps.image_size,
                       hps.image_color_num, hps.verbose)
        if hps.n_test == -1:
            print(" n_test {}".format(n_test_) )
            hps.n_test = n_test_
    
    else:
        raise Exception()

    return train_iterator, test_iterator, data_init


def process_results(results):
    stats = ['loss', 'bits_x', 'bits_y', 'pred_loss']
    assert len(stats) == results.shape[0]
    res_dict = {}
    for i in range(len(stats)):
        res_dict[stats[i]] = "{:.4f}".format(results[i])
    return res_dict


def main(hps):

    # Initialize Horovod.
    hvd.init()

    # Create tensorflow session
    sess = tensorflow_session()

    # Download and load dataset.
    tf.set_random_seed(hvd.rank() + hvd.size() * hps.seed)
    np.random.seed(hvd.rank() + hvd.size() * hps.seed)

    # Get data and set train_its and valid_its
    train_iterator, test_iterator, data_init = get_data(hps, sess)
    hps.train_its, hps.test_its, hps.full_test_its = get_its(hps)

    # Create log dir
    logdir = os.path.abspath(hps.logdir) + "/"
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)

    # Create model
    import model
    model = model.model(sess, hps, train_iterator, test_iterator, data_init)

    # Initialize visualization functions
    visualise = init_visualizations(hps, model, logdir)
    if hps.inference_fast:
        infer_fast(sess, model, hps, test_iterator)
    elif hps.inference:
        infer(sess, model, hps, test_iterator)
    elif hps.sample:
        # Perform batch sampling
        sample(sess, model, hps, logdir)
    elif hps.map:
        # Perform image -> latent mapping
        forwardmap(sess, model, hps, logdir)
    elif hps.inversemap:
        # perform latent -> image mapping
        backwardmap(sess, model, hps, logdir)
    elif hps.interpolation:
        interpolation(sess, model, hps, logdir)
    else:
        # Perform training
        train(sess, model, hps, logdir, visualise)



def forwardmap(sess, model, hps, logdir):
    
    print("start")
    
    filelist = []
    for filepath in open(hps.inlistfilepath, "rt"):
        filelist += [filepath.strip()]
    
    for idx in range(hvd.rank(), len(filelist), hvd.size()):
    
        basedir, filename = os.path.split(filelist[idx])
        filebase, ext = os.path.splitext(filename)
        newfilepath = os.path.join(basedir, filebase) + '.npy'
        
        print("[%02.2d / %02.2d] : %s -> %s" % (hvd.rank(), hvd.size(), filelist[idx], newfilepath))

        x = np.array(PIL.Image.open(filelist[idx]))
        if(len(x.shape) == 2):
            assert(hps.image_color_num == 1)
            x = x[:,:,np.newaxis] ## add ch
        x = x[np.newaxis,:,:,0:hps.image_color_num] ## [MiniBatch,y,x,ch]
        
        z = model.encode(x, np.array([0]))
        
        np.save(newfilepath, z)



def backwardmap(sess, model, hps, logdir):
    
    print("start")
    
    filelist = []
    for filepath in open(hps.inlistfilepath, "rt"):
        filelist += [filepath.strip()]

    for idx in range(hvd.rank(), len(filelist), hvd.size()):
    
        basedir, filename = os.path.split(filelist[idx])
        filebase, ext = os.path.splitext(filename)
        newfilepath = os.path.join(basedir, filebase) + '.png'
        
        print("[%02.2d / %02.2d] : %s -> %s" % (hvd.rank(), hvd.size(), filelist[idx], newfilepath))

        z = np.load(filelist[idx])
        
        x = model.decode(np.array([0]), z)
        
        imageio.imwrite(newfilepath, x[0,:,:,0])



def interpolation(sess, model, hps, logdir):
    
    sess.graph.finalize()

    print("start")
    
    # listfile Usage : [in1file.png], [in2file.png], [interpolationratio], [outfile.png]
    
    in1filelist = []
    in2filelist = []
    ratiolist = []
    outfilelist = []
    for line in open(hps.inlistfilepath, "rt"):
        line = line.strip()
        if len(line)==0:
            continue
        inpath1, inpath2, ratio, outpath = map(str.strip, line.split(','))
        in1filelist.append(inpath1)
        in2filelist.append(inpath2)
        ratiolist.append(float(ratio))
        outfilelist.append(outpath)
    
    for in1, in2, r, out in zip(in1filelist, in2filelist, ratiolist, outfilelist):
        
        # read
        x1 = np.array(PIL.Image.open(in1))
        if(len(x1.shape) == 2):
            assert(hps.image_color_num == 1)
            x1 = x1[:,:,np.newaxis] ## add ch
        x1 = x1[np.newaxis,:,:,0:hps.image_color_num] ## [MiniBatch,y,x,ch]
        
        x2 = np.array(PIL.Image.open(in2))
        if(len(x2.shape) == 2):
            assert(hps.image_color_num == 1)
            x2 = x2[:,:,np.newaxis] ## add ch
        x2 = x2[np.newaxis,:,:,0:hps.image_color_num] ## [MiniBatch,y,x,ch]
        
        # cat
        x = np.concatenate((x1,x2), axis=0)
        
        # fore
        z12 = model.encode(x, np.array([0]))
        
        # split
        z1 = z12[0:1,:]
        z2 = z12[1:2,:]
        
        # interpolate
        z = z1 * r + z2 * (1.-r)
        
        # back
        xi = model.decode(np.array([0]), z)
        
        # write
        imageio.imwrite(out, xi[0,:,:,0])



def infer_fast(sess, model, hps, iterator):
    # Only evaluate likelihood function(s), hence this method is faster compared with "infer".
    if hps.direct_iterator:
        iterator = iterator.get_next()
    labels = []
    objectives = []
    for it in range(hps.full_test_its):
        if hps.direct_iterator:
            image, label = sess.run(iterator)
        else:
            image, label = iterator()
        objective = model.objective(image, label)
        objectives.append(objective)
        labels.append(label)
    arr_labels = np.concatenate(labels, axis=0)
    arr_objectives = np.concatenate(objectives, axis=0)
    
    logdir = os.path.abspath(hps.logdir)
    np.save(os.path.join(logdir, 'labels.npy'), arr_labels)
    np.save(os.path.join(logdir, 'likelihoods.npy'), arr_objectives)
    return

def infer(sess, model, hps, iterator):
    # Example of using model in inference mode. Load saved model using hps.restore_path
    # Can provide x, y from files instead of dataset iterator
    # If model is uncondtional, always pass y = np.zeros([bs], dtype=np.int32)
    if hps.direct_iterator:
        iterator = iterator.get_next()
    
    compare_z = None
    if hps.compare_z_path != '':
        # nearest neighbor normal case search
        compare_z = np.load(hps.compare_z_path)
        ratioOriginalZ = hps.ratioOriginalZ
        anchornum = 1000
    
    if hps.q_path != '':
        # all training case hyperplane projection
        q = np.load(hps.q_path)
        ratioOriginalZ = hps.ratioOriginalZ
    
    images = []
    latents = []
    labels = []
    xprimes = []
    objectives = []
    dLdxs = []
    caseidx=0
    for it in range(hps.full_test_its):
        
        if hps.direct_iterator:
            # replace with x, y, attr if you're getting CelebA attributes, also modify get_data
            image, label = sess.run(iterator)
        else:
            image, label = iterator()
        
        objective = model.objective(image, label)
        objectives.append(objective)
        latent = model.encode(image, label)
        image = model.decode(label, latent)
        
        if 0<hps.dLdxnormalizerNum:
            image_float = model.preprocess(image)
            dLdx = np.zeros_like(image_float)
            velocity = np.zeros_like(image_float)
            gamma = 0.9
            for i in range(hps.dLdxnormalizerNum):
                # gradient ascent 
                dLdx, objective, currentz = model.dLdx(image_float, label)
                print("objective : %f" % objective)
                width = np.abs( np.percentile(dLdx.flatten(), 90) - np.percentile(dLdx.flatten(), 10) )
                velocity = gamma * velocity + (dLdx / width * 0.0001)
                image_float = image_float + velocity
                image_float = np.clip(image_float, -0.5, 0.5)
                np.save(os.path.join(os.path.abspath(hps.logdir), 'case%04.4d_%04.4d.npy' % (caseidx, i)), image_float)
                np.save(os.path.join(os.path.abspath(hps.logdir), 'dLdx%04.4d_%04.4d.npy' % (caseidx, i)), dLdx)
                np.save(os.path.join(os.path.abspath(hps.logdir), 'z%04.4d_%04.4d.npy' % (caseidx, i)), currentz)
                
                if 0:
                
                    threshold = 1.5
                    def modifier(x, coeff):
                        return np.where(x<-threshold, 
                            coeff*(x+threshold)-threshold,
                        np.where(x<threshold,
                            x,
                            coeff*(x-threshold)+threshold
                    ))
                    emphasizer = lambda x: modifier(x, 1.5)
                    eraser     = lambda x: modifier(x, 0.0)
                    
                    currentz_orig = np.copy(currentz)
                    
                    # low level emphasized images
                    length = currentz.shape[1]
                    pos = 0
                    currentz[:,pos:pos+length//2] = emphasizer(currentz[:,pos:pos+length//2])
                    pos += length//2
                    x1 = model.decode(label, currentz)
                    currentz[:,pos:pos+length//4] = emphasizer(currentz[:,pos:pos+length//4])
                    pos += length//4
                    x2 = model.decode(label, currentz)
                    currentz[:,pos:pos+length//8] = emphasizer(currentz[:,pos:pos+length//8])
                    pos += length//8
                    x3 = model.decode(label, currentz)
                    currentz[:,pos:pos+length//16] = emphasizer(currentz[:,pos:pos+length//16])
                    pos += length//16
                    x4 = model.decode(label, currentz)
                    currentz[:,pos:pos+length//32] = emphasizer(currentz[:,pos:pos+length//32])
                    pos += length//32
                    x5 = model.decode(label, currentz)
                    currentz[:,pos:pos+length//64] = emphasizer(currentz[:,pos:pos+length//64])
                    pos += length//64
                    x6 = model.decode(label, currentz)
                    currentz[:,pos:pos+length//128] = emphasizer(currentz[:,pos:pos+length//128])
                    pos += length//128
                    x7 = model.decode(label, currentz)
                    np.save(os.path.join(os.path.abspath(hps.logdir), 'x%04.4d_%04.4d.npy_level1_emphasized' % (caseidx, i)), x1)
                    np.save(os.path.join(os.path.abspath(hps.logdir), 'x%04.4d_%04.4d.npy_level2_emphasized' % (caseidx, i)), x2)
                    np.save(os.path.join(os.path.abspath(hps.logdir), 'x%04.4d_%04.4d.npy_level3_emphasized' % (caseidx, i)), x3)
                    np.save(os.path.join(os.path.abspath(hps.logdir), 'x%04.4d_%04.4d.npy_level4_emphasized' % (caseidx, i)), x4)
                    np.save(os.path.join(os.path.abspath(hps.logdir), 'x%04.4d_%04.4d.npy_level5_emphasized' % (caseidx, i)), x5)
                    np.save(os.path.join(os.path.abspath(hps.logdir), 'x%04.4d_%04.4d.npy_level6_emphasized' % (caseidx, i)), x6)
                    np.save(os.path.join(os.path.abspath(hps.logdir), 'x%04.4d_%04.4d.npy_level7_emphasized' % (caseidx, i)), x7)
                    
                    currentz = np.copy(currentz_orig)
                    
                    # low level erased images
                    length = currentz.shape[1]
                    pos = 0
                    currentz[:,pos:pos+length//2] = eraser(currentz[:,pos:pos+length//2])
                    pos += length//2
                    x1 = model.decode(label, currentz)
                    currentz[:,pos:pos+length//4] = eraser(currentz[:,pos:pos+length//4])
                    pos += length//4
                    x2 = model.decode(label, currentz)
                    currentz[:,pos:pos+length//8] = eraser(currentz[:,pos:pos+length//8])
                    pos += length//8
                    x3 = model.decode(label, currentz)
                    currentz[:,pos:pos+length//16] = eraser(currentz[:,pos:pos+length//16])
                    pos += length//16
                    x4 = model.decode(label, currentz)
                    currentz[:,pos:pos+length//32] = eraser(currentz[:,pos:pos+length//32])
                    pos += length//32
                    x5 = model.decode(label, currentz)
                    currentz[:,pos:pos+length//64] = eraser(currentz[:,pos:pos+length//64])
                    pos += length//64
                    x6 = model.decode(label, currentz)
                    currentz[:,pos:pos+length//128] = eraser(currentz[:,pos:pos+length//128])
                    pos += length//128
                    x7 = model.decode(label, currentz)
                    np.save(os.path.join(os.path.abspath(hps.logdir), 'x%04.4d_%04.4d.npy_level1_erased' % (caseidx, i)), x1)
                    np.save(os.path.join(os.path.abspath(hps.logdir), 'x%04.4d_%04.4d.npy_level2_erased' % (caseidx, i)), x2)
                    np.save(os.path.join(os.path.abspath(hps.logdir), 'x%04.4d_%04.4d.npy_level3_erased' % (caseidx, i)), x3)
                    np.save(os.path.join(os.path.abspath(hps.logdir), 'x%04.4d_%04.4d.npy_level4_erased' % (caseidx, i)), x4)
                    np.save(os.path.join(os.path.abspath(hps.logdir), 'x%04.4d_%04.4d.npy_level5_erased' % (caseidx, i)), x5)
                    np.save(os.path.join(os.path.abspath(hps.logdir), 'x%04.4d_%04.4d.npy_level6_erased' % (caseidx, i)), x6)
                    np.save(os.path.join(os.path.abspath(hps.logdir), 'x%04.4d_%04.4d.npy_level7_erased' % (caseidx, i)), x7)
                
            
            dLdxs.append(dLdx)

        images.append(image)
        latents.append(latent)
        labels.append(label)
        
        caseidx+=1
        
        if hps.compare_z_path != '':
            # nearest neighbor normal case search
            zprime = np.ndarray(latent.shape)
            assert compare_z.shape[1] == latent.shape[1]
            
            for indexinbatch in range(latent.shape[0]):
                zcurr = latent[indexinbatch, :]
                dist = np.sum(
                    (compare_z - zcurr)**2.0, axis=1
                )**.5
                nearest10idx = np.argsort(np.reshape(dist, [-1]))[0:anchornum]

                nearest10 = np.ndarray((anchornum, latent.shape[1]))
                for i in range(anchornum):
                    nearest10[i, :] = compare_z[nearest10idx[i], :]

                # make a 'normalized' image
                disloc9 = nearest10[1:,:] - nearest10[0,:]
                dislocz = zcurr - nearest10[0,:]
                matD = np.ndarray([anchornum-1,anchornum-1])
                vecB = np.ndarray([anchornum-1])
                for j in range(anchornum-1):
                    for i in range(anchornum-1):
                        matD[i,j] = np.dot( disloc9[i,:], disloc9[j,:] )
                    vecB[j] = np.dot( disloc9[j,:], dislocz )
                vecL = np.linalg.solve(matD, vecB)
                vecQ = nearest10[0,:]
                for i in range(anchornum-1):
                    vecQ = vecQ + vecL[i] * disloc9[i,:]
                
                zprime[indexinbatch, :] = (
                    + zcurr * ratioOriginalZ
                    + vecQ * (1.0-ratioOriginalZ)
                )
            
            xprime = model.decode(label, zprime)
            
            xprimes.append(xprime)
            
        elif hps.q_path != '':
            zprime = np.ndarray(latent.shape)
            # make a 'normalized' image
            for indexinbatch in range(latent.shape[0]):
                projected = np.reshape( np.matmul( q, np.matmul(q.T, np.squeeze(latent[indexinbatch, :])) ), [1, latent.shape[1]])
                if not hps.q_Linf: # L2 norm version
                    zprime[indexinbatch, :] = (
                        + latent[indexinbatch, :] * ratioOriginalZ
                        + projected * (1.0-ratioOriginalZ)
                    )
                else: # Linf norm version
                    z1 = np.squeeze(latent[indexinbatch, :])
                    p1 = np.squeeze(projected)
                    d = np.max(np.abs(z1 - p1)) * ratioOriginalZ
                    for idxelem in range(z1.shape[0]):
                        zelem = z1[idxelem]
                        pelem = p1[idxelem]
                        if np.abs(zelem - pelem) < d:
                            zprimeelem = zelem
                        else:
                            if pelem < zelem:
                                zprimeelem = pelem + d
                            else:
                                zprimeelem = pelem - d
                        zprime[indexinbatch, idxelem] = zprimeelem
                if 0 < hps.q_amplifier: # 2nd amplify
                    znormalized = np.copy(zprime[indexinbatch, :])
                    zprime[indexinbatch, :] = (
                        + latent[indexinbatch, :] * hps.q_amplifier
                        + zprime[indexinbatch, :] * (1.0-hps.q_amplifier) 
                    )
            xprime = model.decode(label, zprime)
            xprimes.append(xprime)
        
        elif hps.heuristic_z_emphasize:
            # heuristic z emphasizer
            zprime = np.ndarray(latent.shape)
            for indexinbatch in range(latent.shape[0]):
                z1 = np.squeeze(latent[indexinbatch, :])
                for idxelem in range(len(z1)):
                    zelem = z1[idxelem]
                    zelemsign = np.sign(zelem)
                    zelemabs = np.abs(zelem)
                    if hps.heuristic_z_emphasize_lowerbound < zelemabs:
                        zelemabs = (zelemabs - hps.heuristic_z_emphasize_lowerbound) * hps.heuristic_z_emphasize_multiplier + hps.heuristic_z_emphasize_lowerbound
                    zelem = zelemabs * zelemsign
                    z1[idxelem] = zelem
                zprime[indexinbatch, :] = z1
            xprime = model.decode(label, zprime)
            xprimes.append(xprime)
    
    arr_image = np.concatenate(images, axis=0)
    arr_latent = np.concatenate(latents, axis=0)
    arr_label = np.concatenate(labels, axis=0)
    arr_objective = np.concatenate(objectives, axis=0)
    
    logdir = os.path.abspath(hps.logdir)
    
    np.save(os.path.join(logdir, 'x.npy'), arr_image)
    np.save(os.path.join(logdir, 'z.npy'), arr_latent)
    np.save(os.path.join(logdir, 'y.npy'), arr_label)
    np.save(os.path.join(logdir, 'objective.npy'), arr_objective)
    
    if hps.compare_z_path != '' or hps.q_path != '' or hps.heuristic_z_emphasize:
        xprime = np.concatenate(xprimes, axis=0)
        np.save(os.path.join(logdir, 'xprime.npy'), xprime)
    
    if 0<hps.dLdxnormalizerNum:
        arr_dLdx = np.concatenate(dLdxs, axis=0)
        np.save(os.path.join(logdir, 'dLdx.npy'), arr_dLdx)
    
    return latents

def sample(sess, model, hps, logdir):

    sess.graph.finalize()

    if hvd.rank() == 0:
        _print(hps)
        _print('Starting sampling. Outputs to', logdir)
    
    for indexhvd in range(hps.n_sampletotal // hvd.size() // hps.n_sample):
        indexstart = (indexhvd * hvd.size() + hvd.rank()) * hps.n_sample
    
        # filenames
        filenames = []
        for index in range(hps.n_sample):
            filenames.append( 
                os.path.join(logdir, "sample%08.8d.png" % (indexstart + index))
            )
        # Sample
        y = np.zeros(hps.n_sample, dtype='float32')
        eps = np.ones(hps.n_sample, dtype='float32') * hps.temperature
        samples = model.sample(y, eps)
        
        # write pngs
        for sampleidx in range(samples.shape[0]):
            img = np.squeeze(samples[sampleidx,:,:,:])
            graphics.save_image(img, filenames[sampleidx])
    
    if hvd.rank() == 0:
        _print("Finished!")

def train(sess, model, hps, logdir, visualise):
    _print(hps)
    _print('Starting training. Logging to', logdir)
    _print('epoch n_processed n_images ips dtrain dtest dsample dtot train_results test_results msg')

    # Train
    sess.graph.finalize()

    # dump the variable list
    if hvd.rank() == 0:
        _print("trainable variables:")
        for var in tf.trainable_variables():
            _print(var.name)

    # get level lock flags
    if 0 <= hps.highestleveltolock:
        levellockers = [None] * hps.n_levels
        levelunlockers = [None] * hps.n_levels
        for i in range(hps.n_levels):
            for op in tf.get_collection("levellockers"):
                _print(op.name)
                if "levellocker"+str(i) in op.name:
                    levellockers[i] = op
            for op in tf.get_collection("levelunlockers"):
                _print(op.name)
                if "levelunlocker"+str(i) in op.name:
                    levelunlockers[i] = op
            if levellockers[i] == None:
                _print("warning: levellocker not found : level " + str(i))
            if levelunlockers[i] == None:
                _print("warning: levelunlocker not found : level " + str(i))
        if(hps.restore_path == ''):
            # initialize
            for i in range(hps.highestleveltolock+1):
                _print("locking level " + str(i) + " ...")
                sess.run(levellockers[i])

    if(hps.restore_path != ''):
        hps.n_restore = hps.n_restore + 1

    # Initialization parameters in the restoring phase
    n_processed = hps.n_processed
    n_images = hps.n_images
    n_epoch_processed = hps.n_epoch_processed

    train_time = 0.0
    test_loss_best = 999999

    if hvd.rank() == 0:
        # Generate a new log file each time the restore function works
        filename_train =("train_%02d.txt"%hps.n_restore)
        train_logger = ResultLogger(logdir + filename_train, **hps.__dict__)
        filename_test =("test_%02d.txt"%hps.n_restore)
        test_logger = ResultLogger(logdir + filename_test, **hps.__dict__)

    if hps.flow_permutation == 3:
        conv_weight_regularizer_ops = tf.get_collection(
                                        "conv_weight_regularizer")

    tcurr = time.time()
    for epoch in range(n_epoch_processed+1, hps.epochs+1):

        # invertible 1x1 conv weight matrix regularizer
        if hps.flow_permutation == 3:
            sess.run(conv_weight_regularizer_ops)

        # actnorm scale weight regularizer
        sess.run(tf.get_collection("actnorm_regularizer"))

        # unlock a level
        progressiveLRModifier = 1.
        if 0 <= hps.highestleveltolock:
            if 0<epoch and epoch % hps.unlockinterval == 0:
                unlocklevel = hps.highestleveltolock - (epoch // hps.unlockinterval - 1)
                if 0 <= unlocklevel and unlocklevel < hps.n_levels:
                    if hvd.rank() == 0:
                        _print("unlocking level " + str(unlocklevel))
                    sess.run(levelunlockers[unlocklevel])
                    progressiveLRModifier = 1.0e-8

        t = time.time()

        train_results = []
        for it in range(hps.train_its):

            # Set learning rate, linearly annealed from 0 in the first hps.epochs_warmup epochs.
            lr = hps.lr * min(1., n_processed /
                              (hps.n_train * hps.epochs_warmup))
            
            # learning rate limiter for progressive training
            lr = lr * progressiveLRModifier
            progressiveLRModifier = progressiveLRModifier ** (3/4)
            
            # Run a training step synchronously.
            _t = time.time()
            train_results += [model.train(lr)]
            if hps.verbose and hvd.rank() == 0:
                _print(n_processed, time.time()-_t, train_results[-1])
                sys.stdout.flush()

            # Images seen wrt anchor resolution
            n_processed += hvd.size() * hps.n_batch_train
            # Actual images seen at current resolution
            n_images += hvd.size() * hps.local_batch_train

        train_results = np.mean(np.asarray(train_results), axis=0)

        dtrain = time.time() - t
        ips = (hps.train_its * hvd.size() * hps.local_batch_train) / dtrain
        train_time += dtrain

        if hvd.rank() == 0:
            train_logger.log(epoch=epoch, n_processed=n_processed, n_images=n_images, train_time=int(
                train_time), **process_results(train_results))
        if epoch < 10 or (epoch < 50 and epoch % 10 == 0) or epoch % hps.epochs_full_valid == 0:
            test_results = []
            msg = ''

            t = time.time()
            # model.polyak_swap()
            if epoch % hps.epochs_full_valid == 0:
                # Full validation run
                for it in range(hps.full_test_its):
                    test_results += [model.test()]
                test_results = np.mean(np.asarray(test_results), axis=0)
                
                if hvd.rank() == 0:
                    test_logger.log(epoch=epoch, n_processed=n_processed,
                                    n_images=n_images, **process_results(test_results))

                    # Record the hyperparameters for next restoring
                    hps.n_processed = n_processed
                    hps.n_images = n_images
                    hps.n_epoch_processed = epoch

                    # Save checkpoint
                    json_filename = ""
                    if hps.overwrite_each_checkpoint:
                        filename = ("model_%04d_epoches.ckpt" % (epoch))
                        model.save(logdir + filename)
                        hps.restore_path = logdir + filename
                        msg += ' *'
                        json_filename = ("%sparams_%02d_%04depochs.json" % (logdir, hps.n_restore + 1, epoch))

                    elif test_results[0] < test_loss_best:
                        test_loss_best = test_results[0]
                        model.save(logdir + "model_best_loss.ckpt")
                        hps.restore_path = logdir + "model_best_loss.ckpt"
                        json_filename = ("%sparams_%02d.json" % (logdir, hps.n_restore + 1))    
                        msg += ' *'

                    if json_filename != "":
                        with open(json_filename, 'w') as rf:
                            json.dump(hps.__dict__, rf, sort_keys=True, indent=4)
           
            dtest = time.time() - t

            # Sample
            t = time.time()
            if epoch == 1 or epoch == 10 or epoch % hps.epochs_full_sample == 0:
                visualise(epoch)
            dsample = time.time() - t

            if hvd.rank() == 0:
                dcurr = time.time() - tcurr
                tcurr = time.time()
                _print(epoch, n_processed, n_images, "{:.1f} {:.1f} {:.1f} {:.1f} {:.1f}".format(
                    ips, dtrain, dtest, dsample, dcurr), train_results, test_results, msg)

            # model.polyak_swap()

    if hvd.rank() == 0:
        _print("Finished!")

# Get number of training and validation iterations
def get_its(hps):
    # These run for a fixed amount of time. As anchored batch is smaller, we've actually seen fewer examples
    train_its = int(np.ceil(hps.n_train / (hps.n_batch_train * hvd.size())))
    test_its = int(np.ceil(hps.n_test / (hps.n_batch_train * hvd.size())))
    train_epoch = train_its * hps.n_batch_train * hvd.size()

    # Do a full validation run
    if hvd.rank() == 0:
        print(hps.n_test, hps.local_batch_test, hvd.size())
    assert hps.n_test % (hps.local_batch_test * hvd.size()) == 0
    full_test_its = hps.n_test // (hps.local_batch_test * hvd.size())

    if hvd.rank() == 0:
        print("Train epoch size: " + str(train_epoch))
    return train_its, test_its, full_test_its


'''
Create tensorflow session with horovod
'''
def tensorflow_session():
    # Init session and params
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    # Pin GPU to local rank (one GPU per process)
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    sess = tf.Session(config=config)
    return sess


if __name__ == "__main__":

    # This enables a ctr-C without triggering errors
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
    signal.signal(signal.SIGTSTP, signal.SIG_IGN)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action='store_true', help="Verbose mode")
    parser.add_argument("--restore_path", type=str, default='',
                        help="Location of checkpoint to restore")
    parser.add_argument("--inference", action="store_true",
                        help="Use in inference mode")
    parser.add_argument("--inference_fast", action="store_true",
                        help="Use in fast inference mode")                        
    parser.add_argument("--sample", action="store_true",
                        help="Use in sampling mode")
    parser.add_argument("--map", action="store_true",
                        help="Use in mapping mode")
    parser.add_argument("--inversemap", action="store_true",
                        help="Use in inverse mapping mode")
    parser.add_argument("--interpolation", action="store_true",
                        help="Use in map- >interpolation -> inverse map mode")

    parser.add_argument("--inlistfilepath", type=str, default='', 
                        help="input png/z file path list")

    parser.add_argument("--compare_z_path", type=str, default='',
                        help="path of z.npy for nearest normal search")
    parser.add_argument("--q_path", type=str, default='',
                        help="path of q.npy for normal dataset hyperplane projection")
    parser.add_argument("--ratioOriginalZ", type=float, default=0.7,
                        help="mixing ratio of original image Z (vs normalized image Z)")
    parser.add_argument("--q_Linf",  action="store_true",
                        help="use Linf norm for interpolation, otherwise use L2 norm (use with q_path)")
    parser.add_argument("--q_amplifier", type=float, default=0.0,
                        help="coefficient for 2nd amplification (ex. 1.2)")
    parser.add_argument("--heuristic_z_emphasize",  action="store_true",
                        help="use heuristic abnormality emphasizer for z")
    parser.add_argument("--heuristic_z_emphasize_lowerbound", type=float, default=2.0,
                        help="lower bound abs value where z will be emphasized")
    parser.add_argument("--heuristic_z_emphasize_multiplier", type=float, default=1.2,
                        help="z emphasis multiplier")
    parser.add_argument("--logdir", type=str,
                        default='./logs', help="Location to save logs")
    parser.add_argument("--dLdxnormalizerNum", type=int, default=0,
                        help="dLdx gradient ascent normalizer iter num")

    # Dataset hyperparams:
    parser.add_argument("--problem", type=str, default='cifar10',
                        help="Problem (mnist/fashion_mnist/cifar10/imagenet")
    parser.add_argument("--category", type=str,
                        default='', help="LSUN category")
    parser.add_argument("--data_dir", type=str, default='',
                        help="Location of data")
    parser.add_argument("--dal", type=int, default=1,
                        help="Data augmentation level: 0=None, 1=Standard, 2=Extra")
    parser.add_argument("--image_color_num", type=int, default=3,
                        help="3 for color, 1 for monochrome dataset")

    # New dataloader params
    parser.add_argument("--fmap", type=int, default=1,
                        help="# Threads for parallel file reading")
    parser.add_argument("--pmap", type=int, default=16,
                        help="# Threads for parallel map")

    # Optimization hyperparams:
    parser.add_argument("--n_train", type=int,
                        default=50000, help="Train epoch size")
    parser.add_argument("--n_test", type=int,
                        default=-1, help="Valid epoch size")
    parser.add_argument("--n_batch_train", type=int,
                        default=64, help="Minibatch size")
    parser.add_argument("--n_batch_test", type=int,
                        default=50, help="Minibatch size")
    parser.add_argument("--n_batch_init", type=int, default=256,
                        help="Minibatch size for data-dependent init")
    parser.add_argument("--optimizer", type=str,
                        default="adamax", help="adam or adamax")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Base learning rate")
    parser.add_argument("--beta1", type=float, default=.9, help="Adam beta1")
    parser.add_argument("--polyak_epochs", type=float, default=1,
                        help="Nr of averaging epochs for Polyak and beta2")
    parser.add_argument("--weight_decay", type=float, default=1.,
                        help="Weight decay. Switched off by default.")
    parser.add_argument("--epochs", type=int, default=1000000,
                        help="Total number of training epochs")
    parser.add_argument("--epochs_warmup", type=int,
                        default=10, help="Warmup epochs")
    parser.add_argument("--epochs_full_valid", type=int,
                        default=50, help="Epochs between valid")
    parser.add_argument("--overwrite_each_checkpoint", action="store_true",
                        help="Overwrite model for each checkpoint")
    parser.add_argument("--gradient_checkpointing", type=int,
                        default=1, help="Use memory saving gradients")
    parser.add_argument("--n_processed", type=int, default=0,
                        help="Pseudo number of images processed")
    parser.add_argument("--n_images", type=int, default=0,
                        help="Number of images processed")
    parser.add_argument("--n_epoch_processed", type=int, default=0,
                        help="Number of epoches processed")
    parser.add_argument("--n_restore", type=int, default=0,
                        help="Number of restore")

    # Model hyperparams:
    parser.add_argument("--image_size", type=int,
                        default=-1, help="Image size")
    parser.add_argument("--anchor_size", type=int, default=32,
                        help="Anchor size for deciding batch size")
    parser.add_argument("--width", type=int, default=512,
                        help="Width of hidden layers")
    parser.add_argument("--depth", type=int, default=32,
                        help="Depth of network")
    parser.add_argument("--weight_y", type=float, default=0.00,
                        help="Weight of log p(y|x) in weighted loss")
    parser.add_argument("--n_bits_x", type=int, default=8,
                        help="Number of bits of x")
    parser.add_argument("--n_levels", type=int, default=3,
                        help="Number of levels")

    # Model level locker for progressive training
    parser.add_argument("--highestleveltolock", type=int, default=-1,
                        help="The highest level for initial DCNN parameter lock")
    parser.add_argument("--unlockinterval", type=int, default=100,
                        help="The interval (number of ephchs) to unlock each level")

    # Synthesis/Sampling hyperparameters:
    parser.add_argument("--n_sample", type=int, default=1,
                        help="minibatch size for sample")
    parser.add_argument("--n_sampletotal", type=int, default=131072,
                        help="total number of sampling")
    parser.add_argument("--temperature", type=float, default=.7,
                        help="sampling temparature")
    parser.add_argument("--epochs_full_sample", type=int,
                        default=50, help="Epochs between full scale sample")

    # Ablation
    parser.add_argument("--learntop", action="store_true",
                        help="Learn spatial prior")
    parser.add_argument("--ycond", action="store_true",
                        help="Use y conditioning")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--flow_permutation", type=int, default=2,
                        help="Type of flow. 0=reverse (realnvp), 1=shuffle, 2=invconv (ours), 3=svd, 4=invconv (w/ pseudo inverse)")
    parser.add_argument("--flow_coupling", type=int, default=0,
                        help="Coupling type: 0=additive, 1=affine")

    parser.add_argument("--config_file", type=argparse.FileType("r"),
                        default=None, help="Loading parameters from configure file (JSON)")

    hps = parser.parse_args()  # So error if typo

    # load configure file (JSON format)
    print(hps.config_file)

    if hps.config_file is not None:

        try:
            json_data = json.load(hps.config_file)
        except json.JSONDecodeError as e:
            print(sys.exc_info())
            print('JSONDecodeError: ', e)
            sys.exit()
        except ValueError as e:
            print(sys.exc_info())
            print(e)
            sys.exit()
        except Exception as e:
            print(sys.exc_info())
            print(e)
            sys.exit()

        for key, value in json_data.items():
            setattr(hps, key, value)
            # if (getattr(hps, key, "no attr")=="no attr"):
            #     print('Error: parameter "%s" is not defined in %s' % (key, hps.config_file.name))
            #     sys.exit()
            # else:
            #     setattr(hps, key, value)
    
    main(hps)
