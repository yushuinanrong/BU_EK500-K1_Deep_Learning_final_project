# -*- coding: utf-8 -*-
import sugartensor as tf
import tensorflow as tenf
from sugartensor import sg_loss
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import scipy.ndimage
from scipy.misc import toimage
from tensorflow.examples.tutorials.mnist import input_data
import math
import csv
import cifar10

# set log level to debug
tf.sg_verbosity(10)

#
# hyper parameters
#

batch_size = 20   # batch size
H = 32         # Image height
W = 32         # Image width
Channels = 3   # Number of color channels
#
# inputs
#
## load sg_data set
# data_set = input_data.read_data_sets(tf.sg_data.Mnist._data_dir, reshape=False, one_hot=False)
## save each sg_data set
# train = data_set.train
# valid = data_set.validation
# test  = data_set.test

# MNIST input tensor ( with QueueRunner )
#data = tf.sg_data.Mnist(batch_size = batch_size)#batch_size=batch_size
data = cifar10.Data(batch_size=batch_size)

# print l,c,x,s

# input images
# x = data.train.image
# x = tf.sg_data.Mnist(batch_size = data.train.num_batch*128 )
x = data.train.image


# corrupted image
x_small   = tf.image.resize_nearest_neighbor(x, (H/2, W/2))
x_bicubic = tf.image.resize_images(x_small, (H, W), method=tf.image.ResizeMethod.BICUBIC)#.sg_squeeze()
x_nearest = tf.image.resize_images(x_small, (H, W), tf.image.ResizeMethod.NEAREST_NEIGHBOR)#.sg_squeeze()



# sess = tf.InteractiveSession()

# Add print operation
# a = tf.Print(x_small, [x_small], message="This is a: ")

# print (tf.Print(x_small))
# create generator
#
# I've used ESPCN scheme
# http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shi_Real-Time_Single_Image_CVPR_2016_paper.pdf
#

# generator network
# with tf.sg_context(name='generator', act='relu', bn=True):
#     gen = (x_small
#            .sg_conv(dim=32)
#            .sg_conv()
#            .sg_conv(dim=4, act='sigmoid', bn=False)
#            .sg_periodic_shuffle(factor=2)
#            .sg_squeeze())

with tf.sg_context(name='generator', act='relu', bn=True):
    gen = (x_small
           .sg_conv(dim=32)
           .sg_conv(dim=32)
           .sg_conv()
           .sg_conv(dim=4*Channels, act='sigmoid', bn=False)
           .sg_periodic_shuffle(factor=2))
           #.sg_squeeze())


# #
# # run generator
# #
def log10(x):
  numerator   = tenf.log(x)
  denominator = tenf.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

def MSE(y,y_pred):
    # MSE = sg_loss.sg_mse(y,y_pred)
        # squared error
    MSE = tf.reduce_mean(tf.squared_difference(y_pred, y.sg_squeeze()))
    # print MSE
    # count = 0
    # MSE   = 0
    (num,h,w,c) = y.shape

    # MSE = y_pred.sg_mse(target = y)
    # print num
    # for idx in xrange(0,num):
    #     im  = (y[idx,:,:,:])
    #     MSE = MSE + tf.reduce_mean(np.square(im - (y_pred[idx,:,:])))
    #     count = count + 1
    # PSNR = -10*log10(MSE/num)
    return MSE



# Calculate mean MSE and PSNR
# with tf.Session() as sess:
#      with tf.sg_queue_context(sess):
#         # with tf.sg_context(sess):
#         tf.sg_init(sess)

#         # restore parameters
#         saver = tf.train.Saver()
#         # saver.restore(sess, tf.train.latest_checkpoint('/home/vivo/Desktop/SuperResolution/SRGAN-master/asset/train/ckpt'))
#         saver.restore(sess, "./asset/train/GAN_SR/model_cifar10_down4/model.ckpt-65604")

#         # run generator
#         gt, small, low, bicubic, sr = sess.run([x, x_small, x_nearest, x_bicubic, gen])
#         bicubic = bicubic.clip(0.0, 1.0);

#         diff_SR = sess.run(tf.abs(tf.subtract(gt, sr)))
#         diff_BC = sess.run(tf.abs(tf.subtract(gt, bicubic)))
#             # MSE = sess.run(MSE(x,sr))
#         MSE_SR = np.zeros(gt.shape[0])
#         MSE_BC = np.zeros(gt.shape[0])

#         print gt.shape

#         for i in range(gt.shape[0]):
#             MSE_SR[i] = np.mean(np.square(diff_SR[i, :, :, :]))
#             MSE_BC[i] = np.mean(np.square(diff_BC[i, :, :, :]))

#         PSNR_SR = 10*np.log10(1.0/(MSE_SR))
#         PSNR_BC = 10*np.log10(1.0/(MSE_BC))
            
#         print "SRGAN MSE = " + str(np.mean(MSE_SR))
#         print "Bicubic MSE = " + str(np.mean(MSE_BC))

#         PSNR = 10*math.log10(1.0/(np.mean(MSE_SR)))
#         print "SRGAN PSNR = " + str(PSNR)
#         print "Bicubic PSNR = " + str(10*math.log10(1.0/(np.mean(MSE_BC))))


# Plot and save generated images
for k in range(10):
    fig_name = './asset/train/GAN_SR/images_' + str(k+1) + '.png'
    csv_name = './asset/train/GAN_SR/MSE_' + str(k+1) + '.csv'
    with tf.Session() as sess:
        with tf.sg_queue_context(sess):
        # with tf.sg_context(sess):

            tf.sg_init(sess)

            # restore parameters
            saver = tf.train.Saver()
            # saver.restore(sess, tf.train.latest_checkpoint('/home/vivo/Desktop/SuperResolution/SRGAN-master/asset/train/ckpt'))
            saver.restore(sess, "./asset/train/GAN_SR/model_cifar10/model.ckpt-627924")

            # run generator
            gt, small, low, bicubic, sr = sess.run([x, x_small, x_nearest, x_bicubic, gen])
            bicubic = bicubic.clip(0.0, 1.0);

            #diff = sess.run(tf.reduce_mean(tf.abs(tf.subtract(gt, sr)), axis = 3))
            #diff2 = sess.run(tf.reduce_mean(tf.abs(tf.subtract(gt, bicubic)), axis = 3))
            diff_SR = sess.run(tf.abs(tf.subtract(gt, sr)))
            diff_BC = sess.run(tf.abs(tf.subtract(gt, bicubic)))
            # MSE = sess.run(MSE(x,sr))
            MSE_SR = np.zeros(gt.shape[0])
            MSE_BC = np.zeros(gt.shape[0])

            for i in range(gt.shape[0]):
                MSE_SR[i] = np.mean(np.square(diff_SR[i, :, :, :]))
                MSE_BC[i] = np.mean(np.square(diff_BC[i, :, :, :]))

            PSNR_SR = 10*np.log10(1.0/(MSE_SR))
            PSNR_BC = 10*np.log10(1.0/(MSE_BC))
            
            print "SRGAN MSE = " + str(np.mean(MSE_SR))
            print "Bicubic MSE = " + str(np.mean(MSE_BC))

            PSNR = 10*math.log10(1.0/(np.mean(MSE_SR)))
            print "SRGAN PSNR = " + str(PSNR)
            print "Bicubic PSNR = " + str(10*math.log10(1.0/(np.mean(MSE_BC))))

            with open(csv_name, 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(["Image No.", "MSE_GAN", "MSE_Bicubic", "PSNR_GAN", "PSNR_Bicubic"])
                for i in range(gt.shape[0]):
                    writer.writerow([str(i+1), str(MSE_SR[i]), str(MSE_BC[i]), str(PSNR_SR[i]), str(PSNR_BC[i])])


            ###############plot result
            _, ax = plt.subplots(10, 8, sharex=False, sharey=False)
            for i in range(10):
                for j in range(2):
                    ax[i][j*4].imshow(low[i*2+j])
                    ax[i][j*4].set_axis_off()
                    ax[i][j*4+1].imshow(bicubic[i*2+j])
                    ax[i][j*4+1].set_axis_off()
                    ax[i][j*4+2].imshow(sr[i*2+j])
                    ax[i][j*4+2].set_axis_off()
                    ax[i][j*4+3].imshow(gt[i*2+j])
                    ax[i][j*4+3].set_axis_off()
                    #ax[i][j*6+4].imshow(diff2[i*3+j], 'gray')
                    #ax[i][j*6+4].set_axis_off()
                    #ax[i][j*6+5].imshow(diff[i*3+j], 'gray')
                    #ax[i][j*6+5].set_axis_off()
            plt.subplots_adjust(wspace=0, hspace=0)


            plt.savefig(fig_name, dpi=600)
            tf.sg_info('Sample image saved to "%s"' % fig_name)
            plt.close()

