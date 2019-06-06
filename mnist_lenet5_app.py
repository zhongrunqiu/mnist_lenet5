#coding:utf-8
import tensorflow as tf
from PIL import Image
import mnist_lenet5_forward
import mnist_lenet5_backward
import numpy as np

def restore_model(testPicArr):
    with tf.Graph().as_default():
        x = tf.placeholder(tf.float32,[
            1,
            mnist_lenet5_forward.IMAGE_SIZE,
            mnist_lenet5_forward.IMAGE_SIZE,
            mnist_lenet5_forward.NUM_CHANNELS
        ])
        y = mnist_lenet5_forward.forward(x,False,None)
        preValue = tf.argmax(y,1)

        variable_averages = tf.train.ExponentialMovingAverage(mnist_lenet5_backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(mnist_lenet5_backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess,ckpt.model_checkpoint_path)
                reshaped_x = np.reshape(testPicArr,(
                    1,
                    mnist_lenet5_forward.IMAGE_SIZE,
                    mnist_lenet5_forward.IMAGE_SIZE,
                    mnist_lenet5_forward.NUM_CHANNELS
                ))
                preValue = sess.run(preValue,feed_dict={x:reshaped_x})
                return preValue

def pre_pic(testPic):
    img = Image.open(testPic)
    reIm = img.resize((28,28),Image.ANTIALIAS)
    im_arr = np.array(reIm.convert('L'))
    threshold = 30
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255
    nm_arr = im_arr.reshape([1,784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr,1./255)
    return img_ready

def application():
    testNum = int(input("Input the number of test pictures:"))
    for i in range(testNum):
        testPic = input("The path of test picture:")
        testPicArr = pre_pic(testPic)
        preValue = restore_model(testPicArr)
        print("Thr prediction nunber is:",preValue)

def main():
    application()

if __name__ == '__main__':
    main()