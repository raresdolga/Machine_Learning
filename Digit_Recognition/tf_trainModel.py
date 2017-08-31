import numpy as np
import pandas as pd
import tensorflow as tf

def train_main():
    LEARNING_RATE = 0.001
    TRAINING_EPOCHS = 3000
    BATCH_SIZE = 100
    DISPLAY_STEP = 10
    DROPOUT_CONV = 0.8
    DROPOUT_HIDDEN = 0.6
    VALIDATION_SIZE = 2000
    lable_c = 10

    data = pd.read_csv('all_Data/train.csv')
    img = data.iloc[:,1:].values
    img = img.astype(np.float)

    raw_labels = data.iloc[:, 0].values.ravel()
    labels = format_lables(raw_labels,lable_c)
    img, img_height, img_width, img_size = normalize_data(img)

    # split into validation and training
    val_img = img[0:VALIDATION_SIZE, :]
    val_labels = labels[0:VALIDATION_SIZE, :]
    train_img = img[VALIDATION_SIZE:, :]
    train_labels = labels[VALIDATION_SIZE:, :]
    # img size is the column length: 28*28=784
    X = tf.placeholder('float',shape=[None, img_size])
    Y = tf.placeholder('float', shape= [None, lable_c])

    # create two connected layers using tensorFlow gloval variables(create if do not exist or get if exist)
    # Convolution layers
    W1 = tf.get_variable("W1", shape=[5,5,1,32], initializer=random_init_inInterval(5*5*1,32))
    W2 = tf.get_variable("W2", shape = [5,5,32,64], initializer=random_init_inInterval(5*5*32,64))
    # fully connected layers
    W3_FC = tf.get_variable("W3_FC",shape = [64*7*7,1024], initializer=random_init_inInterval(64*7*7,1024))
    W4_FC = tf.get_variable("W4_FC",shape = [1024,lable_c], initializer=random_init_inInterval(1024,lable_c))
    # Bias initialization
    B1 = biasInit([32])
    B2 = biasInit([64])
    B3_FC = biasInit([1024])
    B4_FC = biasInit([lable_c])

    drop_convutional = tf.placeholder('float')
    drop_hidden = tf.placeholder('float')
    X1 = tf.reshape(X,[-1, img_width, img_height, 1])

    # layer 1
    l1_conv = tf.nn.relu(conv2d(X1, W1) + B1)
    l1_pool = max_pool_2x2(l1_conv)
    l1_drop = tf.nn.dropout(l1_pool, drop_convutional)

    # layer 2
    l2_conv = tf.nn.relu(conv2d(l1_drop, W2) + B2)
    l2_pool = max_pool_2x2(l2_conv)
    l2_drop = tf.nn.dropout(l2_pool, drop_convutional)

   # layer 3 fully complete
    l3_flat = tf.reshape(l2_drop, [-1, W3_FC.get_shape().as_list()[0]])
    l3_feed = tf.nn.relu(tf.matmul(l3_flat, W3_FC) + B3_FC)
    l3_drop = tf.nn.dropout(l3_feed, drop_hidden)

    # last layer fully complete ( output)
    Y_predicted = tf.nn.softmax(tf.matmul(l3_drop, W4_FC) + B4_FC)

    # Cost function & training
    cost_f = -tf.reduce_sum(Y*tf.log(Y_predicted))
    regularizer = (tf.nn.l2_loss(W3_FC) + tf.nn.l2_loss(B3_FC) + tf.nn.l2_loss(W4_FC) + tf.nn.l2_loss(B4_FC))
    # lambda is 5e -4
    cost_f += 5e-4*regularizer


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],strides=[1, 2, 2, 1], padding='SAME')


def format_lables(lables,lable_c):
    m = len(lables)
    # 10 is the number of digits we have to recognize 0,2,3,...9
    y = np.zeros((m, lable_c))
    for i in range(len(lables)):
        y[i][lables[i]] = 1
    return y


def normalize_data(trainSet):
    trainSet = np.multiply(trainSet,1.0/255.0)
    # image size is the length of a row and all rows have common length
    size = len(trainSet[0])
    img_width = img_height = np.ceil(np.sqrt(size)).astype(np.uint8)
    return trainSet, img_height, img_width, size


def weight_variable(shape):
    ini = tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(ini)


def random_init_inInterval(nr_in, nr_out, uniform = True):
        if uniform:
            init_range = tf.sqrt(6.0/(nr_in + nr_out))
            return tf.random_uniform_initializer(-init_range,init_range)
        else:
            sttdev = tf.sqrt(3.0/(nr_in + nr_out))
            return tf.truncated_normal_initializer(stddev=sttdev)


def biasInit(shape):
    ini = tf.constant(0.1, shape=shape)
    return tf.Variable(ini)


if __name__  == "__main__":train_main()
