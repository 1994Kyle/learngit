# TensorFlow实现LeNet实例
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time  # 用来观察模型训练的时间
import matplotlib.pyplot as plt  # 用于绘制准确率曲线


# 载入mnist数据集
mnist = input_data.read_data_sets("C:\\Users\\Kyle Lee\\Anaconda3\\Lib\\site-packages\\tensorflow\\examples\\tutorials\\mnist", one_hot=True)
# 定义占位符，声明输入图片的数据和类别及输出的数据和类别
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# 对输入数据进行转化，需要转变为4维的Tensor用于卷积神经网络的输入
print(mnist.train.images.shape)  # mnist数据集是以[None, 784]的数据格式存放的
x_image = tf.reshape(x, [-1, 28, 28, 1])


# 定义权重函数
def weight_variables(shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)


# 定义偏置函数
def bias_variables(shape):
    initial = tf.truncated_normal(shape)
    return tf.Variable(initial)


# 定义卷积运算函数
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


# 定义池化运算函数
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


"""
这里使用的激活函数为sigmoid，传统神经网络中最常用的两个激活函数为sigmoid和tanh, sigmoid
被视为神经网络的核心所在。非线性的sigmoid函数对中央区的信号增益较大，对两侧区的信号增益较小
在信号的特征空间映射上，有很好的效果
sigmoid和tanh激活函数的缺点：左右两端在很大程度上接近极值，容易饱和，使得神经元梯度接近0，这使得
模型在计算时会多次计算接近于0的梯度，导致花费时间而权重又得不到更新
为了克服这一缺点，出现了一种新的函数，ReLU函数
ReLU函数的优点：
1. 收敛快：对于达到阈值的数据其激活力度随着数值的加大而增大，且呈现一个线性关系
2. 计算简单：max(0, x)
3. 不易过拟合：一部分神经元如果在计算时有过大的梯度，则该神经元的梯度将会被强行设置为0，在其后
的训练中处于失活状态，虽然会导致多样化的丢失，但是也能防止过拟合
ReLU的缺点：
不同的学习率对ReLU模型的训练有很大影响，学习率设置不当会造成大量的神经元被锁死
"""


# 定义第一卷积层和第一池化层
w_conv1 = weight_variables([5, 5, 1, 6])  # 注意这里卷积核的个数为6个
b_conv1 = bias_variables([6])
# 第一层卷积输出
conv1 = conv2d(x_image, w_conv1)
h_conv1 = tf.nn.sigmoid(tf.add(conv1, b_conv1))  # 卷积加偏置，经过激活函数输出，此为卷积层的输出
# 第二层池化输出
h_pool1 = max_pool_2x2(h_conv1)  # 此为池化层的输出

# 定义第二卷积层和第二池化层
w_conv2 = weight_variables([5, 5, 6, 16])  # 注意这里卷积核的个数为16个
b_conv2 = bias_variables([16])
# 第三层卷积输出
conv2 = conv2d(h_pool1, w_conv2)
h_conv2 = tf.nn.sigmoid(tf.add(conv2, b_conv2))
# 第四层池化输出
h_pool2 = max_pool_2x2(h_conv2)
# 定义第三卷积层，注意这次没有池化层，而是直接连接全连接层
w_conv3 = weight_variables([5, 5, 16, 120])  # 卷积核的个数为120个
b_conv3 = bias_variables([120])
# 第五层卷积输出，输出的shape为[?,7,7,120]
conv3 = conv2d(h_pool2, w_conv3)
h_conv3 = tf.nn.sigmoid(tf.add(conv3, b_conv3))

# 定义第六层全连接层
w_fc1 = weight_variables([7*7*120, 80])
b_fc1 = bias_variables([80])
# 把即将进入全连接层的输入h_conv3重塑为一维向量
h_conv3_flat = tf.reshape(h_conv3, [-1, 7*7*120])
# 第六层全连接层的输出
h_fc1 = tf.nn.sigmoid(tf.add(tf.matmul(h_conv3_flat, w_fc1), b_fc1))

# 最后一层全连接层，使用softmax进行分类
w_fc2 = weight_variables([80, 10])
b_fc2 = bias_variables([10])
y_model = tf.nn.softmax(tf.add(tf.matmul(h_fc1, w_fc2), b_fc2))

# 损失函数，采用交叉熵
loss = -tf.reduce_sum(y_ * tf.log(y_model))
# 训练
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# 准确率
correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
train_c = []  # 用来承载训练准确率，以便于绘制准确率曲线
test_c = []  # 用来承载测试准确率，以便于绘制测试准确率曲线

# 启动训练过程
sess = tf.InteractiveSession()
init = tf.global_variables_initializer()
sess.run(init)

start_time = time.time()
for i in range(2000):
    # 获取训练数据
    batch = mnist.train.next_batch(200)
    # 训练数据
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    # 每迭代100个batch，对当前训练数据进行测试，输出训练acc和测试acc
    if i % 2 == 0:
        train_acc = accuracy.eval(feed_dict={x: batch[0], y_: batch[1]})
        test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
        train_c.append(train_acc)
        test_c.append(test_acc)
        print("step %d: \ntraining accuracy %g, testing accuracy %g" % (i, train_acc, test_acc))
        # 计算间隔时间
        end_time = time.time()
        print("time: ", (end_time - start_time))
        start_time = end_time

sess.close()
plt.plot(train_c, label="train accuracy")
plt.plot(test_c, label="test accuracy")
plt.legend()
plt.tight_layout()
plt.show()
plt.savefig("C:\\Users\\Kyle Lee\\PycharmProjects\\hello\\accu.png", dpi=200)