
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from imblearn.over_sampling import ADASYN
import numpy
import math
import time
from tensorflow.keras.optimizers import Adam


# 加载数据
def load_nsl_kdd_dataset():
    # Feature names of the dataset
    feature_names = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty_level'
    ]

    # Read data from CSV file
    df = pd.read_csv('/Users/smile/Desktop/maste paper/python project/KDD/KDDTrain+.txt', header=None, names=feature_names)

    return df


# 数据预处理





class AE(Model):

    def __init__(self, M_type, train_features, init_lr, active_f, feature_size, encode_size, num_output, train_dataset,num_train,train_size):
        self.train_features = train_features
        self.train_dataset = train_dataset
        self.M_type = M_type
        self.init_lr = init_lr
        self.active_f = active_f
        self.feature_size = feature_size
        self.encode_size = encode_size
        self.num_output = num_output
        self.num_train = num_train  # 训练的总轮数，表示整个数据集将被遍历多少次。
        self.train_size = train_size

        self.var_imp = []

    def AE_train(self):
        input_layer = Input(shape=(self.feature_size,), name='input')

        # 创建编码器部分
        encode_layer1 = Dense(units=self.encode_size[0], kernel_initializer='glorot_normal',use_bias=True,activation=self.active_f, name='encode_layer1')(input_layer)
        if self.M_type == "AE_Dropout":
            encode_layer2 = tf.keras.layers.Dropout(0.9)(
                Dense(units=self.encode_size[1], kernel_initializer='glorot_normal',use_bias=True,activation=self.active_f, name='encode_layer2')(encode_layer1))
            encode_layer3 = Dense(units=self.encode_size[2], kernel_initializer='glorot_normal',use_bias=True,activation=self.active_f, name='encode_layer3')(encode_layer2)
        else:
            encode_layer2 = Dense(units=self.encode_size[1], kernel_initializer='glorot_normal',use_bias=True,activation=self.active_f, name='encode_layer2')(encode_layer1)
            encode_layer3 = Dense(units=self.encode_size[2], kernel_initializer='glorot_normal',use_bias=True,activation=self.active_f, name='encode_layer3')(encode_layer2)
        encode_layer4 = Dense(units=self.encode_size[3], kernel_initializer='glorot_normal',use_bias=True,activation=self.active_f, name='encode_layer4')(encode_layer3)

        # 创建解码器部分
        decode_layer1 = Dense(units=self.encode_size[2],kernel_initializer='glorot_normal',use_bias=True, activation=self.active_f, name='decode_layer1')(encode_layer4)
        decode_layer2 = Dense(units=self.encode_size[1], kernel_initializer='glorot_normal',use_bias=True,activation=self.active_f, name='decode_layer2')(decode_layer1)
        decode_layer3 = Dense(units=self.encode_size[0],kernel_initializer='glorot_normal',use_bias=True, activation=self.active_f, name='decode_layer3')(decode_layer2)
        decode_output = Dense(units=self.feature_size, kernel_initializer='glorot_normal',use_bias=True,activation=self.active_f, name='decode_output')(decode_layer3)

        # 构建自动编码器模型
        autoencoder = Model(inputs=input_layer, outputs=decode_output)

        # 编译模型
        autoencoder.compile(optimizer=optimizer, loss='mse')
        x = []
        y = []
        l = []
        with tf.device('/gpu:0'):
            # 初始化模型的权重
            autoencoder.compile(optimizer=optimizer, loss='mse')

            # 获取模型中的可训练变量
            vars = autoencoder.trainable_variables


            for epoch in range(self.num_train):
                num_batches = len(self.train_features) // self.train_size
                self.init_lr = self.init_lr * (0.7 ** (epoch // 25))  # learning rate decay
                optimizer.lr.assign(self.init_lr)  # 更新学习率

                for iteration in range(num_batches):
                    X_batch = self.next_trainset(self.train_size, self.train_dataset)
                    autoencoder.fit(X_batch, X_batch, batch_size=self.train_size, epochs=1, verbose=0)

                # 在 epoch 结束时评估整个训练集的损失
                train_loss = autoencoder.evaluate(self.train_dataset, self.train_dataset, batch_size=self.train_size,
                                                  verbose=0)
                print("epoch {} loss {}".format(epoch, train_loss))
                x.append(epoch)
                y.append(train_loss)

            # 绘制损失曲线
            plt.rcParams['figure.figsize'] = (30, 30)
            plt.plot(x, y)
            plt.xlabel("Train numbers")
            plt.ylabel("Autoencoder Training loss")
            plt.show()

            # 获取特定变量的值
            for var, val in zip(vars, autoencoder.get_weights()):
                if var.shape == (43, 38) or var.shape == (41, 35):
                    l = val

            # 计算变量重要性和选择特征
            self.variable_importance(l)
            bf = self.select_features()

            return bf

    def training(self):

        X = tf.placeholder(tf.float32, shape=[None, self.feature_size])

        #initializer = tf.variance_scaling_initializer()

        #weight_1 = tf.Variable(initializer([self.feature_size, self.encode_size[0]]), dtype=tf.float32)
        #weight_2 = tf.Variable(initializer([self.encode_size[0], self.encode_size[1]]), dtype=tf.float32)
        #weight_3 = tf.Variable(initializer([self.encode_size[1], self.encode_size[2]]), dtype=tf.float32)
        #weight_4 = tf.Variable(initializer([self.encode_size[2], self.encode_size[3]]), dtype=tf.float32)
        #weight_5 = tf.Variable(initializer([self.encode_size[3], self.num_output]), dtype=tf.float32)

        rand = numpy.random.RandomState(int(time.time()))#Xavier初始化方法 对每个layer的weight
        r = math.sqrt(6) / math.sqrt(self.feature_size + self.encode_size[0] + 1)
        weight_1 = numpy.asarray(rand.uniform(low=-r, high=r, size=(self.encode_size[0], self.feature_size)))
        r = math.sqrt(6) / math.sqrt(self.encode_size[0] + self.encode_size[1] + 1)
        weight_2 = numpy.asarray(rand.uniform(low=-r, high=r, size=(self.encode_size[1], self.encode_size[0])))
        r = math.sqrt(6) / math.sqrt(self.encode_size[1] + self.encode_size[2] + 1)
        weight_3 = numpy.asarray(rand.uniform(low=-r, high=r, size=(self.encode_size[2], self.encode_size[1])))
        r = math.sqrt(6) / math.sqrt(self.encode_size[2] + self.encode_size[3] + 1)
        weight_4 = numpy.asarray(rand.uniform(low=-r, high=r, size=(self.encode_size[3], self.encode_size[2])))
        r = math.sqrt(6) / math.sqrt(self.num_output+ self.encode_size[3] + 1)
        weight_5 = numpy.asarray(rand.uniform(low=-r, high=r, size=(self.num_output, self.encode_size[3])))



        b1 = tf.Variable(tf.zeros(self.encode_size[0]))
        b2 = tf.Variable(tf.zeros(self.encode_size[1]))
        b3 = tf.Variable(tf.zeros(self.encode_size[2]))
        b4 = tf.Variable(tf.zeros(self.encode_size[3]))
        b5 = tf.Variable(tf.zeros(self.num_output))
#########  create_encode_node
        encode_node1 = self.active_f(tf.matmul(X, weight_1) + b1)
        if self.M_type == "AE_Dropout":
            encode_node2 = tf.nn.dropout(tf.matmul(encode_node1, weight_2) + b2, 0.9)
            encode_node3 = self.active_f(tf.matmul(encode_node2, weight_3) + b3, 0.9)

        else:
            encode_node2 = self.active_f(tf.matmul(encode_node1, weight_2) + b2)
            encode_node3 = self.active_f(tf.matmul(encode_node2, weight_3) + b3)

        encode_node4 = self.active_f(tf.matmul(encode_node3, weight_4) + b4)
        decode_node = self.active_f(tf.matmul(encode_node4, weight_5) + b5)

        loss = tf.reduce_mean(tf.pow(tf.subtract(decode_node, X), 2.0))
        optimizer = tf.train.AdamOptimizer(self.init_lr)
        train = optimizer.minimize(loss)



        init = tf.global_variables_initializer()
        #num_train = 250  #训练的总轮数，表示整个数据集将被遍历多少次。
        #rain_size = 10000 #每个小批量样本的数量

        x = []
        y = []
        l = []

        with tf.device('/gpu:0'):
            with tf.Session() as sess:
                sess.run(init)
                vars = tf.trainable_variables()
                vars_vals = sess.run(vars)

                for epoch in range(num_train):
                    num_batches = len(self.train_features) // train_size
                    self.init_lr = self.init_lr * (0.7 ** (epoch // 25))# learning rate decay
                    for iteration in range(num_batches):
                        X_batch = self.next_trainset(train_size, self.train_dataset)
                        sess.run(train, feed_dict={X: X_batch})
                    train_loss = loss.eval(feed_dict={X: X_batch})
                    print("epoch {} loss {}".format(epoch, train_loss))
                    x.append(epoch)
                    y.append(train_loss)

        for var, val in zip(vars, vars_vals):
            if var.get_shape() == (43, 38) or var.get_shape() == (41, 35):
                l = val

        plt.rcParams['figure.figsize'] = (30, 30)
        plt.plot(x, y)
        plt.xlabel("Train numbers")
        plt.ylabel("Autoencoder Training loss")
        plt.show()

        self.variable_importance(l)
        bf = self.select_features()

        return bf
    def next_trainset(self, size, data):

        adasyn = ADASYN()

        # 使用 ADASYN 进行数据增强
        X_resampled, y_resampled = adasyn.fit_resample(data[:, :-1], data[:, -1])

        # 随机选择一批样本
        idx = np.arange(0, len(X_resampled))
        np.random.shuffle(idx)
        idx = idx[:size]
        data_shuffle = [X_resampled[i] for i in idx]



        return np.asarray(data_shuffle)

    def variable_importance(self, l):

        for row in l:
            total_weight = 0
            for a in row:
                total_weight += a
            self.var_imp.append(total_weight)
    def select_features(self):
        features = list(range(0, self.train_features.shape[1]))
        feature_names = list(self.train_dataset.columns)

        plt.close()
        plt.rcParams['figure.figsize'] = (30, 30)
        x_vals = np.arange(len(features))
        print(x_vals.shape)
        print(len(self.var_imp))
        plt.bar(x_vals, self.var_imp, align='center', alpha=1)
        plt.xticks(x_vals, features)
        plt.xlabel("Feature Indices")
        plt.ylabel("Feature Importance Values")
        plt.show()

        best_features = sorted(range(len(self.var_imp)), key=lambda i: self.var_imp[i], reverse=True)[:30]
        print("\nTop 30 selected features:")
        for i in best_features:
            print(feature_names[i])

        print("\nVariance", np.var(self.var_imp))

        return best_features


# 构建自动编码器模型



# 主程序
if __name__ == "__main__":
    # 加载数据集
