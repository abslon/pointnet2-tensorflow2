import sys

sys.path.insert(0, './')

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout

from pnet2_layers.layers import Pointnet_SA, Pointnet_FP


class SEM_SEG_Model(Model):
    def __init__(self, num_classes, bn=False, activation=tf.nn.leaky_relu):
        super(SEM_SEG_Model, self).__init__()

        self.activation = activation
        self.keep_prob = 0.5
        self.num_classes = num_classes
        self.bn = bn

        self.kernel_initializer = 'glorot_normal'
        self.kernel_regularizer = None

        self.init_network()

    def init_network(self):
        self.sa_1 = Pointnet_SA(
            npoint=288,
            radius=0.2,
            nsample=8,
            mlp=[32, 32, 64],
            group_all=False,
            activation=self.activation,
            bn=self.bn
        )

        self.sa_2 = Pointnet_SA(
            npoint=14,
            radius=0.4,
            nsample=4,
            mlp=[64, 64, 128],
            group_all=False,
            activation=self.activation,
            bn=self.bn
        )

        self.fp_3 = Pointnet_FP(
            mlp=[256, 128],
            activation=self.activation,
            bn=self.bn
        )

        self.fp_4 = Pointnet_FP(
            mlp=[128, 128, 128],
            activation=self.activation,
            bn=self.bn
        )

        self.dense1 = Dense(128, activation=self.activation)

        self.dropout1 = Dropout(self.keep_prob)

        self.dense2 = Dense(self.num_classes, activation=None)

    def call(self, input, training=True):
        l0_xyz = input[..., :3]
        l0_points = input[..., 3:]

        l1_xyz, l1_points = self.sa_1(l0_xyz, l0_points, training=training)
        l2_xyz, l2_points = self.sa_2(l1_xyz, l1_points, training=training)

        l1_points = self.fp_3(l1_xyz, l2_xyz, l1_points, l2_points, training=training)
        l0_points = self.fp_4(l0_xyz, l1_xyz, l0_points, l1_points, training=training)

        net = self.dense1(l0_points)
        net = self.dropout1(net)
        pred = self.dense2(net)

        return pred


class PN_sph(Model):
    def __init__(self):
        super(PN_sph, self).__init__()
        self.PN_density = SEM_SEG_Model(num_classes=2)
        self.PN_force = SEM_SEG_Model(num_classes=3)

    def call(self, inputs, training=None, mask=None):
        xyz = inputs[..., :3]
        pred_density = self.PN_density(inputs)
        next_inputs = tf.concat([xyz, pred_density], axis=-1)
        pred_acc = self.PN_force(next_inputs)
        return pred_acc
