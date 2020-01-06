import os
import sys
import numpy as np

sys.path.insert(0, './')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf

from models.sph_pn import PN_sph
from data.data import PyFleXDataset

tf.random.set_seed(42)


def train_step(optimizer, model, loss_object, acc_object, train_pts, train_labels):
    with tf.GradientTape() as tape:
        pred = model(train_pts)
        loss = loss_object(train_labels, pred)
        acc = acc_object(train_labels, pred)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return loss, acc


def test_step(model, loss_object, acc_object, test_pts, test_labels):
    pred = model(test_pts)
    loss = loss_object(test_labels, pred)
    acc = acc_object(test_labels, pred)

    return loss, acc


def train(config, params):
    model = PN_sph()
    model.build(input_shape=(params['batch_size'], params['num_points'], 6))
    print(model.summary())
    print('[info] model training...')

    optimizer = tf.keras.optimizers.Adam(lr=params['lr'])
    loss_object = tf.keras.losses.MeanSquaredError()
    acc_object = tf.keras.metrics.MeanSquaredError()

    data_dir = '/home/abslon/Workspace/pointnet2-tf2/data'  # path to data folder ex) ~/Workspace/data
    env = 'DamBreak'
    train_dataset = PyFleXDataset('train', env, data_dir).prefetch(tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.batch(params['batch_size'])
    valid_dataset = PyFleXDataset('valid', env, data_dir).batch(params['batch_size'])

    train_summary_writer = tf.summary.create_file_writer(
        os.path.join(config['log_dir'], config['log_code'])
    )

    with train_summary_writer.as_default():
        for epoch in range(5):
            for i, data in enumerate(train_dataset):
                train_xyz, train_feat, label_xyz, label_feat = data

                train_pts = tf.concat([train_xyz, train_feat], axis=-1)
                loss, acc = train_step(
                    optimizer,
                    model,
                    loss_object,
                    acc_object,
                    train_pts,
                    label_feat  # velocity
                )

                if optimizer.iterations % config['log_freq'] == 0:
                    tf.summary.scalar('train loss', np.sqrt(loss), step=optimizer.iterations)
                    tf.summary.scalar('train accuracy', acc, step=optimizer.iterations)
                    print('epoch {0} [{1}] train loss: '.format(epoch, i), np.sqrt(loss.numpy()))

            for data in valid_dataset:
                test_xyz, test_feat, test_label_xyz, test_label_feat = data
                test_pts = tf.concat([test_xyz, test_feat], axis=-1)
                test_loss, test_acc = test_step(
                    model,
                    loss_object,
                    acc_object,
                    test_pts,
                    test_label_feat  # velocity
                )

                if optimizer.iterations % config['log_freq'] == 0:
                    tf.summary.scalar('test loss', np.sqrt(test_loss), step=optimizer.iterations)
                    tf.summary.scalar('test accuracy', test_acc, step=optimizer.iterations)
                    print('epoch {0} [{1}] test loss: '.format(epoch, i), np.sqrt(test_loss))
        model.save('sph_model_epoch' + str(epoch) + '.h5')

    print("training done.")
    model.save('sph_model.h5')


if __name__ == '__main__':
    config = {
        'dataset_dir': 'data/scannet',
        'log_dir': 'logs',
        'log_code': 'sph',
        'log_freq': 50,
        'test_freq': 100
    }

    params = {
        'batch_size': 2,
        'num_points': 3456,
        'num_classes': 21,
        'lr': 0.001,
        'bn': False
    }

    train(config, params)
