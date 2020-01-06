import tensorflow as tf
import h5py
import numpy as np
import time
import random
from pathlib import Path
import glob


def read_h5(data_names, path):
    hf = h5py.File(path, 'r')
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


class PyFleXDataset(tf.data.Dataset):
    def _generator(phase, env, data_dir):
        data_names = ['positions', 'velocities']
        phase = phase.decode("utf-8")
        env = env.decode("utf-8")
        data_dir = data_dir.decode("utf-8")
        data_phase_dir = Path(data_dir) / ('data_' + env) / phase

        rollout_list = glob.glob(str(data_phase_dir / '*'))
        #rollout_list = [str(data_phase_dir/'1'), str(data_phase_dir/'2'), str(data_phase_dir/'3')]
        random.shuffle(rollout_list)
        # Opening the file
        for rollout in rollout_list:
            data_path_list = glob.glob(str(Path(rollout) / '*.h5'))
            random.shuffle(data_path_list)

            for data_path in data_path_list:
                data_path_split = data_path.split('/')
                data_name = data_path_split[-1]
                data_num = [int(s) for s in data_name.split('.') if s.isdigit()]
                if data_num[0] == (len(data_path_list) - 1):
                    continue
                data_path_split[-1] = str(data_num[0] + 1) + '.h5'
                next_data_path = '/'.join(data_path_split)

                data = read_h5(data_names, data_path)
                data_next = read_h5(data_names, next_data_path)

                # print(data_path)

                # position, velocity, next position, next velocity
                yield (data[0], data[1], data_next[0], data_next[1])

    def __new__(cls, phase, env, data_dir):
        return tf.data.Dataset.from_generator(
            cls._generator,
            output_types=(tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32, tf.dtypes.float32),
            args=(phase, env, data_dir)
        )


if __name__ == '__main__':
    def benchmark(dataset, num_epochs=2):
        start_time = time.perf_counter()
        for epoch_num in range(num_epochs):
            print("epoch: ", epoch_num)
            for data in dataset:
                # Performing a training step
                print("shape : ", data[0].shape)
        tf.print("Execution time:", time.perf_counter() - start_time)

    phase = 'train'
    data_dir = '/home/abslon/Workspace/pointnet2-tf2/data'  # path to data folder ex) ~/Workspace/data
    env = 'DamBreak'
    dataset = PyFleXDataset(phase, env, data_dir).batch(30)

    benchmark(dataset)