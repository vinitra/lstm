import tensorflow as tf


def build_dataset(
    nb_features=2,
    max_timesteps=10,
    num_samples=20,
):

    X_train = tf.random.normal([num_samples, max_timesteps, nb_features])

    Y_train = tf.random.uniform(
        shape=[num_samples, 1, 1], minval=0, maxval=2, dtype=tf.dtypes.int32
    )

    Y_train = tf.repeat(Y_train, max_timesteps, axis=1)

    Y_incremental = tf.constant([i//5 for i in range(max_timesteps)])

    Y_incremental = (
        2 * tf.reshape(Y_train, [num_samples, max_timesteps]) - 1
    ) * Y_incremental

    X_train = X_train + tf.reshape(
        tf.cast(Y_incremental, tf.float32), [num_samples, max_timesteps, 1]
    )

    return X_train, Y_train
