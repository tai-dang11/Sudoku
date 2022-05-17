import tensorflow as tf
import tensorflow_datasets as tfds

def preprocess_img(image, label):
    image = tf.image.rgb_to_grayscale(image)  # Convert images to grayscale
    return tf.cast(image, tf.float32) / 255., label

#During training, it's important to shuffle the data well - poorly shuffled data can result in lower training accuracy.
#num_parallel_calls A tf.int64 scalar tf.Tensor, representing the number of batches to compute asynchronously in parallel
#if the value tf.data.AUTOTUNE is used, then the number of parallel calls is set dynamically based on available resources.

#Prefetching overlaps the preprocessing and model execution of a training step.
# While the model is executing training step s, the input pipeline is reading the data for step s+1.
# Doing so reduces the step time to the maximum (as opposed to the sum) of the training and the time it takes to extract the data.

def processImage(img, batch_size, num_shuffle):
    img = img.map(preprocess_img, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    img = img.cache()
    img = img.shuffle(num_shuffle)
    img = img.batch(batch_size)
    img = img.prefetch(tf.data.experimental.AUTOTUNE)
    return img

def shsvnData():
    (ds_train, ds_test, ds_extra), ds_info = tfds.load(
        'svhn_cropped',
        split=['train', 'test', 'extra'],  # split the data load into train, test, extra
        shuffle_files=True,
        as_supervised=True,  # return 2-tuple structure (input,label)
        with_info=True,  # return ds_info with the information about the dataset
    )
    ds_train = ds_train.concatenate(ds_extra)  # Combine all the datasets
    train_shuffle = ds_info.splits['train'].num_examples + ds_info.splits['extra'].num_examples
    test_shufflfe = ds_info.splits['test'].num_examples
    ds_train = processImage(ds_train, 256, train_shuffle//3)
    ds_test = processImage(ds_test,256,test_shufflfe//3)
    return ds_train,ds_test

