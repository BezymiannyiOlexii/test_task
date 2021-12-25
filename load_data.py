import tensorflow as tf
import zipfile

AUTOTUNE = tf.data.experimental.AUTOTUNE

def zip_calltech_to_dataset():
    with zipfile.ZipFile('caltech256.zip', 'r') as zipref:
        zipref.extractall()
    tf_dataset = tf.data.Dataset.list_files('./256_ObjectCategories/*/*')
    return tf_dataset

def load_mnist():
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train, _), _ = fashion_mnist.load_data()
    train = train.reshape(60000, 28, 28, 1)
    tf_dataset = tf.data.Dataset.from_tensor_slices(train)
    return tf_dataset

def preprocess_calltech_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    return image

def preprocess_fmnist_image(image):
    image = tf.image.resize(image, [224, 224])
    image = tf.image.grayscale_to_rgb(image)
    return image


def make_dataloader(tf_dataset, prepocessing, batch_size=16):
    data = tf_dataset.map(prepocessing, num_parallel_calls=AUTOTUNE).batch(batch_size)
    return data


if __name__ == "__main__":
    pass
