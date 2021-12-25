from keras.applications.vgg16 import VGG16
from keras.models import Model
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from load_data import *


def make_predict(data, dataset_name, number_train_samples=3000):
    if dataset_name == "fmnist":
        n_clusters = 10
    elif dataset_name == "caltech":
        n_clusters = 256
    else:
        raise NameError('Incorrect name of dataset')

    count_of_batches = number_train_samples // 16 + 1

    #features from CNN
    model = VGG16(weights='imagenet', input_shape=(224, 224, 3))
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    features = model.predict(data.take(count_of_batches).prefetch(1), verbose=1)

    #reduce number of features
    pca = PCA(n_components=120, random_state=1)
    reduced_features = pca.fit_transform(features)

    #classifier features
    aggl = AgglomerativeClustering(n_clusters=n_clusters).fit(reduced_features)
    labels = aggl.labels_

    return labels


if __name__ == "__main__":
    dataset = load_mnist()
    preprocess_data = make_dataloader(dataset, preprocess_fmnist_image)
    fmnist_labels = make_predict(preprocess_data, dataset_name="fmnist")
    print(f"caltech_labels for {len(fmnist_labels)} images: {fmnist_labels}")

    print("##################################")

    dataset = zip_calltech_to_dataset()
    preprocess_data = make_dataloader(dataset, preprocess_calltech_image)
    caltech_labels = make_predict(preprocess_data, dataset_name="caltech")
    print(f"caltech_labels for {len(caltech_labels)} images: {caltech_labels}")
