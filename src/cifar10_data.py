from typing import Any, List

from attr import dataclass
from tensorflow.keras.datasets.cifar10 import load_data
from tensorflow.keras.utils import to_categorical

CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclass
class ImageDataSet:
    images: List[Any]
    labels: List[Any]


@dataclass
class ImageData:
    train: ImageDataSet
    test: ImageDataSet


def get_cifar10_data() -> ImageData:
    (x_train, y_train), (x_test, y_test) = load_data()
    y_train = to_categorical(y_train, len(CIFAR10_CLASS_NAMES))
    y_test = to_categorical(y_test, len(CIFAR10_CLASS_NAMES))

    return ImageData(
        train=ImageDataSet(x_train, y_train), test=ImageDataSet(x_test, y_test)
    )
