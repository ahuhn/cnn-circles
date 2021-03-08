from tensorflow.keras.models import load_model

from cifar10_data import get_cifar10_data


def test() -> None:
    input_data = get_cifar10_data()
    model = load_model("trained_models/resnet_squares")
    test_loss, test_acc = model.evaluate(
        input_data.test.images, input_data.test.labels, verbose=2
    )
    print(test_acc)


if __name__ == "__main__":
    test()
