import torchvision

def load_data():
    food101_data =torchvision.datasets.Food101(download=True, root="./", transform=torchvision.transforms.ToTensor())
    print(len(food101_data))
    print(food101_data.classes)
    print(food101_data.class_to_idx)
    print(food101_data[0][0][0])

    pass

load_data()
