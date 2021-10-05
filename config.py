import os


class Config:
    DATA_DIR = 'data'
    CSV_PATH = os.path.join(DATA_DIR, 'train_clean.csv')
    train_batch_size = 10
    val_batch_size = 10
    num_workers = 8
    image_size = 512
    output_dim = 512
    hidden_dim = 1024
    input_dim = 3
    epochs = 35
    lr = 1e-4
    num_of_classes = 88313
    pretrained = True
    model_name = 'resnet101'
    seed = 42
