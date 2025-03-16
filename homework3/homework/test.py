from homework.train_detection import train

train(
    exp_dir="logs",
    model_name="detector",
    num_epoch=50,
    lr=1e-3,


)