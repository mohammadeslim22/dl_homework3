from homework.train_classification import train

train(
    exp_dir="logs",
    model_name="classifier",
    num_epoch=50,
    lr=1e-3,

)