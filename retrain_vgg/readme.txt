Run the following four notebooks sequentially:
manual_data_aug.ipynb
image_to_vector.ipynb
retrain_VGG16_data_aug_bottleneck_feature_train.ipynb
fine_tune_vgg.ipynb

Four types of image transformation are applied to each training image sample.
val_acc achieved 0.84818 in the final tuned VGG16 network.