# Retrain VGG
Run the following four notebooks sequentially:
 1. manual_data_aug.ipynb
 1. image_to_vector.ipynb
 1. retrain_VGG16_data_aug_bottleneck_feature_train.ipynb
 1. fine_tune_vgg.ipynb

Four types of image transformation are applied to each training image sample.
val_acc achieved 0.84818 in the final tuned VGG16 network.