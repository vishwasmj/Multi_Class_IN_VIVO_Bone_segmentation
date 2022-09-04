import tensorflow as tf
import tensorflow.keras as keras
import segmentation_models as sm

from keras.utils.np_utils import normalize
from matplotlib import pyplot as plt
from patchify import patchify
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical
from keras.models import load_model

import shutil
import numpy as np
import tifffile as tiff
import random
import os
import pandas as pd
import glob
import cv2


## Created configuration file

from configuration_img_segmentation import *
from U_Net_VGG16 import finetuned_U_Net_VGG16
from U_Net_Resnet34 import finetuned_U_Net_Resnet34
from U_Net_InceptionV3 import finetuned_U_Net_InceptionV3
from U_Net_VGG19 import finetuned_U_Net_VGG19
from U_Net_Resnet50 import finetuned_U_Net_Resnet50
from U_Net_Seresnet50 import finetuned_U_Net_SEResnet50
from U_Net_Baseline import baseline_vgg16


# Importing data

## Connecting to google drive
from google.colab import drive
drive.mount('/content/gdrive')

bone_images = tiff.imread(r"/content/gdrive/MyDrive/Annotated_bone_images.tif")
corresponding_bone_mask = tiff.imread(r"/content/gdrive/MyDrive/Annotated_masks.tif")

# Data Pre-processing

## Step 1 : Image Resize

### Creating patches from bone images and their corresponding masks
### Function to create patches from images and masks
def create_image_patch(img_to_be_patched):

    ## append patches to a list
    patches_of_images = []

    # Creating patches of each image
    for img in range(img_to_be_patched.shape[0]):
        original_image = img_to_be_patched[img]

        # Creating patches of dim 512 X 512 with no overlap
        image_patch = patchify(original_image, (512, 512), step=512)

        ## Appending each patch to a list
        for i in range(image_patch.shape[0]):
            for j in range(image_patch.shape[1]):
                one_img_patch = image_patch[i, j, :, :]

                patches_of_images.append(one_img_patch)

    return patches_of_images

### Creating patches of images and masks
bone_image_patches = create_image_patch(bone_images)
bone_mask_patches = create_image_patch(corresponding_bone_mask)


###Converting images to an array

train_bone_images = np.array(bone_image_patches)
train_bone_masks = np.array(bone_mask_patches)

## Step 2: Expanding Dimensions

### If input train images are grayscale images then convert them to RGB images
### U-Net model only accepts RGB images

#train_bone_images = np.stack((train_bone_images,)*3, axis=-1)

### Expanding dimensions of the mask images to match the train images
train_bone_masks = np.expand_dims(train_bone_masks, axis=3)


## Step 3: Creating Train-Test split


### Creating training and testing sets from images and masks
X_train_with_validation, X_test, y_train_with_validation, y_test = train_test_split(train_bone_images,
                                                                                    train_bone_masks,
                                                                                    test_size = 0.10)
###Further split training data t a smaller subset for quick testing of models
X_train, X_validation, y_train, y_validation = train_test_split(X_train_with_validation,
                                                                y_train_with_validation,
                                                                test_size = 0.2)


## Sanity check : Checking if the training images and corresponding masks are alligned properly

def sanity_check(x_set,y_set,img_num):
    plt.figure(figsize=(16, 8))
    plt.subplot(112)
    plt.imshow(x_set[img_num, :,:, 0], cmap='gray')
    plt.subplot(112)
    plt.imshow(np.reshape(y_set[img_num], (512, 512)), cmap='gray')
    plt.show()


random_train_image_num = random.randint(0, len(X_train))

### plotting random images and masks
sanity_check(X_train,y_train,random_train_image_num)


## Step 4 : Label Encoding Masks in training and testing sets

### Function to perform categorical encoding on masks
def label_encode_masks(mask_set, total_target_classes):
    categorical_mask_set = to_categorical(mask_set, num_classes = total_target_classes)
    label_encoded_mask = categorical_mask_set.reshape((mask_set.shape[0], mask_set.shape[1], mask_set.shape[2],
                                                       total_target_classes))
    return label_encoded_mask

## Label Encoding masks
y_train_label_enc = label_encode_masks(y_train, 4)
y_test_label_enc = label_encode_masks(y_test, 4)



## Step 5 : Creating a Generator instance

seed = 27

def data_augmentation_bone_img_mask(data_set):

    data_gen_instance = dict(width_shift_range=0.25,
                     height_shift_range=0.25,
                     shear_range=0.5,
                     rotation_range=90,
                     zoom_range=0.1,
                     horizontal_flip=True,
                     vertical_flip=True,
                     fill_mode='reflect')

    data_gen = ImageDataGenerator(**data_gen_instance)
    data_gen.fit(data_set, augment=True, seed=seed)
    augmented_images = data_gen.flow(data_set, seed=seed)

    return augmented_images

### fitting the train and test bone images as well as masks with the data generator instance

augmented_X_train = data_augmentation_bone_img_mask(X_train)
augmented_y_train = data_augmentation_bone_img_mask(y_train_label_enc)

augmented_X_test = data_augmentation_bone_img_mask(X_test)
augmented_y_test = data_augmentation_bone_img_mask(y_test_label_enc)



## Initialising model parameters - Reused for all the U-Net models

target_classes = var_n_class
selected_act_function = var_act_func
learning_rate = var_learning_rate
selected_optimizer = keras.optimizers.Adam(learning_rate)


## Focal-Dice Loss function creation

### Assigning importance for each class, for better model training

dice_loss = sm.losses.DiceLoss(class_weights=np.array([var_class_weight_1, var_class_weight_2, var_class_weight_3,
                                                       var_class_weight_4]))
focal_loss = sm.losses.CategoricalFocalLoss()

### Equal importance given to focal loss and dice loss

# focal_dice_loss = dice_loss + (2 * focal_loss)
focal_dice_loss = dice_loss +  focal_loss

### Segmentation metrics - IOU initialization
segmentation_metrics = [sm.metrics.IOUScore(threshold=0.55), sm.metrics.FScore(threshold=0.55)]


## U - Net Model 1  : Fine tuned VGG16

def fine_tuned_model():
    return finetuned_U_Net_VGG16()

vgg16_model = fine_tuned_model()
vgg16_model.compile(selected_optimizer, focal_dice_loss, metrics=segmentation_metrics)
print(vgg16_model.summary())

## pre-processing inputs according to the encoder
ip_pre_process_vgg16 = sm.get_preprocessing('vgg16')
X_train_vgg16 = ip_pre_process_vgg16(augmented_X_train)
X_test_vgg16 = ip_pre_process_vgg16(augmented_X_test)

history_vgg16 = vgg16_model.fit(X_train_vgg16,
          augmented_y_train,
          batch_size=8,
          epochs=50,
          verbose=1,
          validation_data=(X_test_vgg16, augmented_y_test))

vgg16_model.save('u_net_vgg16.hdf5')

## U - Net Model 2  : Fine tuned Resnet34

def fine_tuned_model():
    return finetuned_U_Net_Resnet34()

resnet34_model = fine_tuned_model()
resnet34_model.compile(selected_optimizer, focal_dice_loss, metrics=segmentation_metrics)
print(resnet34_model.summary())


## pre-processing inputs according to the encoder
ip_pre_process_resnet34 = sm.get_preprocessing('resnet34')
X_train_resnet34 = ip_pre_process_resnet34(augmented_X_train)
X_test_resnet34 = ip_pre_process_resnet34(augmented_X_test)

history_resnet34 = resnet34_model.fit(X_train_resnet34,
          augmented_y_train,
          batch_size=8,
          epochs=50,
          verbose=1,
          validation_data=(X_test_resnet34, augmented_y_test))

resnet34_model.save('u_net_resnet34.hdf5')

## U - Net Model 3  : Fine tuned InceptionV3

def fine_tuned_model():
    return finetuned_U_Net_InceptionV3()

inceptionv3_model = fine_tuned_model()
inceptionv3_model.compile(selected_optimizer, focal_dice_loss, metrics=segmentation_metrics)
print(inceptionv3_model.summary())

## pre-processing inputs according to the encoder
ip_pre_process_inception = sm.get_preprocessing('inceptionv3')
X_train_inception = ip_pre_process_inception(augmented_X_train)
X_test_inception = ip_pre_process_inception(augmented_X_test)

history_inceptionv3 = inceptionv3_model.fit(X_train_inception,
          augmented_y_train,
          batch_size=8,
          epochs=50,
          verbose=1,
          validation_data=(X_test_inception, augmented_y_test))

inceptionv3_model.save('u_net_inceptionv3.hdf5')


## U - Net Model 4  : Fine tuned VGG19

def fine_tuned_model():
    return finetuned_U_Net_VGG19()

vgg19_model = fine_tuned_model()
vgg19_model.compile(selected_optimizer, focal_dice_loss, metrics=segmentation_metrics)
print(vgg19_model.summary())

## pre-processing inputs according to the encoder
ip_pre_process_vgg19 = sm.get_preprocessing('vgg19')
X_train_vgg19 = ip_pre_process_vgg19(augmented_X_train)
X_test_vgg19 = ip_pre_process_vgg19(augmented_X_test)

history_vgg19 = vgg19_model.fit(X_train_vgg19,
          augmented_y_train,
          batch_size=8,
          epochs=50,
          verbose=1,
          validation_data=(X_test_vgg19, augmented_y_test))

vgg19_model.save('u_net_vgg19.hdf5')

## U - Net Model 5  : Fine tuned Resnet50

def fine_tuned_model():
    return finetuned_U_Net_Resnet50()

resnet50_model = fine_tuned_model()
resnet50_model.compile(selected_optimizer, focal_dice_loss, metrics=segmentation_metrics)
print(resnet50_model.summary())

## pre-processing inputs according to the encoder

ip_pre_process_resnet50 = sm.get_preprocessing('resnet50')
X_train_resnet50 = ip_pre_process_resnet50(augmented_X_train)
X_test_resnet50 = ip_pre_process_resnet50(augmented_X_test)

history_resnet50 = resnet50_model.fit(X_train_resnet50,
          augmented_y_train,
          batch_size=8,
          epochs=50,
          verbose=1,
          validation_data=(X_test_resnet50, augmented_y_test))

resnet50_model.save('u_net_resnet50.hdf5')



## U - Net Model 6  : Fine tuned SE-Resnet50
def fine_tuned_model():
    return finetuned_U_Net_SEResnet50()

seresnet50_model = fine_tuned_model()
seresnet50_model.compile(selected_optimizer, focal_dice_loss, metrics=segmentation_metrics)
print(seresnet50_model.summary())
## pre-processing inputs according to the encoder
ip_pre_process_seresnet50 = sm.get_preprocessing('seresnet50')
X_train_seresnet50 = ip_pre_process_seresnet50(augmented_X_train)
X_test_seresnet50 = ip_pre_process_seresnet50(augmented_X_test)

history_seresnet50 = seresnet50_model.fit(X_train_seresnet50,
          augmented_y_train,
          batch_size=8,
          epochs=50,
          verbose=1,
          validation_data=(X_test_seresnet50, augmented_y_test))

seresnet50_model.save('u_net_seresnet50.hdf5')


## U - Net Model 7  : Baseline U-net VGG16
def baseline_vgg_model():
    return baseline_vgg16()

baseline_model = baseline_vgg_model()
baseline_model.compile(selected_optimizer, focal_dice_loss, metrics=segmentation_metrics)
print(baseline_model.summary())
## pre-processing inputs according to the encoder

X_train_baseline = ip_pre_process_vgg16(augmented_X_train)
X_test_baseline = ip_pre_process_vgg16(augmented_X_test)

history_baseline = baseline_model.fit(X_train_baseline,
          augmented_y_train,
          batch_size=8,
          epochs=50,
          verbose=1,
          validation_data=(X_test_baseline, augmented_y_test))

baseline_model.save('u_net_baseline.hdf5')


## Plotting training and validation loss

def plot_train_val_loss(history_of_models):
    loss = history_of_models.history['loss']
    val_loss = history_of_models.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


## Plotting training and validation IOU scores

def plot_train_val_iou(history_of_models):
    loss = history_of_models.history['loss']
    iou = history_of_models.history['iou_score']
    val_iou = history_of_models.history['val_iou_score']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, iou, 'y', label='Training IOU')
    plt.plot(epochs, val_iou, 'r', label='Validation IOU')
    plt.title('Training and validation IOU- SE-Resnet50')
    plt.xlabel('Epochs')
    plt.ylabel('IOU')
    plt.legend()
    plt.show()

## Plotting loss and IOU metrics for U-Net seresnet50
plot_train_val_loss(history_seresnet50)
plot_train_val_iou(history_seresnet50)


## Saving models to google drive

shutil.copy('/content/u_net_vgg16.hdf5','/content/gdrive/MyDrive/Models')
shutil.copy('/content/u_net_resnet34.hdf5','/content/gdrive/MyDrive/Models')
shutil.copy('/content/u_net_inceptionv3.hdf5','/content/gdrive/MyDrive/Models')
shutil.copy('/content/u_net_vgg19.hdf5','/content/gdrive/MyDrive/Models')
shutil.copy('/content/u_net_resnet50.hdf5','/content/gdrive/MyDrive/Models')
shutil.copy('/content/u_net_seresnet50.hdf5','/content/gdrive/MyDrive/Models')

## Loading saved models from google drive

u_net_model1 = load_model('/content/gdrive/MyDrive/Models/u_net_vgg16.hdf5', compile=False)
u_net_model2 = load_model('/content/gdrive/MyDrive/Models/u_net_resnet34.hdf5', compile=False)
u_net_model3 = load_model('/content/gdrive/MyDrive/Models/u_net_inceptionv3.hdf5', compile=False)
u_net_model4 = load_model('/content/gdrive/MyDrive/Models/u_net_vgg19.hdf5', compile=False)
u_net_model5 = load_model('/content/gdrive/MyDrive/Models/u_net_resnet50.hdf5', compile=False)
u_net_model6 = load_model('/content/gdrive/MyDrive/Models/u_net_seresnet50.hdf5', compile=False)


## Predictions of models on test data

u_net_vgg16_pred = u_net_model1.predict(X_test_vgg16)
u_net_resnet34_pred = u_net_model2.predict(X_test_resnet34)
u_net_inceptionv3_pred = u_net_model3.predict(X_test_inception)
u_net_vgg19_pred = u_net_model4.predict(X_test_vgg19)
u_net_resnet50_pred = u_net_model5.predict(X_test_resnet50)
u_net_seresnet50_pred = u_net_model6.predict(X_test_seresnet50)


## Ensemble predictions

def class_iou_predictions(IOU_score, target_classes):

    class_val = np.array(IOU_score.get_weights()).reshape(target_classes, target_classes)

    class_1_IOU_score = class_val[0, 0] / (
                class_val[0, 0] + class_val[0, 1] + class_val[0, 2] + class_val[0, 3] + class_val[1, 0] +
                class_val[2, 0] + class_val[3, 0])

    class_2_IOU_score = class_val[1, 1] / (
                class_val[1, 1] + class_val[1, 0] + class_val[1, 0] + class_val[1, 2] + class_val[0, 1] +
                class_val[2, 1] + class_val[3, 1])

    class_3_IOU_score = class_val[2, 2] / (
                class_val[2, 2] + class_val[2, 0] + class_val[2, 1] + class_val[2, 3] + class_val[0, 2] +
                class_val[1, 2] + class_val[3, 2])

    class_4_IOU_score = class_val[3, 3] / (
                class_val[3, 3] + class_val[3, 0] + class_val[3, 1] + class_val[3, 2] + class_val[0, 3] +
                class_val[1, 3] + class_val[2, 3])

    print("U-net models accuracy for predicting class 1 is: ", class_1_IOU_score)
    print("U-net models accuracy for predicting class 2 is: ", class_2_IOU_score)
    print("U-net models accuracy for predicting class 3 is: ", class_3_IOU_score)
    print("U-net models accuracy for predicting class 4 is: ", class_4_IOU_score)

def ensemble_network(u_net_model_predictions_i,u_net_model_predictions_j,u_net_model_predictions_k,augmented_y_test):

    u_net_model_predictions = np.array([u_net_model_predictions_i, u_net_model_predictions_j,
                                        u_net_model_predictions_k])


    # Assigning random weights to models predictions
    random_model_weights = [0.3, 0.5, 0.2]

    # sum of the products of all elements with random weights
    sum_of_weighted_model_predictions = np.tensordot(u_net_model_predictions, random_model_weights, axes=((0), (0)))
    weighted_ensemble_U_net_predictions = np.argmax(sum_of_weighted_model_predictions, axis=3)

    # Using argmax to find the highest score given by each u-net models

    y_pred_1 = np.argmax(u_net_model_predictions_i, axis=3)
    y_pred_2 = np.argmax(u_net_model_predictions_j, axis=3)
    y_pred_3 = np.argmax(u_net_model_predictions_k, axis=3)

    target_classes = var_n_class

    IOU_score_model_1 = tf.keras.metrics.MeanIoU(num_classes=target_classes)
    IOU_score_model_2 = tf.keras.metrics.MeanIoU(num_classes=target_classes)
    IOU_score_model_3 = tf.keras.metrics.MeanIoU(num_classes=target_classes)
    IOU_weighted_score = tf.keras.metrics.MeanIoU(num_classes=target_classes)

    IOU_score_model_1.update_state(augmented_y_test[:, :, :, 0], y_pred_1)
    IOU_score_model_2.update_state(augmented_y_test[:, :, :, 0], y_pred_2)
    IOU_score_model_3.update_state(augmented_y_test[:, :, :, 0], y_pred_3)
    IOU_weighted_score.update_state(augmented_y_test[:, :, :, 0], weighted_ensemble_U_net_predictions)

    accuracy_for_each_target_class_model_1 = class_iou_predictions(IOU_score_model_1,target_classes)
    accuracy_for_each_target_class_model_2 = class_iou_predictions(IOU_score_model_2, target_classes)
    accuracy_for_each_target_class_model_3 = class_iou_predictions(IOU_score_model_3, target_classes)
    accuracy_for_each_target_class_weighted_model = class_iou_predictions(IOU_weighted_score, target_classes)

    ### Grid search for finding best weights for the models

    best_weights_df = pd.DataFrame([])

    for weight_1 in range(0, 4):
        for weight_2 in range(0, 4):
            for weight_3 in range(0, 4):
                all_weights = [weight_1 / 10., weight_2 / 10., weight_3 / 10.]

                weighted_IOU = tf.keras.metrics.MeanIoU(num_classes=target_classes)

                weighted_predictions = np.tensordot(u_net_model_predictions, all_weights, axes=((0), (0)))

                weighted_ensemble_predictions = np.argmax(weighted_predictions, axis=3)

                weighted_IOU.update_state(augmented_y_test[:, :, :, 0], weighted_ensemble_predictions)

                best_weights_df = best_weights_df.append(pd.DataFrame({'weight_1': all_weights[0],
                                                                       'weight_2': all_weights[1],
                                                                       'weight_3': all_weights[2],
                                                                       'IOU_score_obtained':
                                                                           weighted_IOU.result().numpy()},
                                                                      index=[0]), ignore_index = True)

    best_weights_for_max_iou = best_weights_df.iloc[best_weights_df['IOU_score_obtained'].idxmax()]
    best_found_weights_grid_search = [best_weights_for_max_iou[0], best_weights_for_max_iou[1],
                                      best_weights_for_max_iou[2]]


    best_weighted_predictions = np.tensordot(u_net_model_predictions, best_found_weights_grid_search, axes=((0), (0)))
    ensemble_predictions = np.argmax(best_weighted_predictions, axis=3)

    return ensemble_predictions, best_found_weights_grid_search


u_net_ensmble_network_1, ensmble_unet_network_1_weights = ensemble_network(u_net_vgg16_pred,u_net_resnet34_pred
                                                                           ,u_net_inceptionv3_pred,augmented_y_test)
u_net_ensmble_network_2,ensmble_unet_network_2_weights = ensemble_network(u_net_vgg19_pred,u_net_resnet50_pred,
                                                                          u_net_seresnet50_pred,augmented_y_test)


## Ensemble Network 1 predictions on Unseen test images
def unknown_test_img_prediction_network_1(test_img,annoated_test_img,ensmble_unet_network_1_weights):


    test_img_norm = test_img[:, :, :]
    test_img_input = np.expand_dims(test_img_norm, 0)


    model_1_ip_test_img = ip_pre_process_vgg16(test_img_input)
    model_2_ip_test_img = ip_pre_process_resnet34(test_img_input)
    model_3_ip_test_img = ip_pre_process_inception(test_img_input)

    u_net_vgg16 = u_net_model1.predict(model_1_ip_test_img)
    u_net_resnet34 = u_net_model2.predict(model_2_ip_test_img)
    u_net_inceptionv3 = u_net_model3.predict(model_3_ip_test_img)

    test_image_predictions = np.array([u_net_vgg16, u_net_resnet34, u_net_inceptionv3])

    weighted_test_img_predictions = np.tensordot(test_image_predictions, ensmble_unet_network_1_weights,
                                                 axes=((0), (0)))
    ensemble_test_img_predictions = np.argmax(weighted_test_img_predictions, axis=3)[0, :, :]

    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Test Image')
    plt.imshow(test_img[:, :, 0], cmap='gray')
    plt.subplot(232)
    plt.title('Test Label')
    plt.imshow(annoated_test_img[:, :, 0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image - Ensemble Network 1')
    plt.imshow(ensemble_test_img_predictions, cmap='jet')
    plt.show()

## Ensemble Network 2 predictions on unseen test images
def unknown_test_img_prediction_network_2(test_img,annoated_test_img,ensmble_unet_network_2_weights):


    test_img_norm = test_img[:, :, :]
    test_img_input = np.expand_dims(test_img_norm, 0)


    model_1_ip_test_img = ip_pre_process_vgg19(test_img_input)
    model_2_ip_test_img = ip_pre_process_resnet50(test_img_input)
    model_3_ip_test_img = ip_pre_process_seresnet50(test_img_input)

    u_net_vgg19 = u_net_model1.predict(model_1_ip_test_img)
    u_net_resnet50 = u_net_model2.predict(model_2_ip_test_img)
    u_net_seresnet50 = u_net_model3.predict(model_3_ip_test_img)

    test_image_predictions = np.array([u_net_vgg19, u_net_resnet50, u_net_seresnet50])

    weighted_test_img_predictions = np.tensordot(test_image_predictions, ensmble_unet_network_1_weights,
                                                 axes=((0), (0)))
    ensemble_test_img_predictions = np.argmax(weighted_test_img_predictions, axis=3)[0, :, :]

    plt.figure(figsize=(12, 8))
    plt.subplot(231)
    plt.title('Test Image')
    plt.imshow(test_img[:, :, 0], cmap='gray')
    plt.subplot(232)
    plt.title('Test Label')
    plt.imshow(annoated_test_img[:, :, 0], cmap='jet')
    plt.subplot(233)
    plt.title('Prediction on test image - Ensemble network 2')
    plt.imshow(ensemble_test_img_predictions, cmap='jet')
    plt.show()

test_img_number = random.randint(0, len(augmented_X_test))
test_img = X_test[test_img_number]
annoated_test_img=y_test[test_img_number]

unknown_test_img_prediction_network_1(test_img,annoated_test_img,ensmble_unet_network_1_weights)
unknown_test_img_prediction_network_2(test_img,annoated_test_img,ensmble_unet_network_2_weights)