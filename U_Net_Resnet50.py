from keras.applications import resnet50
from keras.models import Model,load_model
from keras.layers import Input, Conv2D, Conv2DTranspose,AveragePooling2D, MaxPooling2D
from keras.layers import UpSampling2D,LeakyReLU, concatenate, Dropout,BatchNormalization,Activation

# Method 1 : Using defined decoder

def finetuned_U_Net_Resnet50():

    input_img_shape = (512, 512, 3)
    RESNET50_encoder = resnet50.RESNET50(include_top=False, weights='imagenet', input_shape=input_img_shape)
    last_feature_extraction_layer = RESNET50_encoder.output

    ## Freezing extraction layers
    set_trainable = False
    for layer in RESNET50_encoder.layers:
        if layer.name in ['block1_conv1']:
            set_trainable = True
        if layer.name in ['block1_pool' ,'block2_pool' ,'block3_pool' ,'block4_pool' ,'block5_pool']:
            layer.trainable = False

    fine_tuned_u_net_model = Conv2DTranspose(256,(2,2),strides=(2, 2))(last_feature_extraction_layer)
    fine_tuned_u_net_model = LeakyReLU(0.1)(fine_tuned_u_net_model)
    fine_tuned_u_net_model = BatchNormalization()(fine_tuned_u_net_model)
    concat_1 = concatenate([fine_tuned_u_net_model,RESNET50_encoder.get_layer("block5_conv3").output])


    fine_tuned_u_net_model = Conv2D(512 ,(2 ,2) ,strides=(1, 1) ,padding='same')(concat_1)
    fine_tuned_u_net_model = LeakyReLU(0.1)(fine_tuned_u_net_model)
    fine_tuned_u_net_model = BatchNormalization()(fine_tuned_u_net_model)
    fine_tuned_u_net_model = Conv2DTranspose(512 ,(3 ,2) ,strides=(2, 2) ,padding='same')(fine_tuned_u_net_model)
    fine_tuned_u_net_model = LeakyReLU(0.1)(fine_tuned_u_net_model)
    fine_tuned_u_net_model = BatchNormalization()(fine_tuned_u_net_model)
    concat_2 = concatenate([fine_tuned_u_net_model ,RESNET50_encoder.get_layer("block4_conv3").output])



    fine_tuned_u_net_model = Conv2D(512 ,(3 ,3) ,strides=(1, 1) ,padding='same')(concat_2)
    fine_tuned_u_net_model = LeakyReLU(0.1)(fine_tuned_u_net_model)
    fine_tuned_u_net_model = BatchNormalization()(fine_tuned_u_net_model)
    fine_tuned_u_net_model = Conv2DTranspose(512 ,(2 ,2) ,strides=(2, 2))(fine_tuned_u_net_model)
    fine_tuned_u_net_model = LeakyReLU(0.1)(fine_tuned_u_net_model)
    fine_tuned_u_net_model = BatchNormalization()(fine_tuned_u_net_model)
    concat_3 = concatenate([fine_tuned_u_net_model,RESNET50_encoder.get_layer("block3_conv3").output])


    fine_tuned_u_net_model = Conv2D(256 ,(3 ,3) ,strides=(1, 1) ,padding='same')(concat_3)
    fine_tuned_u_net_model = LeakyReLU(0.1)(fine_tuned_u_net_model)
    fine_tuned_u_net_model = BatchNormalization()(fine_tuned_u_net_model)
    fine_tuned_u_net_model = Conv2DTranspose(256 ,(3 ,3) ,strides=(2, 2) ,padding='same')(fine_tuned_u_net_model)
    fine_tuned_u_net_model = LeakyReLU(0.1)(fine_tuned_u_net_model)
    fine_tuned_u_net_model = BatchNormalization()(fine_tuned_u_net_model)
    fine_tuned_u_net_model = Conv2DTranspose(256 ,(2 ,2) ,strides=(1, 1) ,padding='same')(fine_tuned_u_net_model)
    fine_tuned_u_net_model = LeakyReLU(0.1)(fine_tuned_u_net_model)
    fine_tuned_u_net_model = BatchNormalization()(fine_tuned_u_net_model)
    concat_4 = concatenate([fine_tuned_u_net_model ,RESNET50_encoder.get_layer("block2_conv2").output])

    fine_tuned_u_net_model = Conv2D(128 ,(3 ,3) ,strides=(1, 1) ,padding='same')(concat_4)
    fine_tuned_u_net_model = LeakyReLU(0.1)(fine_tuned_u_net_model)
    fine_tuned_u_net_model = BatchNormalization()(fine_tuned_u_net_model)
    fine_tuned_u_net_model = Conv2DTranspose(128 ,(3 ,3) ,strides=(2, 2) ,padding='same')(fine_tuned_u_net_model)
    fine_tuned_u_net_model = LeakyReLU(0.1)(fine_tuned_u_net_model)
    fine_tuned_u_net_model = BatchNormalization()(fine_tuned_u_net_model)
    concat_5 = concatenate([fine_tuned_u_net_model ,RESNET50_encoder.get_layer("block1_conv2").output])

    fine_tuned_u_net_model = Conv2D(64 ,(3 ,3) ,strides=(1, 1) ,padding='same')(concat_5)
    fine_tuned_u_net_model = LeakyReLU(0.1)(fine_tuned_u_net_model)
    fine_tuned_u_net_model = BatchNormalization()(fine_tuned_u_net_model)

    fine_tuned_u_net_model = Conv2D(3 ,(3 ,3) ,strides=(1, 1) ,padding='same')(fine_tuned_u_net_model)
    fine_tuned_u_net_model = LeakyReLU(0.1)(fine_tuned_u_net_model)
    fine_tuned_u_net_model = BatchNormalization()(fine_tuned_u_net_model)

    fine_tuned_u_net_model = Model(RESNET50_encoder.input ,fine_tuned_u_net_model)

    return fine_tuned_u_net_model


# Method 2 : Using segmentation model library

# pre_train_encoder = 'resnet50'
# target_class = 4
# activation_function = 'softmax'
# fine_tuned_u_net_model = sm.Unet(pre_train_encoder, encoder_weights='imagenet', classes= target_class,
# activation = activation_function)