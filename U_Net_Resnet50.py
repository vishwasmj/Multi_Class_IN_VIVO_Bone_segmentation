from keras.applications import resnet50
from keras.models import Model,load_model
from keras.layers import Input, Conv2D, Conv2DTranspose,AveragePooling2D, MaxPooling2D
from keras.layers import UpSampling2D,LeakyReLU, concatenate, Dropout,BatchNormalization,Activation

# Method 1 : Using defined decoder

def finetuned_U_Net_Resnet50():

    input_img_shape = (512, 512, 3)
    RESNET50_encoder = resnet50.RESNET50(include_top=False, weights='imagenet', input_shape=input_img_shape)
    last_feature_extraction_layer = RESNET50_encoder.output
    
    activation = 'softmax'

    ## Freezing extraction layers
    set_trainable = False
     if block_type == 'transpose':
        up_block = Conv2DTranspose
    else:
        up_block = UpSampling2D

    # Skip connects for resnet 50
    skip_layers = list(112, 56, 42, 9)

    for i in range(n_upsample_blocks):

        # check for skip connections
        if i < len(skip_layers):
            skip = RESNET34_encoder.layers[skip_layers[i]].output
        else:
            skip = None
        up_size = (upsample_rates[i], upsample_rates[i])
       filters = last_block_filters * 2 ** (n_upsample_blocks - (i + 1))

    ly = up_block(filters, i, upsample_rate=up_size, skip=skip, **kwargs)(ly)
        
    ly = Conv2D(4, (3, 3), padding='same', name='final_conv')(ly)
    ly = Activation(activation, name=activation)(ly)

    fine_tuned_u_net_model = Model(input, ly)


    return fine_tuned_u_net_model


# Method 2 : Using segmentation model library

# pre_train_encoder = 'resnet50'
# target_class = 4
# activation_function = 'softmax'
# fine_tuned_u_net_model = sm.Unet(pre_train_encoder, encoder_weights='imagenet', classes= target_class,
# activation = activation_function)
