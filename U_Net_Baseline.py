import segmentation_model as sm

# Method 1 : Using defined decoder

def baseline_vgg16():

    pre_train_encoder = 'vgg16'
    target_class = 4
    activation_function = 'softmax'
    fine_tuned_u_net_model = sm.Unet(pre_train_encoder, encoder_weights='imagenet', classes= target_class,
    activation = activation_function)

    return fine_tuned_u_net_model
