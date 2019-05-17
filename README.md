# Easy-Make-UNet
A perfunctory UNet architecture that should be relatively easy to get yourself started

An example is below for images which are 24x24x1

Concatenations occur between encoding and decoding layers, maintain the nomeclature of Layer_0, Layer_1, etc.
The bottom should be Base

    from Network import my_UNet
    layers_dict_conv = {'Layer_0': {'Encoding': [16, 32], 'Decoding': [32, 16]},
                        'Base': {'Encoding': [64]}}

    model = my_UNet(layers_dict=layers_dict_conv, filter_vals=(3,3), pool_size=(2,2), activation='relu',batch_normalization=False,
                            padding='same',input_size=(24, 24, 1), final_activation='softmax',final_classes=10).created_model
