'''
Make_Network.py

This function creates a unique UNet based on data dictionary and image input size given

Please give reference to https://github.com/brianmanderson/Easy-Make-UNet if used
'''
__author__ = 'Brian Mark Anderson'
__email__ = 'bmanderson@mdanderson.org'

from Network import my_UNet
layers_dict_conv = {'Layer_0': {'Encoding': [16, 32], 'Decoding': [32, 16]},
                    'Base': {'Encoding': [64]}}

model = my_UNet(layers_dict=layers_dict_conv, filter_vals=(3,3), pool_size=(2,2), activation='relu',batch_normalization=False,
                        padding='same',input_size=(32, 32, 1), final_activation='softmax',final_classes=2).created_model