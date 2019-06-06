'''
Network.py

This function creates a unique UNet based on data dictionary and image input size given

Please give reference to https://github.com/brianmanderson/Easy-Make-UNet if used
'''
__author__ = 'Brian Mark Anderson'
__email__ = 'bmanderson@mdanderson.org'

from keras.layers import Conv2D, Conv3D, Activation, UpSampling2D, UpSampling3D, BatchNormalization, Input, \
    Concatenate, MaxPooling3D, MaxPooling2D
from keras.models import Model


class Unet(object):
    def define_filters(self, filters):
        self.filters = filters

    def define_activation(self, activation):
        self.activation = activation

    def define_pool_size(self, pool_size):
        self.pool_size = pool_size

    def define_padding(self, padding='same'):
        self.padding = padding

    def define_batch_norm(self, batch_norm=True):
        self.batch_norm = batch_norm

    def conv_block3D(self,output_size,x, name, strides=1):
        x = Conv3D(output_size, self.filters, activation=None, padding=self.padding,
                   name=name, strides=strides)(x)
        x = Activation(self.activation)(x)
        if self.batch_norm:
            x = BatchNormalization()(x)
        return x

    def conv_block2D(self,output_size,x, name, strides=1):
        x = Conv2D(output_size, self.filters, activation=None, padding=self.padding,
                   name=name, strides=strides)(x)
        x = Activation(self.activation)(x)
        if self.batch_norm:
            x = BatchNormalization()(x)
        return x

    def max_pool3D(self, x, name=''):
        x = MaxPooling3D(pool_size=self.pool_size, name=name)(x)
        return x

    def max_pool2D(self, x, name=''):
        x = MaxPooling2D(pool_size=self.pool_size, name=name)(x)
        return x

    def up_sample3D(self, x, name=''):
        x = UpSampling3D(size=self.pool_size, name=name)(x)
        return x

    def up_sample2D(self, x, name=''):
        x = UpSampling2D(size=self.pool_size, name=name)(x)
        return x

    def get_unet(self):
        pass


class my_UNet(Unet):

    def __init__(self, layers_dict=None, filter_vals=(3,3,3), pool_size=(2,2,2), activation='relu',padding='same',
                 input_size=(32, 32, 32, 1), final_activation='softmax',final_classes=2, batch_normalization=True):
        self.final_activation = final_activation
        self.final_classes = final_classes
        layers = 0
        self.layers_names = []
        self.is_2D = True
        self.conv_block = self.conv_block2D
        self.max_pool = self.max_pool2D
        self.up_sample = self.up_sample2D
        if len(input_size) == 4:
            self.is_2D = False
            self.conv_block = self.conv_block3D
            self.max_pool = self.max_pool3D
            self.up_sample = self.up_sample3D
        for name in layers_dict:
            if name.find('Base') != 0:
                layers += 1
        for i in range(layers):
            self.layers_names.append('Layer_' + str(i))
        self.layers_names.append('Base')
        self.define_batch_norm(batch_normalization)
        self.define_pool_size(pool_size)
        self.define_filters(filter_vals)
        self.define_activation(activation)
        self.define_padding(padding)
        self.layers_dict = layers_dict
        self.input_size = input_size
        self.get_unet()

    def get_unet(self):
        layers_dict = self.layers_dict
        x = image_input = Input(shape=self.input_size, name='UNet_Input')
        self.layer = 0
        layer_vals = {}
        self.desc = 'Encoder'
        layer_index = 0
        layer_order = []
        for layer in self.layers_names:
            print(layer)
            if layer == 'Base':
                continue
            layer_order.append(layer)
            all_filters = layers_dict[layer]['Encoding']
            for i in range(len(all_filters)):
                strides = 1
                self.desc = layer + '_Encoding_Conv' + str(i)
                x = self.conv_block(all_filters[i], x=x, strides=strides, name=self.desc)
            layer_vals[layer_index] = x
            x = self.max_pool(x,name='Max_Pool_' + layer)
            layer_index += 1
        if 'Base' in layers_dict:
            strides = 1
            all_filters = layers_dict['Base']['Encoding']
            for i in range(len(all_filters)):
                self.desc = 'Base_Conv' + str(i)
                x = self.conv_block(all_filters[i], x, strides=strides, name = self.desc)
        self.desc = 'Decoder'
        self.layer = 0
        layer_order.reverse()
        for layer in layer_order:
            print(layer)
            layer_index -= 1
            all_filters = layers_dict[layer]['Decoding']
            x = self.up_sample(x, name='Upsampling' + str(self.layer) + '_UNet')
            x = Concatenate(name='concat' + str(self.layer) + '_Unet')([x, layer_vals[layer_index]])
            for i in range(len(all_filters)):
                self.desc = layer + '_Decoding_Conv' + str(i)
                x = self.conv_block(all_filters[i], x, self.desc)
            self.layer += 1

        self.activation = self.final_activation
        x = self.conv_block(self.final_classes,x,name='Output')
        model = Model(inputs=image_input, outputs=x)
        self.created_model = model

if __name__ == '__main__':
    xxx = 1