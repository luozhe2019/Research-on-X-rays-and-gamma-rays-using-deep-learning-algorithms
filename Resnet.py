import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Sequential

class BasicBlock(layers.Layer):

        def __init__(self,filter_num):
           super(BasicBlock,self).__init__()
           self.conv1 = keras.layers.Conv1D(filters=filter_num, kernel_size=8, padding='same')
           self.bn1 = keras.layers.BatchNormalization()
           self.activity= keras.layers.Activation('selu')

           self.conv2 = keras.layers.Conv1D(filters=filter_num, kernel_size=3, padding='same')
           self.bn2 = keras.layers.BatchNormalization()

           self.shortcut_x = keras.layers.Conv1D(filters=filter_num, kernel_size=1, padding='same')
           self.shortcut_y = keras.layers.BatchNormalization()

        def call(self, inputs, training=None):

            out = self.conv1(inputs)

            out = self.bn1(out, training=training)

            out = self.activity(out)

            out = self.conv2(out)

            out = self.bn2(out, training=training)

            identity1 = self.shortcut_x(inputs)
            identity = self.shortcut_y(identity1)

            output = layers.add([out, identity])

            output = tf.nn.relu(output)

            return output


class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=1):  # [2,2,2,2]
        super(ResNet, self).__init__()

        self.stem = Sequential([layers.Conv1D(64, 1, padding='same'),
                                layers.BatchNormalization(),
                                layers.Activation('selu')
                                ])

        self.layer1 = self.build_resblock(16, layer_dims[0])
        self.layer2 = self.build_resblock(32, layer_dims[1])
        self.layer3 = self.build_resblock(48, layer_dims[2])
        self.layer4 = self.build_resblock(64, layer_dims[3])

        # output : [b,512,h,w],
        self.avgpool = keras.layers.GlobalAveragePooling1D()

        self.fc = keras.layers.Dense(num_classes,activation='sigmoid')

    def call(self, inputs, training=None):
        x = self.stem(inputs, training=training)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)

        # [b,c]
        x = self.avgpool(x)
        # [b,100]
        x = self.fc(x)

        return x

    def build_resblock(self, filter_num, blocks):
        res_blocks = Sequential()
        for _ in range(0, blocks):
            res_blocks.add(BasicBlock(filter_num))

        return res_blocks


def resnet18():
    return ResNet([2, 2, 2, 2])


def resnet34():
    return ResNet([3, 4, 6, 3])


def resnet10():
    return ResNet([1, 1, 1, 1])




