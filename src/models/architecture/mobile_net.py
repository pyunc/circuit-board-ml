from models.architecture.base_architecture import BaseArchitecture
import tensorflow as tf
from tensorflow import keras
from keras import layers, preprocessing, models, utils, applications, layers
from input.config.base_config import Config

class MobileNet(BaseArchitecture):

    def __init__(self, config: Config):

        super().__init__(config)

    def build(self):
    
        base_model=applications.MobileNet(weights='imagenet',include_top=False)  #imports the mobilenet model and discards the last 1000 neuron layer.

        x=base_model.output
        x=layers.GlobalAveragePooling2D()(x)
        x=layers.Dense(1024,activation='relu')(x)                          #we add dense layers so that the model can learn more complex functions and classify for better results.
        x=layers.Dense(1024,activation='relu')(x)                          #dense layer 2
        x=layers.Dense(512,activation='relu')(x)                           #dense layer 3
        preds=layers.Dense(2,activation='softmax')(x)                      #final layer with softmax activation

        model=models.Model(inputs=base_model.input,outputs=preds)

        # model.summary()

        # utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

        model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

        for layer in model.layers[:20]:
            layer.trainable=False
        for layer in model.layers[20:]:
            layer.trainable=True
        
        earlystop = tf.keras.callbacks.EarlyStopping(patience=10)

        learning_rate_reduction = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.00001)

        callbacks = [earlystop, learning_rate_reduction]

        return model, callbacks

        


