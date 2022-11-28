"""App Train."""

import warnings
import os
warnings.simplefilter("ignore")
from input.config.base_config import Config
from models.architecture.architecture_definition import ArchsList
from preprocess.preprocess_definition import PreprocessingList

config = Config()

FAST_RUN = True
EVALUATION = False
BATCH_SIZE = 16


def main(config):

    preprocess = PreprocessingList[config.preprocess].value
    train_generator, validation_generator = preprocess(config).run()

    selected_model = ArchsList[config.selected_model].value
    model, callbacks = selected_model(config).build()

    epochs = 5 if FAST_RUN else 10

    # history = model.fit_generator(
    model.fit_generator(
        generator=train_generator,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=int(validation_generator.n)//BATCH_SIZE,
        steps_per_epoch=int(train_generator.n)//BATCH_SIZE,
        callbacks=callbacks
    )

    config.logger.info('Model Training finished!')

    model.save(os.path.join(
        config.model_save_dev_path,
        'model.h5')
    )

    model.save(os.path.join(
        config.model_save_prd_path,
        'model.h5')
    )

    config.logger.info('Model Saving finished!')


if __name__ == '__main__':
    main(config)

# TODO evaluation module

# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
# ax1.plot(history.history['loss'], color='b', label="Training loss")
# ax1.plot(history.history['val_loss'], color='r', label="validation loss")
# ax1.set_xticks(np.arange(1, epochs, 1))
# ax1.set_yticks(np.arange(0, 1, 0.1))

# ax2.plot(history.history['accuracy'], color='b', label="Training accuracy")
# ax2.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
# ax2.set_xticks(np.arange(1, epochs, 1))

# legend = plt.legend(loc='best', shadow=True)
# plt.tight_layout()
# plt.show()
