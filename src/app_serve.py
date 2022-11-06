import cv2
# load last model

# os.path.isdir("/home/paulo/Documents/project-practical-mlops/circuit-board-ml/notebooks/models")
model = tf.keras.models.load_model("/home/paulo/Documents/project-practical-mlops/circuit-board-ml/notebooks/models/model.h5")
# model.save_weights("model.h5")

# adjust input



img = cv2.imread(r'/home/paulo/Documents/project-practical-mlops/circuit-board-ml/notebooks/organized_data/92000108_test.jpg')
img = img[...,::-1]                                                                 #give right image or else it will show error

# predict
model.predict(img[None,...], batch_size=None, verbose=2, steps=1)

