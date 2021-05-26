import tensorflow as tf
import Datasets.DatasetGenerator as dg


model_path = ".local/models/exp-0/model-0002-[4, 4, 4, 4]/model"
model = tf.keras.models.load_model(model_path)
x, y = dg.getDataset(dg.DataTypes.POLYNOMIAL)
model.evaluate(x=x, y=y)
