from re import I
import tensorflow as tf
import math
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU'))) 
data = []
for i in range (10):
    data.append(i) 

sine = []
for i in range (10):
    sine.append(math.sin(i))

model = tf.keras.models.load_model("models/model")
prediction = model.predict(data)
print(*sine, sep='\n')
print(prediction)
