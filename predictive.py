import numpy as np 
import pandas as pd 



from keras.models import model_from_json

# Load the model architecture from JSON
with open("model.json", "r") as json_file:
    loaded_model_json = json_file.read()

# Create an empty model
loaded_model = model_from_json(loaded_model_json)

# Load the model weights
loaded_model.load_weights("model_weights.weights.h5")

import numpy as np 
from keras.preprocessing import image 
test_image = image.load_img('animals.webp', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = loaded_model.predict(test_image)

if result[0][0] == 1:
    prediction = 'DOG'
else:
    prediction = 'CAT'
    
print(f"This is a {prediction}!")

