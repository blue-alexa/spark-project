import cv2
import numpy as np
from keras.models import model_from_yaml

# load fine tuned vgg model with data augmentation
model_path = "fine_tune_VGG_model_aug.yaml"
weights_path = "vgg16_weights_aug.h5"

# load YAML and create model
yaml_file = open(model_path, 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)

# load weights into new model
loaded_model.load_weights(weights_path)
print("Loaded model from disk")

loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

CLASS_LABELS = ['apple','banana','nectarine','plum','peach','watermelon','pear','mango','grape',
                'orange','strawberry','pineapple','radish','carrot','potato','tomato','bellpepper',
                'broccoli','cabbage','cauliflower','celery','eggplant','garlic','spinach','ginger']

image_size = 150

def predict_single_image(model, image, class_labels, image_size):   
    image = cv2.resize(image, (image_size, image_size))      
    image = (image / 255.0) * 2.0 - 1.0
    image = np.expand_dims(image, axis=0)
    pred = model.predict(image)
    class_label = class_labels[np.argmax(pred)]
    return class_label


cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 640)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Display the resulting frame
    cv2.putText(gray, "Press s to start, press q to quit!", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cv2.waitKey(1) & 0xFF == ord('s'):
        img = frame
        cv2.imwrite("test.jpg", img)
        class_label = predict_single_image(loaded_model, img, CLASS_LABELS, image_size)   
        print ("Hello {}".format(class_label))
        break
    cv2.imshow('frame',gray)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
