import numpy, cv2, os
import numpy as np
from tensorflow.keras.utils import to_categorical
import tensorflow
from keras.layers import *
from keras.models import *
from tensorflow.keras.applications import *

def create_train_data(path, size, colormode):                                # Метод для чтения картинок и их классов
  X_train = []                                                               # Создаем массив для картинок
  Y_train = []                                                               # Создаем массив для номеров классов

  labelID  = 0                                                               # Текущий номер класса

  class_names = sorted(os.listdir(path))                                     # Список каталогов, он же список номер-имя класса

  for dir in class_names:                                                    # Проходим по всем каталогам из датасета
    for img in os.listdir(os.path.join(path, dir)):                          # И по всем картинкам
      if img.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.webm')):   # Читаем только изображения указанных форматов
          img_path = os.path.join(os.path.join(path, dir), img)              # Собираем полный путь к картинке
          image = cv2.resize(cv2.imread(img_path, colormode), size)          # Читаем её средствами CV, а также масштабируем до требуемого размера
          X_train.append(list(np.array(image)))                              # Добавляем изображение в массив X_train
          Y_train.append([labelID])                                          # Добавляем метку для него в Y_train
    labelID=labelID+1                                                        # Как каталог закончился - увеличиваем LabelID на 1
       
  X_train=np.array(X_train)                                                  # Преобразовываем LIST в NPDARRAY
  Y_train=np.array(Y_train)   
  Y_train = to_categorical(Y_train, num_classes=labelID, dtype='float32')    # Немного магии: Преобразуем номер в массив 

                                                                             # Например: [4] => [0, 0, 0, 0, 1, 0,...]     
                                                                             #           [1] => [0, 1, 0, 0, 0, 0,...]   
                                                                             #           [0] => [1, 0, 0, 0, 0, 0,...]                                          
  
  return (X_train,Y_train,class_names, len(class_names))                     # Возвращаем результат





# Можно использовать уже готовую архиткутуру сети. 
# Например, MobileNetV2. Нам остаётся только указать размер картинки, и описать 2 последних слоя.
def build_MobileNet(size, outputs):
    input_tensor = Input(shape=(size[0], size[1],3))
    base_model = MobileNetV2(include_top=False, input_tensor=input_tensor, pooling='avg')   
    op = Dense(256, activation='relu')(base_model.output)
    output_tensor = Dense(outputs, activation='softmax')(op)
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile("Adam",
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])
    return model