from directory_handler import *

import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend

def fun_origin_end_directoriesAndClasses(str_list):
    str_list_end = []
    for string in str_list:
        str_list_end.append(string + "_resize")
    return str_list_end

def build_model(input_shape):
    model = Sequential()

    #Modificamos el kernel para mas precision ya que en este ejemplos los colores son importantes.
    """
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    """
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, optimizer='adam', metrics=['accuracy'])

    return model

def main():

    int_workMode = -1
    int_numeroDeRevisionesAlFinal = 10

    strlist_directoriosOrigen   =   ["Acuatico", "Colina_Montana", "Desertico", "Verde"]
    strlist_directoriosFinales  =   fun_origin_end_directoriesAndClasses(strlist_directoriosOrigen)

    int_xsize   = 144
    int_ysize   = 144
    int_canales = 3

    dh_manager                  =   obj_directoryManager(strlist_directoriosOrigen)
    dh_manager.workMode(int_workMode, strlist_directoriosOrigen, strlist_directoriosFinales, int_xsize, int_ysize)

    #Creamos el set de entrenamiento.
    dh_manager.makeVectors(strlist_directoriosFinales, 40)
    dh_manager.shuffle(160)
    dh_manager.image_vector_state_declaration()
    xtrain = np.array(dh_manager.return_vectorrgb())
    ytrain = np.array(dh_manager.return_classvector())

    #Creamos el set de prueba
    dh_manager.reset_vectors()
    dh_manager.makeVectors2(strlist_directoriosFinales)
    dh_manager.shuffle(len(dh_manager.return_classvector()))
    dh_manager.image_vector_state_declaration()
    xtest  = np.array(dh_manager.return_vectorrgb())
    ytest =  np.array(dh_manager.return_classvector())

    #Aqui inicia la ia ------------------------------------------------------------------------------------------------------------------


    if backend.image_data_format() == 'channels_first':
        xtrain = xtrain.reshape(len(xtrain), int_canales, int_xsize, int_ysize)
        xtest  =  xtest.reshape(len(xtest), int_canales, int_xsize, int_ysize)
        input_shape = (int_canales, int_xsize, int_ysize)

    else:
        xtrain = xtrain.reshape(xtrain.shape[0], int_xsize, int_ysize, int_canales)
        xtest = xtest.reshape(xtest.shape[0], int_xsize, int_ysize, int_canales)
        input_shape = (int_xsize, int_ysize, int_canales)

    ytrain = keras.utils.to_categorical(ytrain, 10)
    ytest  = keras.utils.to_categorical(ytest, 10)

    model = build_model(input_shape)
    model.summary()
    model.fit(xtrain, ytrain, batch_size=128, epochs=10, verbose=1, validation_data=(xtest, ytest))


    #Aquí inicia la revision humana:

    strlist_directoriosOrigenContadorExitos  = []
    strlist_directoriosOrigenContadorGeneral = []
    contador_exitos = 0

    for i in strlist_directoriosOrigen:
        strlist_directoriosOrigenContadorExitos.append(0)
        strlist_directoriosOrigenContadorGeneral.append(0)

    for i in range(0, int_numeroDeRevisionesAlFinal):

        random_number = np.random.randint(0, len(dh_manager.return_classvector()))
        patron = xtest[random_number].reshape(1, int_xsize, int_ysize, int_canales)

        int_prediccion = int(np.argmax(model.predict(patron)))
        int_realidad   = dh_manager.return_classvector()[random_number]
        strlist_directoriosOrigenContadorGeneral[int_realidad] = strlist_directoriosOrigenContadorGeneral[int_realidad] + 1

        if int_prediccion == int_realidad:
            contador_exitos = contador_exitos + 1
            strlist_directoriosOrigenContadorExitos[int_realidad] = strlist_directoriosOrigenContadorExitos[int_realidad] + 1

        plt.imshow(xtest[random_number])
        plt.title('Prediction: ' + dh_manager.get_strclassimage(int_prediccion))
        plt.show()

        plt.imshow(dh_manager.return_vectorrgb()[random_number])
        plt.title('Realidad: ' + dh_manager.get_strclassimage(int_realidad))
        plt.show()


    #Contamos Exitos en la revision humana
    print("-----------------------------------------------------------------------------------------------------------------------")
    porcentaje = "null"

    for int_i in range(0, len(strlist_directoriosOrigen)):
        if strlist_directoriosOrigenContadorGeneral[int_i] == 0:
            porcentaje = " No hubo revision de esta clase."
        else:
            porcentaje = str(100/strlist_directoriosOrigenContadorGeneral[int_i] * strlist_directoriosOrigenContadorExitos[int_i])

        print("De la clase: ", dh_manager.get_strclassimage(int_i),
              ", se acertaron: ", strlist_directoriosOrigenContadorExitos[int_i], ", de esta cantidad revisada: ", strlist_directoriosOrigenContadorGeneral[int_i],
              ", porcentaje de acierto en esta clase: ", porcentaje)

    if int_numeroDeRevisionesAlFinal <= 0:
        porcentaje = "No hubo revision humana."
    else:
        porcentaje = str(100/int_numeroDeRevisionesAlFinal * contador_exitos)

    print("Exitos en general: ", contador_exitos, ", de :", int_numeroDeRevisionesAlFinal, ", porcentaje: ", porcentaje)



    #Contamos Exitos en la revision serializada
    print("-----------------------------------------------------------------------------------------------------------------------")

    strlist_directoriosOrigenContadorExitos  = []
    strlist_directoriosOrigenContadorGeneral = []
    contador_exitos = 0

    for i in strlist_directoriosOrigen:
        strlist_directoriosOrigenContadorExitos.append(0)
        strlist_directoriosOrigenContadorGeneral.append(0)

    to_not_hiperpaginate = len(dh_manager.return_classvector())

    for i in range(0, to_not_hiperpaginate):

        print("Revision :", i, " de : ", to_not_hiperpaginate)
        patron = xtest[i].reshape(1, int_xsize, int_ysize, int_canales)

        int_prediccion = int(np.argmax(model.predict(patron)))
        int_realidad   = dh_manager.return_classvector()[i]
        strlist_directoriosOrigenContadorGeneral[int_realidad] = strlist_directoriosOrigenContadorGeneral[int_realidad] + 1

        if int_prediccion == int_realidad:
            contador_exitos = contador_exitos + 1
            strlist_directoriosOrigenContadorExitos[int_realidad] = strlist_directoriosOrigenContadorExitos[int_realidad] + 1

    for int_i in range(0, len(strlist_directoriosOrigen)):
        if strlist_directoriosOrigenContadorGeneral[int_i] == 0:
            porcentaje = " No hubo revision de esta clase."
        else:
            porcentaje = str(100/strlist_directoriosOrigenContadorGeneral[int_i] * strlist_directoriosOrigenContadorExitos[int_i])

        print("De la clase: ", dh_manager.get_strclassimage(int_i),
              ", se acertaron: ", strlist_directoriosOrigenContadorExitos[int_i], ", de esta cantidad revisada: ", strlist_directoriosOrigenContadorGeneral[int_i],
              ", porcentaje de acierto en esta clase: ", porcentaje)

    print("Exitos en general: ", contador_exitos, ", de :", len(dh_manager.return_classvector()), ", porcentaje: ", str(100/len(dh_manager.return_classvector()) * contador_exitos))


if __name__ == '__main__':
    main()
