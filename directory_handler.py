from os import listdir, rename, remove, mkdir
from os.path import isfile, join, exists
from PIL import Image
import matplotlib.image as mpimg
import numpy as np
import random
import matplotlib.pyplot as plt

class obj_directoryManager:

    def __init__(self, strlist_classesnames):
        self._strList_filesNames    =  []
        self._imagevector           =  []
        self._classvector           =  []
        self._classvectorstr        =  strlist_classesnames


    def __setget_list_fileNames(self, str_directoryPath):
        self._strList_filesNames = []
        self._strList_filesNames = [f for f in listdir(str_directoryPath) if isfile(join(str_directoryPath, f))]

    def extset_changeFileNamesOfDirectory(self, str_directoryPath, str_baseName):

        self.__setget_list_fileNames(str_directoryPath)

        for int_i in range(0, len(self._strList_filesNames)):
            str_fileDirectoryPlusName = (join(str_directoryPath, self._strList_filesNames[int_i]))
            #str_fileName = str_fileDirectoryPlusName.split(sep=".")[0]
            str_fileExtn = str_fileDirectoryPlusName.split(sep=".")[1]
            print(str_fileDirectoryPlusName)
            print(join(str_directoryPath, str_baseName + str(int_i) + "." + str_fileExtn))

            if isfile(join(str_directoryPath, str_baseName + str(int_i) + "." + str_fileExtn)):
                print("Archivos ya ordenados.")
                break

            rename(str_fileDirectoryPlusName, join(str_directoryPath, str_baseName + str(int_i) + "." + str_fileExtn))

    def __extset_deleteFilesFromDirectory(self, str_directoryPath):
        for f in listdir(str_directoryPath):
            remove(join(str_directoryPath, f))

    def extset_imageResizerSetToDirectory(self, str_sourceDirectoryPath, str_endDirectoryPath, lon_size1, lon_size2):

        if exists(str_endDirectoryPath):
            self.__extset_deleteFilesFromDirectory(str_endDirectoryPath)
        else:
            mkdir(str_endDirectoryPath)

        self.__setget_list_fileNames(str_sourceDirectoryPath)

        for int_i in range(0, len(self._strList_filesNames)):

            str_fileDirectoryPlusName = (join(str_sourceDirectoryPath, self._strList_filesNames[int_i]))
            str_originalFileName = self._strList_filesNames[int_i]
            str_fileNameWithOutExtension = str_originalFileName.split(sep=".")[0]
            str_fileExtensionWithOutPointAndName = str_originalFileName.split(sep=".")[1]

            image = Image.open(str_fileDirectoryPlusName)
            image = image.resize((lon_size1, lon_size2))
            str_final_name = join(str_endDirectoryPath, str_fileNameWithOutExtension + "_resize" + "." + str_fileExtensionWithOutPointAndName)
            print(str_final_name)
            image.save(str_final_name)

    def __extget_rgbvector255OfImage(self, str_fileName):
        img = mpimg.imread(str_fileName)
        vector_rgb = np.array(img, dtype = np.int)
        return vector_rgb


    def extset_rgbvectorOfImageCollectionAndClass(self, str_directoryPath, int_class, int_limit):
        self.__setget_list_fileNames(str_directoryPath)

        for int_i in range(0, len(self._strList_filesNames)):
            if int_i >= int_limit:
                break
            self._imagevector.append(self.__extget_rgbvector255OfImage(join(str_directoryPath, self._strList_filesNames[int_i])))
            self._classvector.append(int_class)

        print("Hay tal cantidad de datos en el vector de color: ",  len(self._imagevector))
        print("Hay tal cantidad de datos en el vector de clases: ", len(self._classvector))

    def extset_rgbvectorOfImageCollectionAndClass2(self, str_directoryPath, int_class):
        self.__setget_list_fileNames(str_directoryPath)

        for int_i in range(0, len(self._strList_filesNames)):
            self._imagevector.append(self.__extget_rgbvector255OfImage(join(str_directoryPath, self._strList_filesNames[int_i])))
            self._classvector.append(int_class)

        print("Hay tal cantidad de datos en el vector de color: ",  len(self._imagevector))
        print("Hay tal cantidad de datos en el vector de clases: ", len(self._classvector))

    def image_vector_state_declaration(self):
        print("Existe este vector: ", self._classvector)
        #print("Existe estos vectores de imagen float rgb: ", self._imagevector)

    def shuffle(self, int_howManyTimes):
        for i in range(0, int_howManyTimes):

            n1 = random.randint(0, len(self._classvector)-1)
            n2 = random.randint(0, len(self._classvector)-1)

            self._imagevector[n1], self._imagevector[n2] = self._imagevector[n2], self._imagevector[n1]
            self._classvector[n1], self._classvector[n2] = self._classvector[n2], self._classvector[n1]

    def fun_ocurrences_of_each_class(self):
        for i in range(0, len(self._classvectorstr)):
            print("De la clase : ", self.get_strclassimage(i), ", hay: ",  self._classvector.count(i))

    def return_vectorrgb(self):
        return self._imagevector

    def return_classvector(self):
        return self._classvector

    def print_rgbvector_toImage(self, vector_rgb, str_tittle):
        plt.title(str_tittle)
        plt.imshow(vector_rgb)
        plt.show()

    def get_strclassimage(self, int_class):
        return self._classvectorstr[int_class]

    def workMode(self, int_fileMagmentMode, strlist_directoriosOrigen, strlist_directoriosFinales, int_xsize, int_ysize):

        if int_fileMagmentMode == -1:
            return

        if int_fileMagmentMode == 0:
            for int_i in range(0, len(strlist_directoriosOrigen)):
                self.extset_changeFileNamesOfDirectory(strlist_directoriosOrigen[int_i], strlist_directoriosFinales[int_i])
                self.extset_imageResizerSetToDirectory(strlist_directoriosOrigen[int_i], strlist_directoriosFinales[int_i], int_xsize, int_ysize)

        if int_fileMagmentMode == 1:
            for int_i in range(0, len(strlist_directoriosOrigen)):
                self.extset_imageResizerSetToDirectory(strlist_directoriosOrigen[int_i], strlist_directoriosFinales[int_i], int_xsize, int_ysize)

    def makeVectors(self, strlist_directorios, int_numeroDeMuestrasPorDirectorio):
        for int_i in range(0, len(strlist_directorios)):
            self.extset_rgbvectorOfImageCollectionAndClass(strlist_directorios[int_i], int_i, int_numeroDeMuestrasPorDirectorio)

    def makeVectors2(self, strlist_directorios):
        for int_i in range(0, len(strlist_directorios)):
            self.extset_rgbvectorOfImageCollectionAndClass2(strlist_directorios[int_i], int_i)

    def reset_vectors(self):
        self._imagevector = []
        self._classvector = []