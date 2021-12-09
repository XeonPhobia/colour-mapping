import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from tensorflow.python.keras.backend import variable

def calculate_RMSE(input_tensor, comparison_tensor):
    input_array = np.array(input_tensor[0])   
    comparison_array = np.array(comparison_tensor[0])
    diff_array1 = (int(input_array[0]) - int(comparison_array[0]))**2 
    diff_array2 = (int(input_array[1]) - int(comparison_array[1]))**2 
    diff_array3 = (int(input_array[2]) - int(comparison_array[2]))**2      
    diff_array = diff_array1 + diff_array2 + diff_array3 
    diff_array = math.sqrt(diff_array)
    return diff_array

if __name__=='__main__':

    source_img_path = "C:\\VisualStudioCode\\Project2\\red.jpg"
    filter_img_path = "C:\\VisualStudioCode\\Project2\\redblue.jpg"
    source_image=tf.io.read_file(source_img_path)
    filter_image=tf.io.read_file(filter_img_path)
    source_image=tf.image.decode_jpeg(source_image, channels=3)
    filter_image=tf.image.decode_jpeg(filter_image, channels=3)
    
    fit_table = []
    fit_table.insert(0, calculate_RMSE(source_image[0], filter_image[0])) 
    fit_table.insert(1, calculate_RMSE(source_image[0], filter_image[1])) 
    print(fit_table)
