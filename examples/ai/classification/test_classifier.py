import argparse
import os

import numpy as np
import tensorflow as tf

def parse_args():
    args = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    args.add_argument(
        "--dataset_path",
        metavar="dataset_path",
        help="path to dataset",
        default="training_data",
    )
    args.add_argument(
        "--image_width", dest="image_width", type=int, default=324
    )
    args.add_argument(
        "--image_height", dest="image_height", type=int, default=244
    )

    return args.parse_args()

def set_input_tensor(interpreter, input):
    input_details = interpreter.get_input_details()[0]
    tensor_index = input_details["index"]
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = input

def classify_image(interpreter, input):
    set_input_tensor(interpreter, input)
    interpreter.invoke()
    output_details = interpreter.get_output_details()[0]
    output = interpreter.get_tensor(output_details["index"])
    # Outputs from the TFLite model are uint8, so we dequantize the results:
    scale, zero_point = output_details["quantization"]
    output = scale * (output - zero_point)
    top_1 = np.argmax(output)
    return top_1

if __name__ =="__main__":

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    val_generator = val_datagen.flow_from_directory(
        f"{DATASET_PATH}/validation",
        target_size=(args.image_height, args.image_width),
        batch_size=args.batch_size,
        class_mode="categorical",
        color_mode="grayscale",
    )
    
    interpreter = tf.lite.Interpreter(
        f"{ROOT_PATH}/model/classification.tflite"
    )
    interpreter.allocate_tensors()

    batch_images, batch_labels = next(val_generator)

    # Collect all inference predictions in a list
    batch_prediction = []
    batch_truth = np.argmax(batch_labels, axis=1)

    for i in range(len(batch_images)):
        prediction = classify_image(interpreter, batch_images[i])
        batch_prediction.append(prediction)

    # Compare all predictions to the ground truth
    tflite_accuracy = tf.keras.metrics.Accuracy()
    tflite_accuracy(batch_prediction, batch_truth)
    print("TF Lite accuracy: {:.3%}".format(tflite_accuracy.result()))

