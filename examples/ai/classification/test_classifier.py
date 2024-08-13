import argparse
import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
from PIL import Image,ImageDraw, ImageFont

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
    args.add_argument("--batch_size", dest="batch_size", type=int, default=8)

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
    args = parse_args()

    ROOT_PATH = (
        f"{os.path.abspath(os.curdir)}/GAP8/ai_examples/classification/"
    )
    DATASET_PATH = f"{ROOT_PATH}{args.dataset_path}"
    if not os.path.exists(DATASET_PATH):
        ROOT_PATH = "./"
        DATASET_PATH = args.dataset_path
    if not os.path.exists(DATASET_PATH):
        raise ValueError(f"Dataset path '{DATASET_PATH}' does not exist.")
    print(DATASET_PATH + "/*/*/*")

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    val_generator = val_datagen.flow_from_directory(
        f"{DATASET_PATH}/validation",
        target_size=(args.image_height, args.image_width),
        batch_size=args.batch_size,
        class_mode="categorical",
        color_mode="grayscale",
    )
    
    interpreter = tf.lite.Interpreter(
        f"{ROOT_PATH}/model/classification_q.tflite"
    )
    interpreter.allocate_tensors()

    save_dir = 'validation_results'
    os.makedirs(save_dir, exist_ok=True)

    # Initialize the accuracy metric
    overall_accuracy = tf.keras.metrics.Accuracy()

    # Loop over the entire validation set
    for i in tqdm(range(len(val_generator))):

        batch_images, batch_labels = next(val_generator)

        # Collect predictions for the current batch
        batch_prediction = []
        batch_truth = np.argmax(batch_labels, axis=1)

        for j in range(len(batch_images)):
            prediction = classify_image(interpreter, batch_images[j])
            batch_prediction.append(prediction)

            if i == 0:
                image = Image.fromarray((batch_images[j]).astype(np.uint8).squeeze())
                draw = ImageDraw.Draw(image)
                font = ImageFont.load_default()

                # Prepare the text
                pred_text = f"Predicted: {prediction}"
                truth_text = f"Truth: {batch_truth[j]}"

                # Define text position
                text_position = (10, 10)  # top-left corner
                draw.text(text_position, f"{pred_text}, {truth_text}", fill="white", font=font)

                # Save the image with a unique name
                image_name = f'image_{i}_{j}.png'
                image.save(os.path.join(save_dir, image_name))

        # Update the accuracy metric with the current batch
        overall_accuracy.update_state(batch_prediction, batch_truth)

    # Get the final accuracy over the entire validation set
    final_accuracy = overall_accuracy.result().numpy()
    print("TF Lite accuracy: {:.3%}".format(final_accuracy))

