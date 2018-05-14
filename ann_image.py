import cv2
import numpy as np
import math
import random




# A function that takes a bit numpy array and gives an integer
def get_int_for_bits(src_bits):

    answer = 0
    shifted = 1
    
    for i in range(0, len(src_bits)):

        if 1.0 == src_bits[len(src_bits) - i - 1]:
            answer += shifted

        shifted = shifted << 1
        
    return answer


# A function that takes an integer and gives a bit numpy array
def get_bits_for_int(src_min_bits, src_number):
    bits = bin(src_number)[2:]

    a = np.array([])

    for i in range(0, len(bits)):
        a = np.append(a, float(bits[i]))

    num_bits = len(a)
    needed_bits = 0

    if num_bits < src_min_bits:
        needed_bits = src_min_bits - num_bits

    for i in range(0, needed_bits):
        a = np.insert(a, 0, 0.0)

    return a


# A function to snap a floating point value to 0.0 or 1.0
def snapto_0_or_1(position):

    # clamp
    if position < 0:
        position = 0

    if position > 1:
        position = 1

    # round
    return math.floor(0.5 + position)


# A function to shuffle the filenames/classifications
def shuffle(filenames, classifications, num_swaps):

    length = len(filenames)

    for i in range(0, num_swaps):
        index0 = random.randint(0, length - 1)
        index1 = random.randint(0, length - 1)

        temp_filename = filenames[index0]
        temp_classification = classifications[index0]
        
        filenames[index0] = filenames[index1]
        classifications[index0] = classifications[index1]

        filenames[index1] = temp_filename
        classifications[index1] = temp_classification

    return filenames, classifications




# Step 1 -- Train the network

# Read training file/classification list
training_file = open("training_files.txt", "r") 

training_filenames = []
training_classifications = []

for line in training_file:
    training_filenames.append(line.split(" ")[0])
    training_classifications.append(int(line.split(" ")[1]))

# It might be a good idea to pseudorandomly shuffle the filenames/classifications
# This uses an O(n) algorithm to shuffle -- using the length
random.seed(1)
training_filenames, training_classifications = shuffle(training_filenames,
                                                       training_classifications,
                                                       len(training_filenames))

# Get the maximum classification number
max_class = 0

for i in range(0, len(training_classifications)):
    if training_classifications[i] > max_class:
        max_class = training_classifications[i]

num_classes = max_class + 1

# Get minimum number of bits needed to encode num_classes distinct classes
num_bits_needed = math.ceil(math.log(num_classes)/math.log(2.0))

# Get image and ANN parameters
sample_img = cv2.imread(training_filenames[0])
sample_img = cv2.resize(sample_img, (64, 64))

img_rows = sample_img.shape[0]
img_cols = sample_img.shape[1]
channels_per_pixel = 3

num_input_neurons = int(img_rows*img_cols*channels_per_pixel)
num_output_neurons = int(num_bits_needed)
num_hidden_neurons = int(math.ceil(math.sqrt(num_input_neurons*num_output_neurons)))

ann = cv2.ml.ANN_MLP_create()
ann.setLayerSizes(np.array([num_input_neurons, num_hidden_neurons, num_output_neurons], dtype=np.int64))
ann.setActivationFunction(cv2.ml.ANN_MLP_SIGMOID_SYM)
ann.setTermCriteria((cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 1, 0.000001 ))
ann.setTrainMethod(cv2.ml.ANN_MLP_BACKPROP, 0.001)
ann.setBackpropMomentumScale(1.0)
ann.setBackpropWeightScale(0.00001)

# Read image from file
img_input_array = sample_img.flatten()
img_input_array = img_input_array.astype(np.float32)

# Normalize all pixels from [0, 255] to [0, 1]
for i in range(0, img_input_array.shape[0]):
    img_input_array[i] = float(img_input_array[i]) / float(255)

# Get output image
img_output_array = get_bits_for_int(num_output_neurons, training_classifications[0])
img_output_array = img_output_array.astype(np.float32)

# Make both images have 1 row, many columns
img_input_array = img_input_array.reshape(1, img_input_array.shape[0])
img_output_array = img_output_array.reshape(1, img_output_array.shape[0])

# Train the network once, to pull it up by the bootstraps
img_td = cv2.ml.TrainData_create(img_input_array, cv2.ml.ROW_SAMPLE, img_output_array)
ann.train(img_td, cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)

# For each further training iteration, update the weights
# 1 iteration 0.516 OK,
# 10 iterations  OK,
# 100 iterations  OK,
for i in range(0, 10):
    print(i)

    # For each file in the training data
    for j in range(0, len(training_filenames)):

        # Read image from file
        img_input_array = cv2.imread(training_filenames[j])
        img_input_array = cv2.resize(img_input_array, (64, 64))
        img_input_array = img_input_array.flatten()
        img_input_array = img_input_array.astype(np.float32)

        # Normalize all pixels from [0, 255] to [0, 1]
        for k in range(0, img_input_array.shape[0]):
            img_input_array[k] = float(img_input_array[k]) / float(255)

        # Get output image
        img_output_array = get_bits_for_int(num_output_neurons, training_classifications[j])
        img_output_array = img_output_array.astype(np.float32)

        # Make both images have 1 row, many columns
        img_input_array = img_input_array.reshape(1, img_input_array.shape[0])
        img_output_array = img_output_array.reshape(1, img_output_array.shape[0])

        # Train the network, using the update weights parameter
        img_td = cv2.ml.TrainData_create(img_input_array, cv2.ml.ROW_SAMPLE, img_output_array)
        ann.train(img_td, cv2.ml.ANN_MLP_UPDATE_WEIGHTS | cv2.ml.ANN_MLP_NO_INPUT_SCALE | cv2.ml.ANN_MLP_NO_OUTPUT_SCALE)

        
        
        
# Step 2 -- Test the network

# Read in the labels for future use
labels_file = open("meta/labels.txt", "r")

labels = []

for line in labels_file:
    labels_string = line.split("\n")[0]
    labels.append(labels_string)

# Read testing file/classification list
test_file = open("test_files.txt", "r") 

test_filenames = []
test_classifications = []

for line in test_file:
    test_filenames.append(line.split(" ")[0])
    test_classifications.append(int(line.split(" ")[1]))

error_count = 0
ok_count = 0

# For each file in the test data
for i in range(0, len(test_filenames)):

    # Read image from file
    img_input_array = cv2.imread(test_filenames[i])
    img_input_array = cv2.resize(img_input_array, (64, 64))
    img_input_array = img_input_array.flatten()
    img_input_array = img_input_array.astype(np.float32)

    # Normalize all pixels from [0, 255] to [0, 1]
    for j in range(0, img_input_array.shape[0]):
        img_input_array[j] = float(img_input_array[j]) / float(255)

    # Make input image have 1 row, many columns
    img_input_array = img_input_array.reshape(1, img_input_array.shape[0])

    # Ask the network to classify the image
    prediction = ann.predict(img_input_array)

    # snap prediction contents to 0.0s or 1.0s
    for j in range(0, len(prediction[1][0])):
        prediction[1][0][j] = snapto_0_or_1(prediction[1][0][j])

    # Convert to int
    prediction_int = get_int_for_bits(prediction[1][0])
    classification_int = int(test_classifications[i])

    # Sanity check
    # This could be avoided by using one-hot encoding
    if prediction_int > max_class:
        print("Error: prediction " + str(prediction_int) + " out of bounds: max_class = " + str(max_class))
        error_count += 1
        continue
    
    # if the classifications are sane, but not a match, then there is error
    # else it's all good
    if classification_int != prediction_int:
        print("Error: " + labels[classification_int] + " != " + labels[prediction_int])            
        error_count += 1
    else:
        print("OK: " + labels[classification_int] + " == " + labels[prediction_int])    
        ok_count += 1


print(float(ok_count) / float(error_count + ok_count))


