#include <opencv2/opencv.hpp>
using namespace cv;
using namespace ml;
#pragma comment(lib, "opencv_world331.lib")

#include <iostream>
#include <iomanip>
#include <string>
#include <sstream>
#include <algorithm>
#include <cstdlib>
#include <random>
#include <bitset>
using namespace std;


void get_bits_for_int(size_t src_max_bits, size_t src_number, Mat &final_bits)
{
	vector<float> bits;

	ostringstream oss;
	bitset<64> b(src_number);

	for (size_t i = 0; i < src_max_bits; i++)
	{
		if (b[i])
			bits.push_back(1.0f);
		else
			bits.push_back(0.0f);
	}

	reverse(bits.begin(), bits.end());

	final_bits = Mat(1, static_cast<int>(bits.size()), CV_32FC1, &bits[0]).clone();
}

long long unsigned int get_int_for_bits(Mat &bits)
{
	long long unsigned int answer = 0;
	long long unsigned int shifted = 1;

	for (size_t i = 0; i < bits.cols; i++)
	{
		if (1.0 == bits.at<float>(0, bits.cols - i - 1))
			answer += shifted;

		shifted = shifted << 1;
	}

	return answer;
}

float snap_to_0_or_1(float position)
{
	if (position < 0)
		position = 0;
	else if (position > 1)
		position = 1;

	return floor(0.5f + position);
}



void shuffle_files_and_classifications(vector<string> &files, vector<size_t> &classifications, size_t num_swaps, unsigned int prng_seed = 1)
{
	std::mt19937 generator(prng_seed); // mt19937 is a standard //mersenne_twister_engine

	for (size_t i = 0; i < num_swaps; i++)
	{
		size_t index0 = generator() % files.size();
		size_t index1 = generator() % files.size();

		string temp_filename = files[index0];
		size_t temp_classification = classifications[index0];

		files[index0] = files[index1];
		classifications[index0] = classifications[index1];

		files[index1] = temp_filename;
		classifications[index1] = temp_classification;
	}
}

void get_files_and_classifications(const char *training_file_name, vector<string> &files, vector<size_t> &classifications)
{
	files.clear();
	classifications.clear();

	ifstream training_file(training_file_name);

	string line;

	while (getline(training_file, line))
	{
		if (line == "")
			continue;

		size_t space_location = 0;

		for (size_t i = 0; i < line.size(); i++)
		{
			if (line[i] == ' ')
			{
				space_location = i;
				break;
			}
		}

		if (space_location == 0)
			return;

		string a = line.substr(0, space_location);
		string b = line.substr(space_location + 1, line.size() - space_location);
		istringstream b_istream(b);
		size_t b_int = 0;
		b_istream >> b_int;

		files.push_back(a);
		classifications.push_back(b_int);
	}
}

int main(void)
{
	vector<string> training_files;
	vector<size_t> training_classifications;
	get_files_and_classifications("training_files.txt", training_files, training_classifications);

	shuffle_files_and_classifications(training_files, training_classifications, training_files.size());

	size_t max_class = 0;

	for (size_t i = 0; i < training_classifications.size(); i++)
		if (training_classifications[i] > max_class)
			max_class = training_classifications[i];

	size_t num_classes = max_class + 1;

	size_t num_bits_needed = static_cast<size_t>(ceil(log(num_classes) / log(2.0)));

	const int image_width = 64;
	const int image_height = 64;

	// Read in image, resize to 64x64
	Mat sample_img = imread(training_files[0], IMREAD_GRAYSCALE);
	resize(sample_img, sample_img, Size(64, 64));

	// Convert CV_8UC1 to CV_32FC1
	Mat flt_sample_img(sample_img.rows, sample_img.cols, CV_32FC1);

	for (int j = 0; j < sample_img.rows; j++)
		for (int i = 0; i < sample_img.cols; i++)
			flt_sample_img.at<float>(j, i) = sample_img.at<unsigned char>(j, i) / 255.0f;

	size_t img_rows = sample_img.rows;
	size_t img_cols = sample_img.cols;
	size_t channels_per_pixel = 1;

	size_t num_input_neurons = static_cast<size_t>(img_rows*img_cols*channels_per_pixel);
	size_t num_output_neurons = static_cast<size_t>(num_bits_needed);
	size_t num_hidden_neurons = static_cast<size_t>(ceil(sqrt(num_input_neurons*num_output_neurons)));

	Ptr<ANN_MLP> mlp = ANN_MLP::create();

	// Neural network elements
	Mat layersSize = Mat(3, 1, CV_16UC1);
	layersSize.row(0) = static_cast<double>(num_input_neurons);
	layersSize.row(1) = static_cast<double>(num_hidden_neurons);
	layersSize.row(2) = static_cast<double>(num_output_neurons);
	mlp->setLayerSizes(layersSize);

	// Set various parameters
	mlp->setActivationFunction(ANN_MLP::ActivationFunctions::SIGMOID_SYM);
	TermCriteria termCrit = TermCriteria(TermCriteria::Type::COUNT + TermCriteria::Type::EPS, 1, 0.000001);
	mlp->setTermCriteria(termCrit);
	mlp->setTrainMethod(ANN_MLP::TrainingMethods::BACKPROP);
	mlp->setBackpropMomentumScale(0.00001);
	mlp->setBackpropWeightScale(0.1);

	Mat output_data;
	get_bits_for_int(num_output_neurons, training_classifications[0], output_data);
	output_data = output_data.reshape(0, 1);

	// Reshape from 64 rows x 64 columns image to 1 row x (64*64) columns
	flt_sample_img = flt_sample_img.reshape(0, 1);

	Ptr<TrainData> trainingData = TrainData::create(flt_sample_img, SampleTypes::ROW_SAMPLE, output_data);
	mlp->train(trainingData, ANN_MLP::TrainFlags::NO_INPUT_SCALE | ANN_MLP::TrainFlags::NO_OUTPUT_SCALE);

	for (size_t iters = 0; iters < 1; iters++)
	{
		cout << iters << endl;

		for (size_t f = 0; f < training_files.size(); f++)
		{
			//cout << training_files[j] << endl;

			// Read in image, resize to 64x64
			Mat sample_img = imread(training_files[f], IMREAD_GRAYSCALE);
			resize(sample_img, sample_img, Size(64, 64));

			// Convert CV_8UC1 to CV_32FC1
			Mat flt_sample_img(sample_img.rows, sample_img.cols, CV_32FC1);

			for (int j = 0; j < sample_img.rows; j++)
				for (int i = 0; i < sample_img.cols; i++)
					flt_sample_img.at<float>(j, i) = sample_img.at<unsigned char>(j, i) / 255.0f;

			Mat output_data;
			get_bits_for_int(num_output_neurons, training_classifications[f], output_data);
			output_data = output_data.reshape(0, 1);

			// Reshape from 64 rows x 64 columns image to 1 row x (64*64) columns
			flt_sample_img = flt_sample_img.reshape(0, 1);

			Ptr<TrainData> trainingData = TrainData::create(flt_sample_img, SampleTypes::ROW_SAMPLE, output_data);
			mlp->train(trainingData, ANN_MLP::TrainFlags::UPDATE_WEIGHTS | ANN_MLP::TrainFlags::NO_INPUT_SCALE | ANN_MLP::TrainFlags::NO_OUTPUT_SCALE);
		}
	}


	vector<string> labels;
	
	ifstream labels_file("meta/labels.txt");

	string line;

	while (getline(labels_file, line))
	{
		if (line == "")
			continue;

		labels.push_back(line);
	}

	vector<string> testing_files;
	vector<size_t> testing_classifications;
	get_files_and_classifications("test_files.txt", testing_files, testing_classifications);

	size_t error_count = 0;
	size_t ok_count = 0;

	for (size_t f = 0; f < testing_files.size(); f++)
	{
		// Read in image, resize to 64x64
		Mat sample_img = imread(testing_files[f], IMREAD_GRAYSCALE);
		resize(sample_img, sample_img, Size(64, 64));

		// Convert CV_8UC1 to CV_32FC1
		Mat flt_sample_img(sample_img.rows, sample_img.cols, CV_32FC1);

		for (int j = 0; j < sample_img.rows; j++)
			for (int i = 0; i < sample_img.cols; i++)
				flt_sample_img.at<float>(j, i) = sample_img.at<unsigned char>(j, i) / 255.0f;

		// Reshape from 64 rows x 64 columns image to 1 row x (64*64) columns
		flt_sample_img = flt_sample_img.reshape(0, 1);

		Mat p;
		mlp->predict(flt_sample_img, p);

		for (size_t i = 0; i < p.cols; i++)
			p.at<float>(0, i) = snap_to_0_or_1(p.at<float>(0, i));

		size_t prediction_int = get_int_for_bits(p);
		size_t classification_int = testing_classifications[f];

		if (prediction_int > max_class)
		{
			cout << "Error: prediction " << prediction_int << " out of bounds: max_class = " << max_class << endl;
			error_count++;
			continue;
		}

		if (classification_int != prediction_int)
		{
			cout << "Error: " << labels[classification_int] << " != " << labels[prediction_int] << endl;
			error_count++;
		}
		else
		{
			cout << "OK: " << labels[classification_int] << " == " << labels[prediction_int] << endl;
			ok_count++;
		}
	}

	cout << float(ok_count) / float(error_count + ok_count) << endl;
	return 0;
}	