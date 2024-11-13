#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <algorithm>

using namespace std;

// Helper functions ---------------------------------------------
double reLU(double x) {
    return x > 0 ? x : 0;
}
double reLU_derivative(double x) {
    return x > 0 ? 1 : 0;
}
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}
double crossEntropy(const vector<double>& predictedOutput, const uint8_t& trueLabel) {
    // Convert trueLabel to one-hot encoding
    vector<double> oneHotLabel(10, 0.0);
    oneHotLabel[trueLabel] = 1.0;

    // Calculate cross-entropy loss
    double loss = 0.0;
    const double epsilon = 1e-9;  // Small value to avoid log(0)
    for (int i = 0; i < 10; ++i) {
        double predicted = max(epsilon, min(1 - epsilon, predictedOutput[i]));  // clip values to ensure they are between epsilon and 1-epsilon
        if (oneHotLabel[i] == 1.0) {
            loss -= log(predicted);
        }
    }
    return loss;
}
double meanSquaredError(const vector<double>& predictedOutput, const uint8_t& trueLabel) {
    // Convert trueLabel to one-hot encoding
    vector<double> oneHotLabel(10, 0.0);
    oneHotLabel[trueLabel] = 1.0;

    // Calculate mean squared error
    double loss = 0.0;
    for (int i = 0; i < 10; ++i) {
        loss += (oneHotLabel[i] - predictedOutput[i]) * (oneHotLabel[i] - predictedOutput[i]);
    }
    return loss / 10.0;  // Divide by the number of classes for mean
}

void printImage(const vector<uint8_t>& image) {
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            cout << (image[i * 28 + j] > 128 ? "#" : " ");
        }
        cout << endl;
    }
}
void writeImage(const vector<uint8_t>& image) {
    ofstream file("output.txt", ios::app);
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            file << (image[i * 28 + j] > 128 ? "#" : " ");
        }
        file << endl;
    }
    file << endl;
    file.close();
}
// --------------------------------------------------------------

// Network class ------------------------------------------------
class Network {
    public:
        Network(int argc, char **argv);
        ~Network();
        void const printArguments();
        void const printWeights();
        void train(const string& imageFile, const string& labelFile, const string& activationFunction, const string& lossFunction, const double& learningRate, const int numEpochs = 10);
        void test(const string& testImageFile, const string& testLabelFile);

    private:
        int argc;
        char **argv;

        string activationFunction;
        string lossFunction;
        double learningRate;

        int imageMagicNumber;
        string imageFile;
        string imagePath;
        int numImages;
        int rows;
        int cols;
        vector<vector<uint8_t>> images;

        int labelMagicNumber;
        string labelFile;
        string labelPath;
        int numLabels;
        vector<uint8_t> labels;

        vector<vector<double>> weights_hidden;
        vector<vector<double>> weights_output;
        vector<double> bias_hidden;
        vector<double> bias_output;

        vector<vector<uint8_t>> readMNISTImages(const string& filename);
        vector<uint8_t> readMNISTLabels(const string& filename);
        int32_t readInt(std::ifstream &file);

        void randomInitialization();
        vector<double> activateHidden(const vector<uint8_t>& image);
        vector<double> activateOutput(const vector<double>& hidden_activations);
        double computeLoss(const vector<double>& predictedOutput, const uint8_t& trueLabel);
        void backpropagate(const vector<uint8_t>& image, const vector<double>& predictedOutput, const vector<double>& hiddenActivations, const uint8_t& trueLabel);
        double testAccuracy(const vector<vector<uint8_t>>& testImages, const vector<uint8_t>& testLabels);
};
// -----------------------------------------------------------

// Network class constructor and destructor ---------------------
Network::Network(int argc, char **argv) {
    cout << "Network object created" << endl;

    // Store the arguments
    this->argc = argc;
    this->argv = argv;
}
Network::~Network() {
    cout << "Network object destroyed" << endl;
}
// -----------------------------------------------------------

// Network class private functions ------------------------------
vector<vector<uint8_t>> Network::readMNISTImages(const string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    // Read metadata
    this->imageMagicNumber = this->readInt(file);
    this->numImages = this->readInt(file);
    this->rows = this->readInt(file);
    this->cols = this->readInt(file);

    // Read image data
    std::vector<std::vector<uint8_t>> images(this->numImages, std::vector<uint8_t>(this->rows * this->cols));
    for (int i = 0; i < this->numImages; ++i) {
        file.read(reinterpret_cast<char*>(images[i].data()), this->rows * this->cols);
    }

    file.close();
    return images;
}
vector<uint8_t> Network::readMNISTLabels(const string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open file: " << filename << std::endl;
        exit(1);
    }

    // Read metadata
    this->labelMagicNumber = this->readInt(file);
    this->numLabels = this->readInt(file);

    // Read label data
    std::vector<uint8_t> labels(this->numLabels);
    file.read(reinterpret_cast<char*>(labels.data()), this->numLabels);

    file.close();
    return labels;
}
int32_t Network::readInt(std::ifstream &file) {
    int32_t result = 0;
    file.read(reinterpret_cast<char*>(&result), 4);
    return __builtin_bswap32(result);  // For big-endian to little-endian conversion
}

void Network::randomInitialization() {
    double stddev_hidden = sqrt(2.0 / 784); // He initialization for ReLU activations
    double stddev_output = sqrt(2.0 / 128); // He initialization for ReLU activations

    for (auto& row : this->weights_hidden) {
        for (auto& w : row) {
            w = stddev_hidden * static_cast<double>(rand()) / RAND_MAX;
        }
    }

    for (auto& row : this->weights_output) {
        for (auto& w : row) {
            w = stddev_output * static_cast<double>(rand()) / RAND_MAX;
        }
    }

    for (auto& b : this->bias_hidden) {
        b = static_cast<double>(rand()) / RAND_MAX;
    }

    for (auto& b : this->bias_output) {
        b = static_cast<double>(rand()) / RAND_MAX;
    }
}
vector<double> Network::activateHidden(const vector<uint8_t>& input) {
    // 128 neurons in the hidden layer
    vector<double> hiddenActivations(128, 0.0);

    for (int j = 0; j < 128; ++j) {
        double activation = 0.0;
        for (int i = 0; i < 784; ++i) {
            activation += input[i] / 255.0 * this->weights_hidden[i][j]; // Normalized input
        }
        // add the bias term
        activation += this->bias_hidden[j];

        // Apply the activation function
        if (this->activationFunction == "reLU") {
            hiddenActivations[j] = reLU(activation);
        } else if (this->activationFunction == "sigmoid") {
            hiddenActivations[j] = sigmoid(activation);
        }
    }

    return hiddenActivations;
}
vector<double> Network::activateOutput(const vector<double>& hiddenActivations) {
    vector<double> outputActivations(10, 0.0);
    vector<double> preSoftmax(10, 0.0);

    for (int k = 0; k < 10; ++k) {
        double activation = 0.0;
        for (int j = 0; j < 128; ++j) {
            activation += hiddenActivations[j] * this->weights_output[j][k]; // weighted sum
        }
        // add the bias term
        activation += this->bias_output[k];

        // Store the pre-softmax value
        preSoftmax[k] = activation;
    }

    // Apply softmax
    double maxValue = *max_element(preSoftmax.begin(), preSoftmax.end());
    double sumExp = 0.0;
    for (const double& val : preSoftmax) {
        sumExp += exp(val - maxValue); // Stability improvement by subtracting maxValue
    }
    for (int k = 0; k < 10; ++k) {
        outputActivations[k] = exp(preSoftmax[k] - maxValue) / sumExp;  // Normalize to get probabilities
    }

    return outputActivations;
}
double Network::computeLoss(const vector<double>& predictedOutput, const uint8_t& trueLabel) {
    double loss = 0.0;

    if (this->lossFunction == "crossEntropy") {
        loss = crossEntropy(predictedOutput, trueLabel);
    }
    else if (this->lossFunction == "MSE") {
        loss = meanSquaredError(predictedOutput, trueLabel);
    }

    return loss;
}
void Network::backpropagate(const vector<uint8_t>& image, const vector<double>& predictedOutput, const vector<double>& hiddenActivations, const uint8_t& trueLabel) {
    // Step 1: Convert true label to one-hot encoding
    vector<double> oneHotLabel(10, 0.0);
    oneHotLabel[trueLabel] = 1.0;

    // Step 2: Calculate output layer error
    vector<double> outputErrors(10);
    for (int j = 0; j < 10; ++j) {
        outputErrors[j] = predictedOutput[j] - oneHotLabel[j];
    }

    // Step 3: Calculate hidden layer error
    vector<double> hiddenErrors(hiddenActivations.size());
    for (int i = 0; i < hiddenActivations.size(); ++i) {
        hiddenErrors[i] = 0.0;
        for (int j = 0; j < 10; ++j) {
            hiddenErrors[i] += outputErrors[j] * this->weights_output[i][j];
        }

        // May be reLU or sigmoid
        if (this->activationFunction == "reLU") {
            hiddenErrors[i] *= reLU_derivative(hiddenActivations[i]);
        } else if (this->activationFunction == "sigmoid") {
            hiddenErrors[i] *= hiddenActivations[i] * (1 - hiddenActivations[i]);
        }
    }

    // Step 4: Update weights for hidden-to-output layer
    for (int i = 0; i < hiddenActivations.size(); ++i) {
        for (int j = 0; j < 10; ++j) {
            this->weights_output[i][j] -= this->learningRate * outputErrors[j] * hiddenActivations[i];
        }
    }

    // Step 5: Update weights for input-to-hidden layer
    for (int i = 0; i < 784; ++i) {
        for (int j = 0; j < hiddenActivations.size(); ++j) {
            this->weights_hidden[i][j] -= this->learningRate * hiddenErrors[j] * (image[i] / 255.0);
        }
    }

    // Update biases
    for (int j = 0; j < 10; ++j) {
        this->bias_output[j] -= this->learningRate * outputErrors[j];
    }
    for (int j = 0; j < hiddenActivations.size(); ++j) {
        this->bias_hidden[j] -= this->learningRate * hiddenErrors[j];
    }
}
double Network::testAccuracy(const vector<vector<uint8_t>>& testImages, const vector<uint8_t>& testLabels) {
    int correctPredictions = 0;

    for (size_t i = 0; i < testImages.size(); ++i) {
        vector<double> hiddenActivations = this->activateHidden(testImages[i]);
        vector<double> outputActivations = this->activateOutput(hiddenActivations);

        auto maxElementIt = std::max_element(outputActivations.begin(), outputActivations.end());
        int prediction = std::distance(outputActivations.begin(), maxElementIt);

        if (prediction == testLabels[i]) {
            correctPredictions++;
        }

        // Write the image and prediction to a file called output.txt
        ofstream file("output.txt", ios::app);
        file << "Prediction: " << prediction << ", True Label: " << static_cast<int>(testLabels[i]) << endl;
        writeImage(testImages[i]);
        file.close();
    }

    double accuracy = static_cast<double>(correctPredictions) / testImages.size();
    return accuracy;
}
// -----------------------------------------------------------

// Network class public functions -------------------------------
void const Network::printArguments() {
    for (int i = 0; i < this->argc; ++i) {
        cout << "Argument " << i << ": " << this->argv[i] << endl;
    }
}
void const Network::printWeights() {
    cout << "Hidden Layer Weights: " << endl;
    for (int i = 0; i < this->weights_hidden.size(); ++i) {
        for (int j = 0; j < this->weights_hidden[i].size(); ++j) {
            cout << this->weights_hidden[i][j] << " ";
        }
        cout << endl;
    }

    cout << "Output Layer Weights: " << endl;
    for (int i = 0; i < this->weights_output.size(); ++i) {
        for (int j = 0; j < this->weights_output[i].size(); ++j) {
            cout << this->weights_output[i][j] << " ";
        }
        cout << endl;
    }
}
void Network::train(const string& imageFile, const string& labelFile, const string& activationFunction, const string& lossFunction, const double& learningRate, const int numEpochs) {
    string folder = "data/";
    this->imageFile = imageFile;
    this->labelFile = labelFile;
    this->activationFunction = activationFunction;
    this->lossFunction = lossFunction;
    this->learningRate = learningRate;

    this->imagePath = folder + imageFile;
    this->labelPath = folder + labelFile;

    this->images = this->readMNISTImages(this->imagePath);
    this->labels = this->readMNISTLabels(this->labelPath);
    
    cout << "Training the network..." << endl;

    this->weights_hidden = vector<vector<double>>(784, vector<double>(128));
    this->weights_output = vector<vector<double>>(128, vector<double>(10));
    this->bias_hidden = vector<double>(128);
    this->bias_output = vector<double>(10);

    this->randomInitialization();

    // Training loop
    for (int epoch = 0; epoch < numEpochs; ++epoch) {
        double epoch_loss = 0.0;

        for (size_t i = 0; i < images.size(); ++i) {
            // Step 1: Forward pass
            vector<double> hiddenActivations = activateHidden(images[i]);
            vector<double> outputActivations = activateOutput(hiddenActivations);

            // Step 2: Compute loss
            double loss = computeLoss(outputActivations, labels[i]);

            // Step 3: Backpropagate
            backpropagate(images[i], outputActivations, hiddenActivations, labels[i]);

            epoch_loss += loss;

            if (i % 1000 == 999) {
                cout << "Epoch " << epoch + 1 << "/" << numEpochs 
                        << ", Image " << i + 1 << "/" << images.size() 
                        << ", Loss: " << epoch_loss / (i + 1) << endl;
            }
        }

        // Average loss for the epoch
        epoch_loss /= images.size();
        cout << "Epoch " << epoch + 1 << "/" << numEpochs << ", Loss: " << epoch_loss << endl;
    }

    cout << "Training complete." << endl;
}
void Network::test(const string& testImageFile, const string& testLabelFile) {
    string folder = "data/";
    string testImagePath = folder + testImageFile;
    string testLabelPath = folder + testLabelFile;

    // Clear the text currently in the output.txt file
    ofstream file("output.txt");
    file.close();

    cout << "Testing the network..." << endl;
    vector<vector<uint8_t>> testImages = this->readMNISTImages(testImagePath);
    vector<uint8_t> testLabels = this->readMNISTLabels(testLabelPath);

    double accuracy = testAccuracy(testImages, testLabels);
    cout << "Testing Accuracy: " << accuracy * 100 << "%\n";
}
// -----------------------------------------------------------

// Network function taking in arguments -------------------------
int main(int argc, char **argv) {
    Network network(argc, argv);

    // Print the arguments
    network.printArguments();

    string trainImages = argv[1];
    string trainLabels = argv[2];
    string testImages = argv[3];
    string testLabels = argv[4];
    string activation = argv[5];
    string loss = argv[6];
    double learningRate = stod(argv[7]);
    int numEpochs = stoi(argv[8]);

    // Activation: reLU, sigmoid
    // Loss: crossEntropy, MSE
    network.train(trainImages, trainLabels, activation, loss, learningRate, numEpochs);
    network.test(testImages, testLabels);
    return 0;
}
// -----------------------------------------------------------