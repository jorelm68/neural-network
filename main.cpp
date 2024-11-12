#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>
#include <cmath>

using namespace std;

// Helper functions ---------------------------------------------
double reLU(double x) {
    return x > 0 ? x : 0;
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
    loss /= 10.0;  // Divide by the number of classes for mean
    return loss;
}
// --------------------------------------------------------------

// Network class ------------------------------------------------
class Network {
    public:
        Network(int argc, char **argv);
        ~Network();
        void const printArguments();
        void const printImage(const int& index);
        void const printWeights();
        void readMNIST(const string& imageFile, const string& labelFile, const string& activationFunction, const string& lossFunction, const int& learningRate);
        void train(const int numEpochs);

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
        void backpropagate(const vector<uint8_t>& image, const vector<double>& predictedOutput, const vector<double>& output_activations, const uint8_t& trueLabel);
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
    return __builtin_bswap32(result);  // For big-endian to little-endian
}

void Network::randomInitialization() {
    for (auto& row : this->weights_hidden) {
        for (auto& w : row) {
            w = static_cast<double>(rand()) / RAND_MAX;
        }
    }

    for (auto& row : this->weights_output) {
        for (auto& w : row) {
            w = static_cast<double>(rand()) / RAND_MAX;
        }
    }
}
vector<double> Network::activateHidden(const vector<uint8_t>& input) {
    // 128 neurons in the hidden layer
    vector<double> hiddenActivations(128, 0.0);

    for (int j = 0; j < 128; ++j) {
        double activation = 0.0;
        for (int i = 0; i < 784; ++i) {
            activation += input[i] * this->weights_hidden[i][j]; // weighted sum
        }
        // add the bias term
        activation += this->bias_hidden[j];

        // Apply the activation function
        if (this->activationFunction == "reLU") {
            hiddenActivations[j] = reLU(activation);
        }
    }

    return hiddenActivations;
}
vector<double> Network::activateOutput(const vector<double>& hidden_activations) {
    vector<double> outputActivations(10, 0.0);

    for (int k = 0; k < 10; ++k) {
        double activation = 0.0;
        for (int j = 0; j < 128; ++j) {
            activation += hidden_activations[j] * this->weights_output[j][k]; // weighted sum
        }
        // add the bias term
        activation += this->bias_output[k];

        // Apply the activation function
        if (this->activationFunction == "reLU") {
            outputActivations[k] = reLU(activation);
        }
    }

    // Apply softmax
    double sumExp = 0.0;
    for (double val : outputActivations) {
        sumExp += exp(val);
    }
    for (double& val : outputActivations) {
        val = exp(val) / sumExp;  // Normalize to get probabilities
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
        outputErrors[j] = (predictedOutput[j] - oneHotLabel[j]) * predictedOutput[j] * (1 - predictedOutput[j]); // Derivative of MSE + softmax
    }

    // Step 3: Calculate hidden layer error
    vector<double> hiddenErrors(hiddenActivations.size());
    for (int i = 0; i < hiddenActivations.size(); ++i) {
        hiddenErrors[i] = 0.0;
        for (int j = 0; j < 10; ++j) {
            hiddenErrors[i] += outputErrors[j] * this->weights_output[i][j];
        }
        hiddenErrors[i] *= hiddenActivations[i] * (1 - hiddenActivations[i]); // Derivative of sigmoid
    }

    // Step 4: Update weights for hidden-to-output layer
    for (int i = 0; i < hiddenActivations.size(); ++i) {
        for (int j = 0; j < 10; ++j) {
            this->weights_output[i][j] -= this->learningRate * outputErrors[j] * hiddenActivations[i];
        }
    }

    // Step 5: Update weights for input-to-hidden layer
    for (int i = 0; i < image.size(); ++i) {
        for (int j = 0; j < hiddenActivations.size(); ++j) {
            this->weights_hidden[i][j] -= this->learningRate * hiddenErrors[j] * image[i];
        }
    }
}
// -----------------------------------------------------------

// Network class public functions -------------------------------
void const Network::printArguments() {
    for (int i = 0; i < this->argc; ++i) {
        cout << "Argument " << i << ": " << this->argv[i] << endl;
    }
}
void const Network::printImage(const int& index) {
    // Display the first image and label as an example
    if (numImages > 0 && numLabels > 0) {
        std::cout << "Printing image " << index << " with label " <<  static_cast<int>(this->labels[index]) << "\n";
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                std::cout << (images[index][i * cols + j] > 0 ? "*" : " ") << " ";
            }
            std::cout << "\n";
        }
    }
}
void const Network::printWeights() {
    for (int i = 0; i < this->weights_hidden.size(); ++i) {
        for (int j = 0; j < this->weights_hidden[i].size(); ++j) {
            cout << this->weights_hidden[i][j] << " ";
        }
        cout << endl;
    }

    for (int i = 0; i < this->weights_output.size(); ++i) {
        for (int j = 0; j < this->weights_output[i].size(); ++j) {
            cout << this->weights_output[i][j] << " ";
        }
        cout << endl;
    }
}
void Network::readMNIST(const string& imageFile, const string& labelFile, const string& activationFunction, const string& lossFunction, const int& learningRate) {
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
}
void Network::train(const int numEpochs = 10) {
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
            epoch_loss += computeLoss(outputActivations, labels[i]);

            // Step 3: Backpropagate
            backpropagate(images[i], outputActivations, hiddenActivations, labels[i]);

            // Print loss every 1000 images
            if (i % 1000 == 0) {
                cout << "Epoch " << epoch + 1 << "/" << numEpochs << ", Image " << i << "/" << images.size() << ", Loss: " << epoch_loss / (i + 1) << endl;
            }
        }

        // Average loss for the epoch
        epoch_loss /= images.size();
        cout << "Epoch " << epoch + 1 << "/" << numEpochs << ", Loss: " << epoch_loss << endl;
    }

    cout << "Training complete." << endl;
}
// -----------------------------------------------------------

// Network function taking in arguments -------------------------
int main(int argc, char **argv) {
    Network network(argc, argv);

    // Activation: reLU
    // Loss: crossEntropy, MSE
    network.readMNIST("train-images-idx3-ubyte", "train-labels-idx1-ubyte", "reLU", "crossEntropy", 0.01);

    network.train(2);
    return 0;
}
// -----------------------------------------------------------