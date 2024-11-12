#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

using namespace std;

// Network class ------------------------------------------------
class Network {
    public:
        Network(int argc, char **argv);
        ~Network();
        void const printArguments();
        void const printImage(const int& index);
        void readMNIST(const string& imageFile, const string& labelFile);
        void train();

    private:
        int argc;
        char **argv;

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

        vector<vector<uint8_t>> readMNISTImages(const string& filename);
        vector<uint8_t> readMNISTLabels(const string& filename);
        int32_t readInt(std::ifstream &file);
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
void Network::readMNIST(const string& imageFile, const string& labelFile) {
    string folder = "data/";
    this->imageFile = imageFile;
    this->labelFile = labelFile;

    this->imagePath = folder + imageFile;
    this->labelPath = folder + labelFile;

    this->images = this->readMNISTImages(this->imagePath);
    this->labels = this->readMNISTLabels(this->labelPath);
}
void Network::train() {
    cout << "Training the network..." << endl;

    // Train the network here

}
// -----------------------------------------------------------

// Network function taking in arguments -------------------------
int main(int argc, char **argv) {
    Network network(argc, argv);

    network.readMNIST("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    return 0;
}
// -----------------------------------------------------------