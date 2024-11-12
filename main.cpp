#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <cstdint>

using namespace std;

// Main class ------------------------------------------------
class Main {
    public:
        Main(int argc, char **argv);
        ~Main();
        void const printArguments();
        void const printImage(const int& index);
        void MNIST(const string& imageFile, const string& labelFile);

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

// Main class constructor and destructor ---------------------
Main::Main(int argc, char **argv) {
    cout << "Main object created" << endl;

    // Store the arguments
    this->argc = argc;
    this->argv = argv;
}
Main::~Main() {
    cout << "Main object destroyed" << endl;
}
// -----------------------------------------------------------

// Main class private functions ------------------------------
vector<vector<uint8_t>> Main::readMNISTImages(const string& filename) {
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
vector<uint8_t> Main::readMNISTLabels(const string& filename) {
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
int32_t Main::readInt(std::ifstream &file) {
    int32_t result = 0;
    file.read(reinterpret_cast<char*>(&result), 4);
    return __builtin_bswap32(result);  // For big-endian to little-endian
}
// -----------------------------------------------------------

// Main class public functions -------------------------------
void const Main::printArguments() {
    for (int i = 0; i < this->argc; ++i) {
        cout << "Argument " << i << ": " << this->argv[i] << endl;
    }
}
void const Main::printImage(const int& index) {
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
void Main::MNIST(const string& imageFile, const string& labelFile) {
    string folder = "data/";
    this->imageFile = imageFile;
    this->labelFile = labelFile;

    this->imagePath = folder + imageFile;
    this->labelPath = folder + labelFile;

    // Read images
    this->images = this->readMNISTImages(this->imagePath);

    // Read labels
    this->labels = this->readMNISTLabels(this->labelPath);
}
// -----------------------------------------------------------

// Main function taking in arguments -------------------------
int main(int argc, char **argv) {
    Main main(argc, argv);

    main.MNIST("train-images-idx3-ubyte", "train-labels-idx1-ubyte");
    return 0;
}
// -----------------------------------------------------------