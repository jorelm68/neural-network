#include <iostream>
#include <string>
#include <vector>

using namespace std;

// Main class ------------------------------------------------
class Main {
    public:
        Main(int argc, char **argv);
        ~Main();
        void printArguments();

    private:
        int argc;
        char **argv;
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

// Main class member functions -------------------------------
void Main::printArguments() {
    for (int i = 0; i < this->argc; ++i) {
        cout << "Argument " << i << ": " << this->argv[i] << endl;
    }
}
// -----------------------------------------------------------

// Main function taking in arguments -------------------------
int main(int argc, char **argv) {
    Main main(argc, argv);

    main.printArguments();
    return 0;
}
// -----------------------------------------------------------