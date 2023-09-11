#include <random>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

#include <anon/graph/complete-graph.hxx>
#include <anon/graph/partition-comparison.hxx>

int main() {
    typedef anon::graph::CompleteGraph<> Graph;
    typedef anon::RandError<double> RandError;
    typedef anon::VariationOfInformation<double> VI;

    std::string filePath = "../../training/models/similarity/split-3600/analysis/classifier_outputs_test_all.csv";

    std::ifstream fin;
    fin.open(filePath);
    // pre read header

    std::string line;
    std::getline(fin, line);

    std::vector<int> predictions;
    std::vector<int> groundTruths;

    std::cout << "Reading in file..." << std::endl;
    while (std::getline(fin, line)){

        std::stringstream  ss(line);
        std::vector<std::string> values;

        std::string w;
        for (auto x : line){
            if ((x == ',') || (x == '\n') || (x == '\r')){
                values.push_back(w);
                w = "";
            } else {
                w = w + x;
            }
        }
        values.push_back(w);

        int prediction = std::stoi(values[1]);
        int truth = std::stoi(values[2]);

        predictions.push_back(prediction);
        groundTruths.push_back(truth);
    }

    size_t count = 0;
    size_t tp = 0;

    for (size_t index = 0; index < predictions.size(); ++index){
        if (predictions[index] == groundTruths[index]){
            tp += 1;
        }
        count += 1;
    }

    std::cout << "Classification Accuracy: " << (double)tp / (double)count << std::endl;


    // get TP, TN, FP, FN
    std::cout << "Using Partition Comparison Classes..." << std::endl;

    RandError randError(groundTruths.begin(), groundTruths.end(), predictions.begin(), false);
    VI vi(groundTruths.begin(), groundTruths.end(), predictions.begin(), false);

    std::cout << "Rand Metrics" << std::endl;
    std::cout << std::string (20, '=') << std::endl;
    std::cout << "Precision Cuts: " << randError.precisionOfCuts() << std::endl;
    std::cout << "Recall Cuts: " << randError.recallOfCuts() << std::endl;
    std::cout << "Precision Joins: " << randError.precisionOfJoins() << std::endl;
    std::cout << "Recall Joins: " << randError.recallOfJoins() << std::endl;
    std::cout << "Rand Index: " << randError.index() << std::endl;
    std::cout << "Rand Error: " << randError.error() << std::endl;

    std::cout << "False Joins (Rand class): " << randError.falseJoins() << std::endl;
    std::cout << "False Cuts (Rand class): " << randError.falseCuts() << std::endl;

    std::cout << std::endl;
    std::cout << "VI Metrics" << std::endl;
    std::cout << std::string (20, '=') << std::endl;
    std::cout << "VI: " << vi.value() << std::endl;
    std::cout << "VI False Cuts: " << vi.valueFalseCut() << std::endl;
    std::cout << "VI False Joins: " << vi.valueFalseJoin() << std::endl;





    return 0;
}
