#include <random>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>

#include <anon/graph/complete-graph.hxx>
#include <anon/graph/multicut/kernighan-lin.hxx>
#include <anon/graph/components.hxx>
#include <anon/graph/partition-comparison.hxx>


int main() {
    typedef anon::graph::CompleteGraph<> Graph;
    typedef anon::RandError<double> RandError;
    typedef anon::VariationOfInformation<double> VI;

    std::string filePath = "../../training/models/similarity/split-3600/analysis/siamese_outputs_unseen_only.csv";


    std::ifstream fin;
    fin.open(filePath);
    // pre read header

    std::string line;
    fin >> line;

    size_t tp= 0;
    size_t count = 0;

    std::cout << "Reading in file..." << std::endl;
    while (!fin.eof()){
        fin >> line;

        std::stringstream  ss(line);
        std::vector<std::string> values;

        std::string w;
        for (auto x : line){
            if ((x == ',') || (x == '\n')){
                values.push_back(w);
                w = "";
            } else {
                w = w + x;
            }
        }
        values.push_back(w);

        int fromIndex = std::stoi(values[0]);
        int toIndex = std::stoi(values[1]);
        if (fromIndex == toIndex) continue;

        count += 1;
        float pred = std::stof(values[2]);

        size_t pred_label;
        if (pred < 0.5){
            pred_label = 0;
        } else {
            pred_label = 1;
        }

        size_t groundTruth = std::stoi(values[3]);

        if (pred_label == groundTruth){
            tp += 1;
        }
    }

    std::cout << (double)tp / (double)count;


    return 0;
}
