#include <random>
#include <vector>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iostream>

#include <anon/graph/complete-graph.hxx>
#include <anon/graph/multicut/greedy-additive.hxx>
#include <anon/graph/multicut/kernighan-lin.hxx>
#include <anon/graph/partition-comparison.hxx>
#include <anon/graph/multicut/edge-label-mask.hxx>
#include <anon/graph/components.hxx>

template<class T>
using CutSolutionMask = anon::graph::multicut::CutSolutionMask<T>;


int main() {
    typedef anon::graph::CompleteGraph<> Graph;
    typedef anon::RandError<double> RandError;
    typedef anon::VariationOfInformation<double> VI;

    std::string const filePath = "../../training/models/similarity/split-3600/analysis/siamese_outputs_test_set_only.csv";
    std::string const classesFilePath = "../../training/models/similarity/split-3600/analysis/siamese_outputs_test_set_only_classes.csv";

    std::string outFilePath = filePath;
    outFilePath.replace(filePath.find(".csv"), std::string(".csv").length(), "_outLog.txt");
    std::ofstream outFile(outFilePath);

    std::string primaryClass = "TEST";
    std::string secondaryClass = "TRAIN";

    std::vector<std::string> classSubsets;
    std::vector<std::string> classLabels;
    std::string line;

    std::ifstream fclasses;
    fclasses.open(classesFilePath);

    std::getline(fclasses, line);

    while (std::getline(fclasses, line)) {
        std::vector<std::string> values;

        std::string w;
        for (auto x: line) {
            if ((x == ',') || (x == '\n') || (x == '\r')) {
                values.push_back(w);
                w = "";
            } else {
                w += x;
            }
        }
        size_t index = std::stoi(values[0]);

        assert(index == classSubsets.size());

        classSubsets.push_back(values[2]);
        classLabels.push_back(values[1]);
    }


    std::ifstream fin;
    fin.open(filePath);
    // pre read header

    std::getline(fin, line);

    std::vector<int> fromIndices;
    std::vector<int> toIndices;
    std::vector<double> predictions;
    std::vector<size_t> groundTruths;

    // [0] -> class-class
    // [1] -> class-mixin
    // [2] -> mixin-mixin
    std::vector<size_t> trueJoinsSubsets(3, 0);
    std::vector<size_t> trueCutsSubsets(3, 0);
    std::vector<size_t> falseJoinsSubsets(3, 0);
    std::vector<size_t> falseCutsSubsets(3, 0);
//    std::vector<size_t> subsetClass;

    int maxIndex = 0;

    size_t trueJoins = 0;
    size_t trueCuts = 0;
    size_t falseJoins = 0;
    size_t falseCuts = 0;

    std::cout << "Reading in file..." << std::endl;
    while (std::getline(fin, line)) {
//        fin >> line;

        std::stringstream ss(line);
        std::vector<std::string> values;

        std::string w;
        for (auto x: line) {
            if ((x == ',') || (x == '\n') || (x == '\r')) {
                values.push_back(w);
                w = "";
            } else {
                w = w + x;
            }
        }
        values.push_back(w);

        int fromIndex = std::stoi(values[0]);
        int toIndex = std::stoi(values[1]);

        bool isFirstMixin = classSubsets[fromIndex] == secondaryClass;
        bool isSecondMixin = classSubsets[toIndex] == secondaryClass;

        maxIndex = std::max(maxIndex, fromIndex);
        maxIndex = std::max(maxIndex, toIndex);

        double pred = std::stod(values[2]);

        predictions.push_back(pred);

        size_t groundTruth = std::stoi(values[3]);

        fromIndices.push_back(fromIndex);
        toIndices.push_back(toIndex);
        groundTruths.push_back(groundTruth);
    }
    fin.close();

    outFile << "Maximum Index: " << maxIndex << std::endl;

    outFile << "Building Graph..." << std::endl;

    Graph graph(maxIndex + 1);


    std::cout << "Populating Edge Costs..." << std::endl;
    std::vector<double> edgeCosts(graph.numberOfEdges(), 0);
    std::vector<size_t> truePartition(graph.numberOfEdges(), std::numeric_limits<size_t>::max());

    std::vector<double> edgePredictionsAccumulated(graph.numberOfEdges());

    for (int i = 0; i < fromIndices.size(); ++i) {
        int fromIndex = fromIndices[i];
        int toIndex = toIndices[i];
        if (fromIndex == toIndex) {
            // no edge for this
            continue;
        }
        double pred = predictions[i];
        size_t gt = 1 - groundTruths[i];

        auto edge = graph.findEdge(fromIndex, toIndex);
        if (edge.first) {
            // add cost
            edgePredictionsAccumulated[edge.second] += pred / 2;
            if (truePartition[edge.second] != std::numeric_limits<size_t >::max() && truePartition[edge.second] != gt){
                std::cout << "Stored: " << truePartition[edge.second] << ", but want to set " << gt << std::endl;
                throw std::runtime_error("Conflicting ground truth detected.");
            } else {
                truePartition[edge.second] = gt;
            }
        } else {
            throw std::runtime_error("Edge not found.");
        }
    }

    for (int i = 0; i < fromIndices.size(); ++i) {
        int fromIndex = fromIndices[i];
        int toIndex = toIndices[i];
        if (fromIndex == toIndex) {
            continue;
        }

        auto edge = graph.findEdge(fromIndex, toIndex);
        if (edge.first) {
            double pred = edgePredictionsAccumulated[edge.second];

            size_t predLabel;
            if (pred < 0.5){
                predLabel = 0;
            } else {
                predLabel = 1;
            }

            bool isFirstMixin = classSubsets[fromIndex] == secondaryClass;
            bool isSecondMixin = classSubsets[toIndex] == secondaryClass;
            size_t groundTruth = groundTruths[i];

            if (predLabel == 0 && groundTruth == 0) {
                trueCuts += 1;
                if (!isFirstMixin && !isSecondMixin) {
                    trueCutsSubsets[0] += 1;
                } else if (isFirstMixin && isSecondMixin) {
                    trueCutsSubsets[2] += 1;
                } else {
                    trueCutsSubsets[1] += 1;
                }
            } else if (predLabel == 0 && groundTruth == 1) {
                falseCuts += 1;
                if (!isFirstMixin && !isSecondMixin) {
                    falseCutsSubsets[0] += 1;
                } else if (isFirstMixin && isSecondMixin) {
                    falseCutsSubsets[2] += 1;
                } else {
                    falseCutsSubsets[1] += 1;
                }
            } else if (predLabel == 1 && groundTruth == 0) {
                falseJoins += 1;
                if (!isFirstMixin && !isSecondMixin) {
                    falseJoinsSubsets[0] += 1;
                } else if (isFirstMixin && isSecondMixin) {
                    falseJoinsSubsets[2] += 1;
                } else {
                    falseJoinsSubsets[1] += 1;
                }
            } else if (predLabel == 1 && groundTruth == 1) {
                trueJoins += 1;
                if (!isFirstMixin && !isSecondMixin) {
                    trueJoinsSubsets[0] += 1;
                } else if (isFirstMixin && isSecondMixin) {
                    trueJoinsSubsets[2] += 1;
                } else {
                    trueJoinsSubsets[1] += 1;
                }
            } else {
                throw std::runtime_error("Ground truth or predicted label is not 0 or 1.");
            }
            double cost = std::min(1.0f - 1e-6, std::max(1e-6, pred));
            // if pred -> 0 , cost gets high positive -> want to cut -> high reward for cutting this edge -> high negative cost for MC problem
            // if pred -> 1 , cost gets high negative -> want to join -> high positive penalty for cutting this edge -> high positive cost for MC problem
            cost = std::log((1 - pred) / pred);
            edgeCosts[edge.second] = -cost;
        } else {
            throw std::runtime_error("Edge not found.");
        }
    }

    outFile << "True Cuts (Total): " << trueCuts << std::endl;
    outFile << "True Joins (Total): " << trueJoins << std::endl;
    outFile << "False Cuts (Total): " << falseCuts << std::endl;
    outFile << "False Joins (Total): " << falseJoins << std::endl;

    outFile << "True Cuts (" << primaryClass << "-" << primaryClass << "): " << trueCutsSubsets[0] << std::endl;
    outFile << "True Joins (" << primaryClass << "-" << primaryClass << "): " << trueJoinsSubsets[0] << std::endl;
    outFile << "False Cuts (" << primaryClass << "-" << primaryClass << "): " << falseCutsSubsets[0] << std::endl;
    outFile << "False Joins (" << primaryClass << "-" << primaryClass << "): " << falseJoinsSubsets[0] << std::endl;

    outFile << "True Cuts (" << primaryClass << "-" << secondaryClass << "): " << trueCutsSubsets[1] << std::endl;
    outFile << "True Joins (" << primaryClass << "-" << secondaryClass << "): " << trueJoinsSubsets[1] << std::endl;
    outFile << "False Cuts (" << primaryClass << "-" << secondaryClass << "): " << falseCutsSubsets[1] << std::endl;
    outFile << "False Joins (" << primaryClass << "-" << secondaryClass << "): " << falseJoinsSubsets[1] << std::endl;

    outFile << "True Cuts (" << secondaryClass << "-" << secondaryClass << "): " << trueCutsSubsets[2] << std::endl;
    outFile << "True Joins (" << secondaryClass << "-" << secondaryClass << "): " << trueJoinsSubsets[2] << std::endl;
    outFile << "False Cuts (" << secondaryClass << "-" << secondaryClass << "): " << falseCutsSubsets[2] << std::endl;
    outFile << "False Joins (" << secondaryClass << "-" << secondaryClass << "): " << falseJoinsSubsets[2] << std::endl;


    outFile << "Rounding Precision / Recalls" << std::endl;
    outFile << std::string(50, '=') << std::endl;

    outFile << "Precision Cuts (wo CC): " << (double)trueCuts / ((double)trueCuts + (double)falseCuts) << std::endl;
    outFile << "Recall Cuts (wo CC): " << (double)trueCuts / ((double)trueCuts + (double)falseJoins) << std::endl;
    outFile << "Precision Joins (wo CC): " << (double)trueJoins / ((double)trueJoins + (double)falseJoins) << std::endl;
    outFile << "Recall Joins (wo CC): " << (double)trueJoins / ((double)trueJoins + (double)falseCuts) << std::endl;

    outFile << std::string(50, '=') << std::endl;

    outFile << "Rounding Precision / Recalls - " << primaryClass << " / " << primaryClass << std::endl;
    outFile << "Precision Cuts - Class / Class (wo CC): "
            << (double)trueCutsSubsets[0] / ((double)trueCutsSubsets[0] + (double)falseCutsSubsets[0]) << std::endl;
    outFile << "Recall Cuts (wo CC): " << (double)trueCutsSubsets[0] / ((double)trueCutsSubsets[0] + (double)falseJoinsSubsets[0]) << std::endl;
    outFile << "Precision Joins (wo CC): " << (double)trueJoinsSubsets[0] / ((double)trueJoinsSubsets[0] + (double)falseJoinsSubsets[0])
            << std::endl;
    outFile << "Recall Joins (wo CC): " << (double)trueJoinsSubsets[0] / ((double)trueJoinsSubsets[0] + (double)falseCutsSubsets[0])
            << std::endl;
    outFile << std::endl;

    outFile << "Rounding Precision / Recalls - " << primaryClass << " / " << secondaryClass << std::endl;
    outFile << "Precision Cuts - Class / Class (wo CC): "
            << (double)trueCutsSubsets[1] / ((double)trueCutsSubsets[1] + (double)falseCutsSubsets[1]) << std::endl;
    outFile << "Recall Cuts (wo CC): " << (double)trueCutsSubsets[1] / ((double)trueCutsSubsets[1] + falseJoinsSubsets[1]) << std::endl;
    outFile << "Precision Joins (wo CC): " << (double)trueJoinsSubsets[1] / ((double)trueJoinsSubsets[1] + (double)falseJoinsSubsets[1])
            << std::endl;
    outFile << "Recall Joins (wo CC): " << (double)trueJoinsSubsets[1] / ((double)trueJoinsSubsets[1] + (double)falseCutsSubsets[1])
            << std::endl;
    outFile << std::endl;

    outFile << "Rounding Precision / Recalls - " << secondaryClass << " / " << secondaryClass << std::endl;
    outFile << "Precision Cuts - Class / Class (wo CC): "
            << (double)trueCutsSubsets[2] / ((double)trueCutsSubsets[2] + (double)falseCutsSubsets[2]) << std::endl;
    outFile << "Recall Cuts (wo CC): " << (double)trueCutsSubsets[2] / ((double)trueCutsSubsets[2] + (double)falseJoinsSubsets[2]) << std::endl;
    outFile << "Precision Joins (wo CC): " << (double)trueJoinsSubsets[2] / ((double)trueJoinsSubsets[2] + (double)falseJoinsSubsets[2])
            << std::endl;
    outFile << "Recall Joins (wo CC): " << (double)trueJoinsSubsets[2] / ((double)trueJoinsSubsets[2] + (double)falseCutsSubsets[2])
            << std::endl;
    outFile << std::endl;



//    size_t gt = groundTruths[i];

    CutSolutionMask<size_t> truthMask(truePartition);
    anon::graph::ComponentsBySearch<Graph> componentsBySearch;
    componentsBySearch.build(graph, truthMask);

    std::vector<size_t> truthLabels = componentsBySearch.labels_;


    // write out truth label classes
    std::map<size_t, std::string> truthLabelClasses;
    for (size_t i = 0; i < truthLabels.size(); ++i) {
        size_t truthLabelId = truthLabels[i];
        std::string truthLabelClass = classLabels[i];
        if (!truthLabelClasses[truthLabelId].empty() && truthLabelClasses[truthLabelId] != truthLabelClass) {
            std::cout << "Attempted to set truth label to " << truthLabelClass << " while it was already set to " << truthLabelClass[truthLabelId] << std::endl;
            throw std::runtime_error("Conflicting truth label classes.");
        } else {
            truthLabelClasses[truthLabelId] = truthLabelClass;
        }
    }

    for (auto &pair: truthLabelClasses) {
        std::cout << "Truth Cluster " << pair.first << " belongs to class " << pair.second << std::endl;
    }

    std::string truthClusterNamesFilePath = filePath;
    truthClusterNamesFilePath.replace(truthClusterNamesFilePath.find(".csv"), std::string(".csv").length(),
                                      "_truthClusterNames.csv");
    std::ofstream truthClusterNamesFile;
    truthClusterNamesFile.open(truthClusterNamesFilePath);

    // predExceptTruthClusterSize is the number of elements in predCluster and not in truthCluster
    // truthExceptPredClusterSize is the number of elements in truthCluster and not in predCluster
    truthClusterNamesFile << "truthClusterId,name\n";

    for (auto &pair: truthLabelClasses) {
        truthClusterNamesFile << pair.first << "," << pair.second << "\n";
    }

    truthClusterNamesFile.close();

//    for (auto & pair : majorityMatching){
//        matchingFile << pair.first << "," << pair.second << "," << predClusterSizes[pair.first] << "," << truthClusterSizes[pair.second] << "," << predictedToTruthClusterOverlap[pair.first][pair.second] << "\n";
//    }


    std::vector<std::size_t> edgeLabels(graph.numberOfEdges(), 1);

    std::cout << "Applying Additive Edge Contraction..." << std::endl;
    anon::graph::multicut::greedyAdditiveEdgeContractionCompleteGraph(
            graph,
            edgeCosts,
            edgeLabels
    );

    std::cout << "Applying Kernighan Lin..." << std::endl;
    anon::graph::multicut::kernighanLin(
            graph,
            edgeCosts,
            edgeLabels,
            edgeLabels
    );
    std::cout << "Applying Kernighan Lin done..." << std::endl;


    std::vector<size_t> trueJoinsSubsetsCC(3, 0);
    std::vector<size_t> trueCutsSubsetsCC(3, 0);
    std::vector<size_t> falseJoinsSubsetsCC(3, 0);
    std::vector<size_t> falseCutsSubsetsCC(3, 0);

    // get TP, TN, FP, FN

    // true cuts
    size_t tp = 0;
    // true joins
    size_t tn = 0;
    // false cuts
    size_t fp = 0;
    // false joins
    size_t fn = 0;

    for (int i = 0; i < fromIndices.size(); ++i) {
        int fromIndex = fromIndices[i];
        int toIndex = toIndices[i];

        if (fromIndex == toIndex) {
            continue;
        }

        // edgeLabel is inverted, 1 means cut; 0 means join
        size_t edgeLabel = 1 - edgeLabels[graph.findEdge(fromIndex, toIndex).second];
        size_t gt = groundTruths[i];

        bool isFirstMixin = classSubsets[fromIndex] == secondaryClass;
        bool isSecondMixin = classSubsets[toIndex] == secondaryClass;
        size_t index;
        if (isFirstMixin && isSecondMixin){
            index = 2;
        } else if (!isFirstMixin && !isSecondMixin){
            index = 0;
        } else{
            index = 1;
        }

        if (edgeLabel == 1 && gt == 1) {
            tp += 1;
            trueJoinsSubsetsCC[index] += 1;
        } else if (edgeLabel == 1 && gt == 0) {
            fp += 1;
            falseJoinsSubsetsCC[index] += 1;
        } else if (edgeLabel == 0 && gt == 1) {
            fn += 1;
            falseCutsSubsetsCC[index] += 1;
        } else {
            tn += 1;
            trueCutsSubsetsCC[index] += 1;
        }
    }
    tp = tp / 2;
    fp = fp / 2;
    tn = tn / 2;
    fn = fn / 2;

    outFile << "True Cuts CC (" << primaryClass << "-" << primaryClass << "): " << trueCutsSubsetsCC[0] << std::endl;
    outFile << "True Joins CC (" << primaryClass << "-" << primaryClass << "): " << trueJoinsSubsetsCC[0] << std::endl;
    outFile << "False Cuts CC (" << primaryClass << "-" << primaryClass << "): " << falseCutsSubsetsCC[0] << std::endl;
    outFile << "False Joins CC (" << primaryClass << "-" << primaryClass << "): " << falseJoinsSubsetsCC[0] << std::endl;

    outFile << "True Cuts CC (" << primaryClass << "-" << secondaryClass << "): " << trueCutsSubsetsCC[1] << std::endl;
    outFile << "True Joins CC (" << primaryClass << "-" << secondaryClass << "): " << trueJoinsSubsetsCC[1] << std::endl;
    outFile << "False Cuts CC (" << primaryClass << "-" << secondaryClass << "): " << falseCutsSubsetsCC[1] << std::endl;
    outFile << "False Joins CC (" << primaryClass << "-" << secondaryClass << "): " << falseJoinsSubsetsCC[1] << std::endl;

    outFile << "True Cuts CC (" << secondaryClass << "-" << secondaryClass << "): " << trueCutsSubsetsCC[2] << std::endl;
    outFile << "True Joins CC (" << secondaryClass << "-" << secondaryClass << "): " << trueJoinsSubsetsCC[2] << std::endl;
    outFile << "False Cuts CC (" << secondaryClass << "-" << secondaryClass << "): " << falseCutsSubsetsCC[2] << std::endl;
    outFile << "False Joins CC (" << secondaryClass << "-" << secondaryClass << "): " << falseJoinsSubsetsCC[2] << std::endl;


    outFile << std::endl;
    outFile << "Rounding Precision / Recalls - " << primaryClass << " / " << primaryClass << " - AFTER CC" << std::endl;
    outFile << std::string(50, '=') << std::endl;
    outFile << "Precision Cuts - Class / Class (CC): "
            << (double)trueCutsSubsetsCC[0] / ((double)trueCutsSubsetsCC[0] + (double)falseCutsSubsetsCC[0]) << std::endl;
    outFile << "Recall Cuts (CC): " << (double)trueCutsSubsetsCC[0] / ((double)trueCutsSubsetsCC[0] + (double)falseJoinsSubsetsCC[0])
            << std::endl;
    outFile << "Precision Joins (CC): " << (double)trueJoinsSubsetsCC[0] / ((double)trueJoinsSubsetsCC[0] + (double)falseJoinsSubsetsCC[0])
            << std::endl;
    outFile << "Recall Joins (CC): " << (double)trueJoinsSubsetsCC[0] / ((double)trueJoinsSubsetsCC[0] + (double)falseCutsSubsetsCC[0])
            << std::endl;
    outFile << std::endl;

    outFile << "Rounding Precision / Recalls - " << primaryClass << " / " << secondaryClass << " - AFTER CC"
            << std::endl;
    outFile << "Precision Cuts - Class / Class (CC): "
            << (double)trueCutsSubsetsCC[1] / ((double)trueCutsSubsetsCC[1] + (double)falseCutsSubsetsCC[1]) << std::endl;
    outFile << "Recall Cuts (CC): " << (double)trueCutsSubsetsCC[1] / ((double)trueCutsSubsetsCC[1] + (double)falseJoinsSubsetsCC[1])
            << std::endl;
    outFile << "Precision Joins (CC): " << (double)trueJoinsSubsetsCC[1] / ((double)trueJoinsSubsetsCC[1] + (double)falseJoinsSubsetsCC[1])
            << std::endl;
    outFile << "Recall Joins (CC): " <<(double) trueJoinsSubsetsCC[1] / ((double)trueJoinsSubsetsCC[1] + (double)falseCutsSubsetsCC[1])
            << std::endl;
    outFile << std::endl;

    outFile << "Rounding Precision / Recalls - " << secondaryClass << " / " << secondaryClass << " - AFTER CC"
            << std::endl;
    outFile << "Precision Cuts - Class / Class (CC): "
            << (double)trueCutsSubsetsCC[2] / ((double)trueCutsSubsetsCC[2] + (double)falseCutsSubsetsCC[2]) << std::endl;
    outFile << "Recall Cuts (CC): " << (double)trueCutsSubsetsCC[2] / ((double)trueCutsSubsetsCC[2] + (double)falseJoinsSubsetsCC[2])
            << std::endl;
    outFile << "Precision Joins (CC): " << (double)trueJoinsSubsetsCC[2] / ((double)trueJoinsSubsetsCC[2] + (double)falseJoinsSubsetsCC[2])
            << std::endl;
    outFile << "Recall Joins (CC): " << (double)trueJoinsSubsetsCC[2] / ((double)trueJoinsSubsetsCC[2] + (double)falseCutsSubsetsCC[2])
            << std::endl;
    outFile << std::endl;

    CutSolutionMask<size_t> edgeLabelsMask(edgeLabels);
    componentsBySearch.build(graph, edgeLabelsMask);

    std::vector<size_t> edgeNodeLabels = componentsBySearch.labels_;


    std::vector<size_t > truthLabelsPrimary;
    std::vector<size_t > predictedLabelsPrimary;
    for (size_t i = 0; i < classSubsets.size(); ++i){
        if (classSubsets[i] == primaryClass){
            truthLabelsPrimary.push_back(truthLabels[i]);
            predictedLabelsPrimary.push_back(edgeNodeLabels[i]);
        }
    }

    RandError randErrorPrimary(truthLabelsPrimary.begin(), truthLabelsPrimary.end(), predictedLabelsPrimary.begin(), false);
    VI viPrimary(truthLabelsPrimary.begin(), truthLabelsPrimary.end(), predictedLabelsPrimary.begin(), false);
    outFile << std::endl;
    outFile << "Rand Metrics " << primaryClass << std::endl;
    outFile << std::string(20, '=') << std::endl;
    outFile << "Precision Cuts: " << randErrorPrimary.precisionOfCuts() << std::endl;
    outFile << "Recall Cuts: " << randErrorPrimary.recallOfCuts() << std::endl;
    outFile << "Precision Joins: " << randErrorPrimary.precisionOfJoins() << std::endl;
    outFile << "Recall Joins: " << randErrorPrimary.recallOfJoins() << std::endl;
    outFile << "Rand Index: " << randErrorPrimary.index() << std::endl;
    outFile << "Rand Error: " << randErrorPrimary.error() << std::endl;

    outFile << std::endl;
    outFile << "VI Metrics" << primaryClass << std::endl;
    outFile << std::string(20, '=') << std::endl;
    outFile << "VI: " << viPrimary.value() << std::endl;
    outFile << "VI False Cuts: " << viPrimary.valueFalseCut() << std::endl;
    outFile << "VI False Joins: " << viPrimary.valueFalseJoin() << std::endl;

    outFile << std::endl;

    // calculate matchings
    std::map<size_t, std::map<size_t, size_t>> predictedToTruthClusterOverlap;
    std::map<size_t, std::map<size_t, size_t >> truthToPredictedClusterOverlaps;

    std::map<size_t, size_t> predClusterSizes;
    std::map<size_t, size_t> truthClusterSizes;

    for (auto &nodeLabel: edgeNodeLabels) {
        predictedToTruthClusterOverlap[nodeLabel] = std::map<size_t, size_t>();
        predClusterSizes[nodeLabel] += 1;
    }

    for (auto &nodeLabel: truthLabels) {
        truthToPredictedClusterOverlaps[nodeLabel] = std::map<size_t, size_t>();
        truthClusterSizes[nodeLabel] += 1;
    }

    // iterates over all node indices in the solution of the CC problem
    for (size_t nodeIndex = 0; nodeIndex < edgeNodeLabels.size(); ++nodeIndex) {
        predictedToTruthClusterOverlap[edgeNodeLabels[nodeIndex]][truthLabels[nodeIndex]] += 1;
        truthToPredictedClusterOverlaps[truthLabels[nodeIndex]][edgeNodeLabels[nodeIndex]] += 1;
    }


    // matches the predicted cluster to the majority label
    std::map<size_t, size_t> majorityMatching;

    for (auto &pair: predictedToTruthClusterOverlap) {
        size_t nodeLabel = pair.first;
        std::map<size_t, size_t> &sizes = pair.second;

        std::map<size_t, size_t>::iterator best
                = std::max_element(sizes.begin(), sizes.end(), [](const std::pair<size_t, size_t> &a,
                                                                  const std::pair<size_t, size_t> &b) -> bool {
                    return a.second < b.second;
                });

        majorityMatching[nodeLabel] = best->first;
    }

    // get truth cluster labels
    std::map<size_t , std::string> truthClusterLabels;
    for (size_t index = 0; index < truthLabels.size(); ++index){
        std::string classLabel = classLabels[index];
        if (!truthClusterLabels[truthLabels[index]].empty() && truthClusterLabels[truthLabels[index]] != classLabel){
            throw std::runtime_error("Error with truth clusters.");
        }
        truthClusterLabels[truthLabels[index]] = classLabel;
    }

    size_t count = 0;
    // TP, TN, FP, FN
    std::vector<size_t > classificationMetrics {0, 0, 0, 0};
    // iterate over all edgeNodeLabels
    for (size_t i = 0; i <  edgeNodeLabels.size(); ++i){
        if (classSubsets[i] != primaryClass) continue;

        count += 1;
        size_t edgeNodeLabel = edgeNodeLabels[i];
        std::string predictedClass = truthLabelClasses[majorityMatching[edgeNodeLabel]];
        std::string actualClass = classLabels[i];
        if (predictedClass == actualClass){
            // one tp
            classificationMetrics[0] += 1;
            // all other labels +1 TN
            classificationMetrics[1] += truthLabelClasses.size() - 1;
        } else {
            // one fp for the predictedClass
            classificationMetrics[2] += 1;
            // one fn for the actualClass
            classificationMetrics[3] += 1;
            // n - 2 TN's for all other classes
            classificationMetrics[1] += truthLabels.size() - 2;
        }
    }

//    std::cout << "Classification Recall: " << (double) classificationMetrics[0] / ((double) classificationMetrics[0] + (double) classificationMetrics[3]) << std::endl;
//    std::cout << "Classification Precision: " << (double) classificationMetrics[0] / ((double) classificationMetrics[0] + (double) classificationMetrics[2]) << std::endl;
    std::cout << "Classification Accuracy: " << (double) classificationMetrics[0] / (double)count << std::endl;

    // also store the translation of truth clusters and class name together with the firstClass / secondClass flag
    std::string clusterMatchingPath = filePath;
    clusterMatchingPath.replace(clusterMatchingPath.find(".csv"), std::string(".csv").length(), "_clusterMatching.csv");

    std::cout << clusterMatchingPath << std::endl;

    std::cout << "Number of Truth Clusters: " << truthClusterSizes.size() << std::endl;

//    for (auto & pair : majorityMatching){
//        std::cout << predictedToTruthClusterOverlap[pair.first][pair.second] << " - " << pair.first << "->" << pair.second << " - " << truthClusterSizes[pair.second] << std::endl;
//    }
    std::ofstream matchingFile;
    matchingFile.open(clusterMatchingPath);

    // predExceptTruthClusterSize is the number of elements in predCluster and not in truthCluster
    // truthExceptPredClusterSize is the number of elements in truthCluster and not in predCluster
    matchingFile
            << "predCluster,truthCluster,predClusterSize,truthClusterSize,overlap,predExceptTruthClusterSize,truthExceptPredClusterSize\n";

    for (auto &predLabel: edgeNodeLabels) {
        for (auto &pair: predictedToTruthClusterOverlap[predLabel]) {
            size_t truthLabel = pair.first;
            size_t predTruthOverlap = pair.second;
            size_t truthPredOverlap = truthToPredictedClusterOverlaps[truthLabel][predLabel];
            assert(predTruthOverlap == truthPredOverlap);
            size_t truthClusterSize = truthClusterSizes[truthLabel];
            size_t predClusterSize = predClusterSizes[predLabel];
            size_t predExceptTruthClusterSize = predClusterSize - truthPredOverlap;
            size_t truthExceptPredClusterSize = truthClusterSize - truthPredOverlap;
            matchingFile << predLabel << "," << truthLabel << "," << predClusterSize << "," << truthClusterSize << ","
                         << predTruthOverlap << "," << predExceptTruthClusterSize << "," << truthExceptPredClusterSize
                         << "\n";
        }
    }

    matchingFile.close();

    // calculate cumulative distribution / predicted clusters
    std::vector<size_t> predClusterSizesSorted(predClusterSizes.size());
    for (auto &pair: predClusterSizes) {
        predClusterSizesSorted[pair.first] = pair.second;
    }
    std::sort(predClusterSizesSorted.begin(), predClusterSizesSorted.end(), std::less_equal<>());
    std::string cumDistributionPredClusterSizesPath = filePath;
    cumDistributionPredClusterSizesPath.replace(cumDistributionPredClusterSizesPath.find(".csv"),
                                                std::string(".csv").length(), "_cumDistrPredClusterSizes.csv");

    std::ofstream cumDistrPredClustersFile;
    cumDistrPredClustersFile.open(cumDistributionPredClusterSizesPath);
    cumDistrPredClustersFile << "clusterSize,numClusters\n";
    for (size_t i = 0; i < predClusterSizesSorted.size(); ++i) {
        cumDistrPredClustersFile << predClusterSizesSorted[i] << "," << i << "\n";
    }
    cumDistrPredClustersFile.close();

    // calculate cumulative distribution / predicted clusters
    std::vector<size_t> truthClusterSizesSorted(truthClusterSizes.size());
    for (auto &pair: truthClusterSizes) {
        truthClusterSizesSorted[pair.first] = pair.second;
    }
    std::sort(truthClusterSizesSorted.begin(), truthClusterSizesSorted.end(), std::less_equal<>());
    std::string cumDistributionTruthClusterSizesPath = filePath;
    cumDistributionTruthClusterSizesPath.replace(cumDistributionTruthClusterSizesPath.find(".csv"),
                                                 std::string(".csv").length(), "_cumDistributionTruthClusterSizes.csv");

    std::ofstream cumDistributionTruthClustersFile;
    cumDistributionTruthClustersFile.open(cumDistributionTruthClusterSizesPath);
    cumDistributionTruthClustersFile << "clusterSize,numClusters\n";
    for (size_t i = 0; i < truthClusterSizesSorted.size(); ++i) {
        cumDistributionTruthClustersFile << truthClusterSizesSorted[i] << "," << i << "\n";
    }
    cumDistributionTruthClustersFile.close();


    // tex file writing
    std::string cumClustersTexPath = filePath;
    cumClustersTexPath.replace(cumClustersTexPath.find(".csv"), std::string(".csv").length(), "_cumClusterSizes.tex");

    std::ofstream cumClustersTexFile;

    size_t numberOfPredClusters = predClusterSizesSorted.size();
    size_t numberOfTrueClusters = truthClusterSizesSorted.size();

    cumClustersTexFile.open(cumClustersTexPath);
    cumClustersTexFile << "\\documentclass{standalone}\n"
                       << "\\usepackage{pgfplots}\n"
                       << "\\pgfplotsset{compat=newest}\n"
                       << "\\begin{document}\n"
                       << "\\begin{tikzpicture}\n";
    cumClustersTexFile
            << "\\begin{axis}[xlabel={ClusterSize}, ylabel={Fraction of Clusters}, legend pos={south east}]\n";

    // predicted cluster sizes
    cumClustersTexFile << "\\addplot[green] table[x index=0, y index=1, col sep=comma] {";

    for (size_t index = 0; index < numberOfPredClusters; ++index) {
        cumClustersTexFile << predClusterSizesSorted[index] << ","
                           << static_cast<double>(index + 1) / static_cast<double>(numberOfPredClusters) << std::endl;
    }

    cumClustersTexFile << "};\n";

    // truth cluster sizes
    cumClustersTexFile << "\\addplot[red] table[x index=0, y index=1, col sep=comma] {";

    for (size_t index = 0; index < numberOfTrueClusters; ++index) {
        cumClustersTexFile << truthClusterSizesSorted[index] << ","
                           << static_cast<double>(index + 1) / static_cast<double>(numberOfTrueClusters) << std::endl;
    }

    cumClustersTexFile << "};\n";
    cumClustersTexFile << "\\legend{predicted, truth}\n";

    cumClustersTexFile << "\\end{axis}\n"
                       << "\\end{tikzpicture}\n"
                       << "\\end{document}";
    cumClustersTexFile.close();

    outFile << "TP: " << tp << std::endl;
    outFile << "TN: " << tn << std::endl;
    outFile << "FP: " << fp << std::endl;
    outFile << "FN: " << fn << std::endl;

    outFile << "Accuracy: " << ((double)tp + (double )tn) / ((double )tp + (double )tn + (double )fn + (double )fp) << std::endl;
    outFile << "Recall: " << (double )tp / ((double )tp + (double )fn) << std::endl;
    outFile << "Precision: " << (double )tp / ((double )tp + (double )fp) << std::endl;


    outFile << "Using Partition Comparison Classes..." << std::endl;

    RandError randError(truthLabels.begin(), truthLabels.end(), edgeNodeLabels.begin(), false);
    VI vi(truthLabels.begin(), truthLabels.end(), edgeNodeLabels.begin(), false);

    outFile << "Rand Metrics" << std::endl;
    outFile << std::string(20, '=') << std::endl;
    outFile << "Precision Cuts: " << randError.precisionOfCuts() << std::endl;
    outFile << "Recall Cuts: " << randError.recallOfCuts() << std::endl;
    outFile << "Precision Joins: " << randError.precisionOfJoins() << std::endl;
    outFile << "Recall Joins: " << randError.recallOfJoins() << std::endl;
    outFile << "Rand Index: " << randError.index() << std::endl;
    outFile << "Rand Error: " << randError.error() << std::endl;

    outFile << "False Joins (Rand class): " << randError.falseJoins() << std::endl;
    outFile << "False Joins (Self): " << fn << std::endl;

    outFile << "False Cuts (Rand class): " << randError.falseCuts() << std::endl;
    outFile << "False Cuts (Self): " << fp << std::endl;

    outFile << std::endl;
    outFile << "VI Metrics" << std::endl;
    outFile << std::string(20, '=') << std::endl;
    outFile << "VI: " << vi.value() << std::endl;
    outFile << "VI False Cuts: " << vi.valueFalseCut() << std::endl;
    outFile << "VI False Joins: " << vi.valueFalseJoin() << std::endl;

    outFile.close();

    return 0;
}
