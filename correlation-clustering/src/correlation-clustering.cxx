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

    std::string filePath = "../../training/models/similarity/split-3600/analysis/siamese_outputs_unseen_only.csv";
    std::string classesFilePath = "../../training/models/similarity/split-3600/analysis/siamese_outputs_unseen_only_classes.csv";

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

    int maxIndex = 0;

    size_t trueJoins = 0;
    size_t trueCuts = 0;
    size_t falseJoins = 0;
    size_t falseCuts = 0;

    std::cout << "Reading in file..." << std::endl;
    while (std::getline(fin, line)){

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

        maxIndex = std::max(maxIndex, fromIndex);
        maxIndex = std::max(maxIndex, toIndex);

        double pred = std::stod(values[2]);

        predictions.push_back(pred);

        size_t groundTruth = std::stoi(values[3]);

        fromIndices.push_back(fromIndex);
        toIndices.push_back(toIndex);
        groundTruths.push_back(groundTruth);
    }


    std:: cout << std::string (50, '=') << std::endl;

    std:: cout << "Maximum Index: " << maxIndex << std::endl;

    std::cout << "Building Graph..." << std::endl;

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

            size_t groundTruth = groundTruths[i];

            if (predLabel == 0 && groundTruth == 0) {
                trueCuts += 1;
            } else if (predLabel == 0 && groundTruth == 1) {
                falseCuts += 1;
            } else if (predLabel == 1 && groundTruth == 0) {
                falseJoins += 1;
            } else if (predLabel == 1 && groundTruth == 1) {
                trueJoins += 1;
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

    std::cout << "Precision Joins (wo CC): " << (double)trueJoins / ((double)trueJoins + (double)falseJoins) << std::endl;
    std::cout << "Recall Joins (wo CC): " << (double)trueJoins / ((double)trueJoins + (double)falseCuts) << std::endl;
    std::cout << "Precision Cuts (wo CC): " << (double)trueCuts / ((double)trueCuts + (double)falseCuts) << std::endl;
    std::cout << "Recall Cuts (wo CC): " << (double)trueCuts / ((double)trueCuts + (double)falseJoins) << std::endl;


    CutSolutionMask<size_t > truthMask(truePartition);
    anon::graph::ComponentsBySearch<Graph> componentsBySearch;
    componentsBySearch.build(graph, truthMask);

    std::vector<size_t > truthLabels = componentsBySearch.labels_;
    std::map<size_t , size_t > truthLabelSizes;
    for (auto & nodeLabel: truthLabels){
        truthLabelSizes[nodeLabel] += 1;
    }

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

    // get TP, TN, FP, FN

    // true cuts
    double tp = 0;
    // true joins
    double tn = 0;
    // false cuts
    double fp = 0;
    // false joins
    double fn = 0;

    for (int i = 0; i < fromIndices.size(); ++i){
        int fromIndex = fromIndices[i];
        int toIndex = toIndices[i];

        if (fromIndex == toIndex){
            continue;
        }

        // edgeLabel is inverted, 1 means cut; 0 means join
        size_t edgeLabel = 1 - edgeLabels[graph.findEdge(fromIndex, toIndex).second];
        size_t gt = groundTruths[i];

        if (edgeLabel == 1 && gt == 1){
            tp += 1;
        } else if (edgeLabel == 1 && gt == 0){
            fp += 1;
        } else if (edgeLabel == 0 && gt == 1){
            fn += 1;
        } else {
            tn += 1;
        }
    }
    tp = tp / 2;
    fp = fp / 2;
    tn = tn / 2;
    fn = fn / 2;

    CutSolutionMask<size_t > edgeLabelsMask(edgeLabels);
    componentsBySearch.build(graph, edgeLabelsMask);

    std::vector<size_t > edgeNodeLabels = componentsBySearch.labels_;

    // calculate matchings
    // calculate matchings
    std::map<size_t, std::map<size_t, size_t>> predictedToTruthClusterOverlap;
    std::map<size_t , std::map<size_t , size_t >> truthToPredictedClusterOverlaps;

    std::map<size_t , size_t > predClusterSizes;
    std::map<size_t , size_t > truthClusterSizes;

    for(auto & nodeLabel : edgeNodeLabels){
        predictedToTruthClusterOverlap[nodeLabel] = std::map<size_t , size_t >();
        predClusterSizes[nodeLabel] += 1;
    }

    for (auto & nodeLabel : truthLabels){
        truthToPredictedClusterOverlaps[nodeLabel] = std::map<size_t , size_t >();
        truthClusterSizes[nodeLabel] += 1;
    }

    // iterates over all node indices in the solution of the CC problem
    for (size_t nodeIndex = 0; nodeIndex < edgeNodeLabels.size(); ++nodeIndex){
        predictedToTruthClusterOverlap[edgeNodeLabels[nodeIndex]][truthLabels[nodeIndex]] += 1;
        truthToPredictedClusterOverlaps[truthLabels[nodeIndex]][edgeNodeLabels[nodeIndex]] += 1;
    }

    std::map<size_t, size_t> clusterMatching;

    for (auto & pair : predictedToTruthClusterOverlap){
        size_t nodeLabel = pair.first;
        std::map<size_t, size_t > & sizes = pair.second;

        std::map<size_t, size_t>::iterator best
                = std::max_element(sizes.begin(),sizes.end(),[] (const std::pair<size_t ,size_t>& a, const std::pair<size_t ,size_t>& b)->bool{ return a.second < b.second; } );

        clusterMatching[nodeLabel] = best->first;
    }

    // also store the translation of truth clusters and class name together with the firstClass / secondClass flag
    std::string clusterMatchingPath = filePath;
    clusterMatchingPath.replace(clusterMatchingPath.find(".csv"), std::string(".csv").length(), "_clusterMatching2.csv");

    std::cout << clusterMatchingPath << std::endl;

    std::cout << "Number of Truth Clusters: " << truthClusterSizes.size() << std::endl;

//    for (auto & pair : majorityMatching){
//        std::cout << predictedToTruthClusterOverlap[pair.first][pair.second] << " - " << pair.first << "->" << pair.second << " - " << truthClusterSizes[pair.second] << std::endl;
//    }
    std::ofstream matchingFile;
    matchingFile.open(clusterMatchingPath);

    // predExceptTruthClusterSize is the number of elements in predCluster and not in truthCluster
    // truthExceptPredClusterSize is the number of elements in truthCluster and not in predCluster
    matchingFile << "predCluster,truthCluster,predClusterSize,truthClusterSize,overlap,predExceptTruthClusterSize,truthExceptPredClusterSize\n";

    for (auto & predTruthMatch : predictedToTruthClusterOverlap){
        size_t predLabel = predTruthMatch.first;
        for (auto & pair : predictedToTruthClusterOverlap[predLabel]){
            size_t truthLabel = pair.first;
            size_t predTruthOverlap = pair.second;
            size_t truthPredOverlap = truthToPredictedClusterOverlaps[truthLabel][predLabel];
            assert(predTruthOverlap == truthPredOverlap);
            size_t truthClusterSize = truthClusterSizes[truthLabel];
            size_t predClusterSize = predClusterSizes[predLabel];
            size_t predExceptTruthClusterSize = predClusterSize - truthPredOverlap;
            size_t truthExceptPredClusterSize = truthClusterSize - truthPredOverlap;
            matchingFile << predLabel << "," << truthLabel << "," << predClusterSize << "," << truthClusterSize << "," << predTruthOverlap << "," << predExceptTruthClusterSize << "," << truthExceptPredClusterSize << "\n";
        }
    }

    matchingFile.close();

    // calculate cumulative distribution / predicted clusters
    std::vector<size_t> predClusterSizesSorted(predClusterSizes.size());
    for (auto & pair: predClusterSizes){
        predClusterSizesSorted[pair.first] = pair.second;
    }
    std::sort(predClusterSizesSorted.begin(), predClusterSizesSorted.end(), std::less_equal<>());
    std::string cumDistributionPredClusterSizesPath = filePath;
    cumDistributionPredClusterSizesPath.replace(cumDistributionPredClusterSizesPath.find(".csv"), std::string(".csv").length(), "_cumDistrPredClusterSizes.csv");

    std::ofstream cumDistrPredClustersFile;
    cumDistrPredClustersFile.open(cumDistributionPredClusterSizesPath);
    cumDistrPredClustersFile << "clusterSize,numClusters\n";
    for (size_t i = 0; i < predClusterSizesSorted.size(); ++i){
        cumDistrPredClustersFile << predClusterSizesSorted[i] << "," << i << "\n";
    }
    cumDistrPredClustersFile.close();



    // calculate cumulative distribution / predicted clusters
    std::vector<size_t > truthClusterSizesSorted(truthLabelSizes.size());
    for (auto & pair : truthLabelSizes){
        truthClusterSizesSorted[pair.first] = pair.second;
    }
    std::sort(truthClusterSizesSorted.begin(), truthClusterSizesSorted.end(), std::less_equal<>());
    std::string cumDistributionTruthClusterSizesPath = filePath;
    cumDistributionTruthClusterSizesPath.replace(cumDistributionTruthClusterSizesPath.find(".csv"), std::string(".csv").length(), "_cumDistributionTruthClusterSizes.csv");

    std::ofstream cumDistributionTruthClustersFile;
    cumDistributionTruthClustersFile.open(cumDistributionTruthClusterSizesPath);
    cumDistributionTruthClustersFile << "clusterSize,numClusters\n";
    for (size_t i = 0; i < truthClusterSizesSorted.size(); ++i){
        cumDistributionTruthClustersFile << truthClusterSizesSorted[i] << "," << i << "\n";
    }
    cumDistributionTruthClustersFile.close();


    // tex file writing
    std::string cumClustersTexPath = filePath;
    cumClustersTexPath.replace(cumClustersTexPath.find(".csv"), std::string(".csv").length(), "_cumClusterSizes.tex");

    std::ofstream  cumClustersTexFile;

    size_t numberOfPredClusters = predClusterSizesSorted.size();
    size_t numberOfTrueClusters = truthClusterSizesSorted.size();

    cumClustersTexFile.open(cumClustersTexPath);
    cumClustersTexFile << "\\documentclass{standalone}\n"
                       << "\\usepackage{pgfplots}\n"
                       << "\\pgfplotsset{compat=newest}\n"
                       << "\\begin{document}\n"
                       << "\\begin{tikzpicture}\n";
    cumClustersTexFile << "\\begin{axis}[xlabel={ClusterSize}, ylabel={Fraction of Clusters}, legend pos={south east}]\n";

    // predicted cluster sizes
    cumClustersTexFile << "\\addplot[green] table[x index=0, y index=1, col sep=comma] {";

    for (size_t index = 0; index < numberOfPredClusters; ++index){
        cumClustersTexFile << predClusterSizesSorted[index] << "," << static_cast<double>(index + 1) / static_cast<double>(numberOfPredClusters) << std::endl;
    }

    cumClustersTexFile << "};\n";

    // truth cluster sizes
    cumClustersTexFile << "\\addplot[red] table[x index=0, y index=1, col sep=comma] {";

    for (size_t index = 0; index < numberOfTrueClusters; ++index){
        cumClustersTexFile << truthClusterSizesSorted[index] << "," << static_cast<double>(index + 1) / static_cast<double>(numberOfTrueClusters) << std::endl;
    }

    cumClustersTexFile << "};\n";
    cumClustersTexFile << "\\legend{predicted, truth}\n";

    cumClustersTexFile << "\\end{axis}\n"
                           << "\\end{tikzpicture}\n"
                           << "\\end{document}";


    // TP, TN, FP, FN
    std::vector<size_t > classificationMetrics {0, 0, 0, 0};
    // iterate over all edgeNodeLabels
    for (size_t i = 0; i <  edgeNodeLabels.size(); ++i){
        size_t edgeNodeLabel = edgeNodeLabels[i];
        std::string predictedClass = truthLabelClasses[clusterMatching[edgeNodeLabel]];
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
    std::cout << "Classification Accuracy: " << (double) classificationMetrics[0] / (double)edgeNodeLabels.size() << std::endl;


    std::cout << "TP: " << tp << std::endl;
    std::cout << "TN: " << tn << std::endl;
    std::cout << "FP: " << fp << std::endl;
    std::cout << "FN: " << fn << std::endl;

    std::cout << "Accuracy: " << (tp + tn) / (tp + tn + fn + fp) << std::endl;
    std::cout << "Recall: " << tp / (tp + fn) << std::endl;
    std::cout << "Precision: " << tp / (tp + fp) << std::endl;


    std::cout << "Using Partition Comparison Classes..." << std::endl;

    RandError randError(truthLabels.begin(), truthLabels.end(), edgeNodeLabels.begin(), false);
    VI vi(truthLabels.begin(), truthLabels.end(), edgeNodeLabels.begin(), false);

    std::cout << "Rand Metrics" << std::endl;
    std::cout << std::string (20, '=') << std::endl;
    std::cout << "Precision Cuts: " << randError.precisionOfCuts() << std::endl;
    std::cout << "Recall Cuts: " << randError.recallOfCuts() << std::endl;
    std::cout << "Precision Joins: " << randError.precisionOfJoins() << std::endl;
    std::cout << "Recall Joins: " << randError.recallOfJoins() << std::endl;
    std::cout << "Rand Index: " << randError.index() << std::endl;
    std::cout << "Rand Error: " << randError.error() << std::endl;

    std::cout << "False Joins (Rand class): " << randError.falseJoins() << std::endl;
    std::cout << "False Joins (Self): " << fn << std::endl;

    std::cout << "False Cuts (Rand class): " << randError.falseCuts() << std::endl;
    std::cout << "False Cuts (Self): " << fp << std::endl;

    std::cout << std::endl;
    std::cout << "VI Metrics" << std::endl;
    std::cout << std::string (20, '=') << std::endl;
    std::cout << "VI: " << vi.value() << std::endl;
    std::cout << "VI False Cuts: " << vi.valueFalseCut() << std::endl;
    std::cout << "VI False Joins: " << vi.valueFalseJoin() << std::endl;





    return 0;
}
