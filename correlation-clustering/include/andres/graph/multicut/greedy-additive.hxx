#pragma once
#ifndef ANDRES_GRAPH_MULTICUT_GREEDY_ADDITIVE_HXX
#define ANDRES_GRAPH_MULTICUT_GREEDY_ADDITIVE_HXX

#include <cstddef>
#include <iterator>
#include <vector>
#include <algorithm>
#include <map>
#include <queue>
#include <set>

#include "andres/partition.hxx"


namespace andres {
namespace graph {
namespace multicut {

/// Greedy agglomerative decomposition of a graph.
///
template<typename GRAPH, typename EVA, typename ELA>
void greedyAdditiveEdgeContraction(
    const GRAPH& graph,
    EVA const& edge_values,
    ELA& edge_labels
)
{
    class DynamicGraph
    {
    public:
        DynamicGraph(size_t n) :
            vertices_(n)
        {}

        bool edgeExists(size_t a, size_t b) const
        {
            return !vertices_[a].empty() && vertices_[a].find(b) != vertices_[a].end();
        }

        std::map<size_t, typename EVA::value_type> const& getAdjacentVertices(size_t v) const
        {
            return vertices_[v];
        }

        typename EVA::value_type getEdgeWeight(size_t a, size_t b) const
        {
            return vertices_[a].at(b);
        }

        void removeVertex(size_t v)
        {
            for (auto& p : vertices_[v])
                vertices_[p.first].erase(v);

            vertices_[v].clear();
        }

        void updateEdgeWeight(size_t a, size_t b, typename EVA::value_type w)
        {
            vertices_[a][b] += w;
            vertices_[b][a] += w;
        }

    private:
        std::vector<std::map<size_t, typename EVA::value_type>> vertices_;
    };

    struct Edge
    {
        Edge(size_t _a, size_t _b, typename EVA::value_type _w)
        {
            if (_a > _b)
                std::swap(_a, _b);

            a = _a;
            b = _b;

            w = _w;
        }

        size_t a;
        size_t b;
        size_t edition;
        typename EVA::value_type w;

        bool operator <(Edge const& other) const
        {
            return w < other.w;
        }
    };

    std::vector<std::map<size_t, size_t>> edge_editions(graph.numberOfVertices());
    DynamicGraph original_graph_cp(graph.numberOfVertices());
    std::priority_queue<Edge> Q;


    std::cout << "Greedy Additive starting init..." << std::endl;

    for (size_t i = 0; i < graph.numberOfEdges(); ++i)
    {
        auto a = graph.vertexOfEdge(i, 0);
        auto b = graph.vertexOfEdge(i, 1);

        original_graph_cp.updateEdgeWeight(a, b, edge_values[i]);

        auto e = Edge(a, b, edge_values[i]);
        e.edition = ++edge_editions[e.a][e.b];
        
        Q.push(e);
    }

    std::cout << "Greedy Additive done init..." << std::endl;

    andres::Partition<size_t> partition(graph.numberOfVertices());

    while (!Q.empty())
    {
        auto edge = Q.top();
        Q.pop();

        if (!original_graph_cp.edgeExists(edge.a, edge.b) || edge.edition < edge_editions[edge.a][edge.b])
            continue;
        
        if (edge.w < typename EVA::value_type())
            break;

        auto stable_vertex = edge.a;
        auto merge_vertex = edge.b;

        if (original_graph_cp.getAdjacentVertices(stable_vertex).size() < original_graph_cp.getAdjacentVertices(merge_vertex).size())
            std::swap(stable_vertex, merge_vertex);

        partition.merge(stable_vertex, merge_vertex);

        for (auto& p : original_graph_cp.getAdjacentVertices(merge_vertex))
        {
            if (p.first == stable_vertex)
                continue;

            original_graph_cp.updateEdgeWeight(stable_vertex, p.first, p.second);

            auto e = Edge(stable_vertex, p.first, original_graph_cp.getEdgeWeight(stable_vertex, p.first));
            e.edition = ++edge_editions[e.a][e.b];

            Q.push(e);
        }

        original_graph_cp.removeVertex(merge_vertex);
    }

    for (size_t i = 0; i < graph.numberOfEdges(); ++i)
        edge_labels[i] = partition.find(graph.vertexOfEdge(i, 0)) == partition.find(graph.vertexOfEdge(i, 1)) ? 0 : 1;
}



/// Greedy agglomerative decomposition of a graph.
///
    template<typename GRAPH, typename EVA, typename ELA>
    void greedyAdditiveEdgeContractionCompleteGraph(
            const GRAPH& graph,
            EVA const& edge_values,
            ELA& edge_labels
    )
    {
        class DynamicGraph
        {
        public:
            DynamicGraph(size_t const & n) :
                numberOfElements_(n),
                edgeCosts_(numberOfElements_ * (numberOfElements_ - 1) / 2, 0),
                isEdgeRemoved_(numberOfElements_ * (numberOfElements_ - 1) / 2, false)
            {}

            bool edgeExists(size_t a, size_t b) const
            {
                if (a > b){
                    return !isEdgeRemoved_[indexOfEdge(a, b)];
                } else if (b > a){
                    return !isEdgeRemoved_[indexOfEdge(b, a)];
                } else {
                    throw std::runtime_error("No edge with same origin as destination.");
                }
            }

            std::map<size_t, typename EVA::value_type> const getAdjacentVertices(size_t v) const
            {
                std::map<size_t , typename EVA::value_type> adjacentVertices;
                for (size_t i = 0; i < v; ++i){
                    size_t edgeIndex = indexOfEdge(v, i);

                    if (!isEdgeRemoved_[edgeIndex]){
                        adjacentVertices[i] = edgeCosts_[edgeIndex];
                    }
                }
                for (size_t i = v+1; i < numberOfElements_; ++i){
                    size_t edgeIndex = indexOfEdge(i, v);

                    if (!isEdgeRemoved_[edgeIndex]){
                        adjacentVertices[i] = edgeCosts_[edgeIndex];
                    }
                }
                return adjacentVertices;
            }

            typename EVA::value_type getEdgeWeight(size_t a, size_t b) const
            {
                if (a > b) {
                    size_t edgeIndex = indexOfEdge(a, b);
                    if (isEdgeRemoved_[edgeIndex]){
                        throw std::runtime_error("Edge has been removed.");
                    }

                    return edgeCosts_[edgeIndex];
                } else if (b > a) {
                    size_t edgeIndex = indexOfEdge(b, a);
                    if (isEdgeRemoved_[edgeIndex]){
                        throw std::runtime_error("Edge has been removed.");
                    }

                    return edgeCosts_[edgeIndex];
                } else {
                    throw std::runtime_error("No Edge with same origin as destination.");
                }
            }

            void removeVertex(size_t v)
            {
                // we simply set these costs to zero and set the removed flag for the edges to true
                for (size_t i = 0; i < v; ++i){
                    size_t edgeIndex = indexOfEdge(v, i);
                    isEdgeRemoved_[edgeIndex] = true;
                    edgeCosts_[edgeIndex] = 0;
                }

                for (size_t i = v + 1; i < numberOfElements_; ++i){
                    size_t edgeIndex = indexOfEdge(i, v);
                    isEdgeRemoved_[edgeIndex] = true;
                    edgeCosts_[edgeIndex] = 0;
                }
            }

            void updateEdgeWeight(size_t a, size_t b, typename EVA::value_type w)
            {
                if (a > b){
                    size_t edgeIndex = indexOfEdge(a, b);
                    if (isEdgeRemoved_[edgeIndex]){
                        throw std::runtime_error("Edge has been removed.");
                    }

                    edgeCosts_[edgeIndex] += w;
                } else if (b > a){
                    size_t edgeIndex = indexOfEdge(b, a);
                    if (isEdgeRemoved_[edgeIndex]){
                        throw std::runtime_error("Edge has been removed.");
                    }

                    edgeCosts_[edgeIndex] += w;
                } else {
                    throw std::runtime_error("No Edge with same origin as destination.");
                }
            }

        private:
            size_t numberOfElements_;
            std::vector<typename EVA::value_type> edgeCosts_;
            size_t indexOfEdge(size_t const i, size_t const j) const {
                assert(i > j);
                return i * (i - 1) / 2 + j;
            }
            std::vector<bool> isEdgeRemoved_;
        };

        struct Edge
        {
            Edge(size_t _a, size_t _b, typename EVA::value_type _w)
            {
                if (_a > _b)
                    std::swap(_a, _b);

                a = _a;
                b = _b;

                w = _w;
            }

            size_t a;
            size_t b;
            size_t edition;
            typename EVA::value_type w;

            bool operator <(Edge const& other) const
            {
                return w < other.w;
            }
        };

        std::vector<std::map<size_t, size_t>> edge_editions(graph.numberOfVertices());
        DynamicGraph original_graph_cp(graph.numberOfVertices());
        std::priority_queue<Edge> Q;


        std::cout << "Greedy Additive starting init..." << std::endl;

        for (size_t i = 0; i < graph.numberOfEdges(); ++i)
        {
            auto a = graph.vertexOfEdge(i, 0);
            auto b = graph.vertexOfEdge(i, 1);

            original_graph_cp.updateEdgeWeight(a, b, edge_values[i]);

            auto e = Edge(a, b, edge_values[i]);
            e.edition = ++edge_editions[e.a][e.b];

            Q.push(e);
        }

        std::cout << "Greedy Additive done init..." << std::endl;

        andres::Partition<size_t> partition(graph.numberOfVertices());

        while (!Q.empty())
        {
            auto edge = Q.top();
            Q.pop();

            if (!original_graph_cp.edgeExists(edge.a, edge.b) || edge.edition < edge_editions[edge.a][edge.b])
                continue;

            if (edge.w < typename EVA::value_type())
                break;

            auto stable_vertex = edge.a;
            auto merge_vertex = edge.b;

            if (original_graph_cp.getAdjacentVertices(stable_vertex).size() < original_graph_cp.getAdjacentVertices(merge_vertex).size())
                std::swap(stable_vertex, merge_vertex);

            partition.merge(stable_vertex, merge_vertex);

            for (auto& p : original_graph_cp.getAdjacentVertices(merge_vertex))
            {
                if (p.first == stable_vertex)
                    continue;

                original_graph_cp.updateEdgeWeight(stable_vertex, p.first, p.second);

                auto e = Edge(stable_vertex, p.first, original_graph_cp.getEdgeWeight(stable_vertex, p.first));
                e.edition = ++edge_editions[e.a][e.b];

                Q.push(e);
            }

            original_graph_cp.removeVertex(merge_vertex);
        }

        for (size_t i = 0; i < graph.numberOfEdges(); ++i)
            edge_labels[i] = partition.find(graph.vertexOfEdge(i, 0)) == partition.find(graph.vertexOfEdge(i, 1)) ? 0 : 1;
    }

} // namespace multicut
} // namespace graph
} // namespace andres

#endif // #ifndef ANDRES_GRAPH_MULTICUT_GREEDY_ADDITIVE_HXX
