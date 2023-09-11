#pragma once
#ifndef ANON_GRAPH_SUBGRAPH_HXX
#define ANON_GRAPH_SUBGRAPH_HXX

namespace anon {
namespace graph {

/// An entire graph.
template<class T = std::size_t>
struct DefaultSubgraphMask {
    typedef T Value;

    bool vertex(const Value v) const
        { return true; }
    bool edge(const Value e) const
        { return true; }
};

} // namespace graph
} // namespace anon

#endif // #ifndef ANON_GRAPH_SUBGRAPH_HXX
