#ifndef ANON_MULTICUT_EDGELABELMASK_HXX
#define ANON_MULTICUT_EDGELABELMASK_HXX

#include "vector"


namespace anon{
    namespace graph {
        namespace multicut {

            template<class T = std::size_t>
            struct CutSolutionMask {
                typedef T Value;

                explicit CutSolutionMask(std::vector<std::size_t> & edgeLabelIterator);

                bool vertex(const Value v) const
                { return true; }
                bool edge(const Value e) const
                {
                    if (edgeLabels_[e] == 0){
                        return true;
                    } else if (edgeLabels_[e] == 1){
                        // edgeLabel is 1 -> cut, so do not traverse
                        return false;
                    } else {
                        throw std::runtime_error("Edge Label was neither 0 or 1.");
                    }
                }

            private:
                std::vector<std::size_t> & edgeLabels_;
            };

            template<class T>
            CutSolutionMask<T>::CutSolutionMask(std::vector<std::size_t > & edgeLabelIterator): edgeLabels_(edgeLabelIterator) {

            }
        }
    }
}

#endif //ANON_MULTICUT_EDGELABELMASK_HXX
