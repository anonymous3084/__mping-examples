#pragma once

#include "bfs/common.hpp"
#include "xxxxx-3ing/collectives/allreduce.hpp"
#include "xxxxx-3ing/collectives/alltoall.hpp"

namespace bfs_xxxxx-3ing {
//> START BFS xxxxx-6
using namespace xxxxx-3ing;
class BFSFrontier final : public graph::BFSFrontier {
 public:
  BFSFrontier(MPI_Comm comm) : _comm{comm} {}
  std::pair<graph::VertexBuffer, bool> exchange() override {
    if (is_empty()) {
      return std::make_pair(graph::VertexBuffer{}, true);
    }
    graph::VertexBuffer data;
    std::vector<int> sCounts(_comm.size());
    for (size_t rank = 0; rank < _comm.size(); rank++) {
      auto it = _data.find(rank);
      if (it == _data.end()) {
        sCounts[rank] = 0;
        continue;
      }
      auto &local_data = it->second;
      data.insert(data.end(), local_data.begin(), local_data.end());
      sCounts[rank] = local_data.size();
    }
    _data.clear();
    auto new_frontier = _comm.alltoallv(send_buf(data), send_counts(sCounts));
    return std::make_pair(std::move(new_frontier), false);
  }
  bool is_empty() const {
    return _comm.allreduce_single(send_buf(_data.empty()),
                                  op(std::logical_and<>{}));
  }

 private:
  xxxxx-3ing::Communicator<> _comm;
};
//> END
}  // namespace bfs_xxxxx-3ing
