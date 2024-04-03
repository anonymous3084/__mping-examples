#pragma once

#include "bfs/common.hpp"
#include "xxxxx-3ing/collectives/allreduce.hpp"
#include "xxxxx-3ing/collectives/alltoall.hpp"
#include "xxxxx-3ing/utils/flatten.hpp"

namespace bfs_xxxxx-3ing_flattened {
//> START BFS xxxxx-6_FLATTENED
using namespace xxxxx-3ing;
class BFSFrontier final : public graph::BFSFrontier {
 public:
  BFSFrontier(MPI_Comm comm) : _comm{comm} {}
  std::pair<graph::VertexBuffer, bool> exchange() override {
    if (is_empty()) {
      return std::make_pair(graph::VertexBuffer{}, true);
    }
    auto new_frontier =
        with_flattened(_data, _comm.size()).call([&](auto... flattened) {
          _data.clear();
          return _comm.alltoallv(std::move(flattened)...);
        });
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
}  // namespace bfs_xxxxx-3ing_flattened
