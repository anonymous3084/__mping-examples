#pragma once
#include <xxxxx-3ing/collectives/allgather.hpp>
#include <xxxxx-3ing/collectives/alltoall.hpp>
#include <xxxxx-3ing/communicator.hpp>
#include <xxxxx-3ing/utils/flatten.hpp>
#include <random>

#include "sorting/common.hpp"

namespace xxxxx-3ing_flattened {
//> START SORTING xxxxx-6_FLATTENED
template <typename T>
void sort(MPI_Comm comm_, std::vector<T> &data, size_t seed) {
  using namespace xxxxx-3ing;
  Communicator<> comm(comm_);
  const size_t oversampling_ratio = 16 * std::log2(comm.size()) + 1;
  std::vector<T> local_samples(oversampling_ratio);
  std::sample(data.begin(), data.end(), local_samples.begin(),
              oversampling_ratio, std::mt19937{seed});
  auto global_samples = comm.allgather(send_buf(local_samples));
  pick_splitters(comm.size() - 1, oversampling_ratio, global_samples);
  auto buckets = build_buckets(data, global_samples);
  data = with_flattened(buckets, comm.size()).call([&](auto... flattened) {
    return comm.alltoallv(std::move(flattened)...);
  });
  std::sort(data.begin(), data.end());
}
//> END
}  // namespace xxxxx-3ing_flattened
