#pragma once
#include <xxxxx-3ing/collectives/allgather.hpp>
#include <xxxxx-3ing/collectives/alltoall.hpp>
#include <xxxxx-3ing/communicator.hpp>
#include <xxxxx-3ing/utils/flatten.hpp>
#include <random>

#include "sorting/common.hpp"

namespace xxxxx-3ing {
//> START SORTING xxxxx-6
template <typename T>
void sort(MPI_Comm comm_, std::vector<T> &data, size_t seed) {
  using namespace xxxxx-3ing;
  xxxxx-3ing::Communicator comm(comm_);
  const size_t oversampling_ratio = 16 * std::log2(comm.size()) + 1;
  std::vector<T> local_samples(oversampling_ratio);
  std::sample(data.begin(), data.end(), local_samples.begin(),
              oversampling_ratio, std::mt19937{seed});
  std::vector<T> global_samples(local_samples.size() * comm.size());
  comm.allgather(send_buf(local_samples), recv_buf(global_samples));
  pick_splitters(comm.size() - 1, oversampling_ratio, global_samples);
  auto buckets = build_buckets(data, global_samples);
  std::vector<int> sCounts, sDispls, rCounts(comm.size()), rDispls(comm.size());
  int send_pos = 0;
  for (auto &bucket : buckets) {
    data.insert(data.end(), bucket.begin(), bucket.end());
    sCounts.push_back(bucket.size());
    sDispls.push_back(send_pos);
    send_pos += bucket.size();
  }
  comm.alltoall(xxxxx-3ing::send_buf(sCounts), xxxxx-3ing::recv_buf(rCounts));
  // exclusive prefix sum of recv displacements
  std::exclusive_scan(rCounts.begin(), rCounts.end(), rDispls.begin(), 0);
  std::vector<T> rData(rDispls.back() + rCounts.back());
  comm.alltoallv(send_buf(data), send_counts(sCounts), send_displs(sDispls),
                 recv_buf(rData), recv_counts(rCounts), recv_displs(rDispls));
  std::sort(rData.begin(), rData.end());
  rData.swap(data);
}
//> END
}  // namespace xxxxx-3ing
