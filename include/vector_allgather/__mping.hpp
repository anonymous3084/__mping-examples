#pragma once

namespace xxxxx-3ing {
template <typename T>
std::vector<T> get_whole_vector(std::vector<T> const& v_local, MPI_Comm comm_) {
  xxxxx-3ing::Communicator comm(comm_);
  //> START VECTOR_ALLGATHER xxxxx-6
  return comm.allgatherv(xxxxx-3ing::send_buf(v_local));
  //> END
}
}  // namespace xxxxx-3ing
