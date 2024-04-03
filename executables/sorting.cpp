#if defined(xxxxx-6_EXAMPLES_USE_BOOST)
#include "sorting/bindings/boost.hpp"
#endif
#include <mpi.h>

#include <CLI/CLI.hpp>
#include <algorithm>
#include <iostream>
#include <xxxxx-3ing/collectives/gather.hpp>
#include <xxxxx-3ing/communicator.hpp>
#include <xxxxx-3ing/measurements/printer.hpp>
#include <xxxxx-3ing/measurements/timer.hpp>
#include <mpl/mpl.hpp>  // needed for initialization
#include <random>
#include <vector>

#include "sorting/bindings/xxxxx-3ing.hpp"
#include "sorting/bindings/xxxxx-3ing_flattened.hpp"
#include "sorting/bindings/mpi.hpp"
#include "sorting/bindings/mpl.hpp"
#include "sorting/bindings/rwth_mpi.hpp"
#include "sorting/common.hpp"

template <typename T>
bool globally_sorted(MPI_Comm comm, std::vector<T> const &data,
                     std::vector<T> &original_data) {
  xxxxx-3ing::Communicator xxxxx-3ing_comm(comm);
  auto global_data = xxxxx-3ing_comm.gatherv(xxxxx-3ing::send_buf(data));
  auto global_data_original =
      xxxxx-3ing_comm.gatherv(xxxxx-3ing::send_buf(original_data));

  std::sort(global_data_original.begin(), global_data_original.end());
  return global_data_original == global_data;
  // std::is_sorted(global_data.begin(), global_data.end());
}

template <typename T>
auto generate_data(size_t n_local, size_t seed) -> std::vector<T> {
  std::mt19937 eng(seed + xxxxx-3ing::world_rank());
  std::uniform_int_distribution<T> dist(0, std::numeric_limits<T>::max());
  std::vector<T> data(n_local);
  auto gen = [&] { return dist(eng); };
  std::generate(data.begin(), data.end(), gen);
  return data;
}

void log_results(std::string const &json_output_path,
                 std::string const &algorithm, size_t n_local, size_t seed,
                 bool correct) {
  std::unique_ptr<std::ostream> output_stream;
  if (json_output_path == "stdout") {
    output_stream = std::make_unique<std::ostream>(std::cout.rdbuf());
  } else {
    std::ofstream file_output(json_output_path);
    output_stream = std::make_unique<std::ofstream>(std::move(file_output));
  }
  if (mpl::environment::comm_world().rank() == 0) {
    *output_stream << "{\n";
  }
  xxxxx-3ing::measurements::timer().aggregate_and_print(
      xxxxx-3ing::measurements::SimpleJsonPrinter<>{*output_stream});
  if (mpl::environment::comm_world().rank() == 0) {
    *output_stream << ",\n";
    *output_stream << "\"info\": {\n";
    *output_stream << "  \"algorithm\": "
                   << "\"" << algorithm << "\",\n";
    *output_stream << "  \"p\": " << mpl::environment::comm_world().size()
                   << ",\n";
    *output_stream << "  \"n_local\": " << n_local << ",\n";
    *output_stream << "  \"seed\": " << seed << ",\n";
    *output_stream << "  \"correct\": " << std::boolalpha << correct << "\n";
    *output_stream << "}\n";
    *output_stream << "}";
  }
}

int main(int argc, char *argv[]) {
  mpl::environment::comm_world();  // this perform MPI_init, MPL has no other
                                   // way to do it and calls it implicitly when
                                   // first accessing a communicator
  CLI::App app{"Parallel sorting"};
  std::string algorithm;
  app.add_option("--algorithm", algorithm);
  size_t n_local;
  app.add_option("--n_local", n_local);
  size_t seed = 42;
  app.add_option("--seed", seed);
  size_t iterations = 1;
  app.add_option("--iterations", iterations);
  bool check = false;
  app.add_flag("--check", check);
  bool warmup = false;
  app.add_flag("--warmup", warmup);
  std::string json_output_path = "stdout";
  app.add_option("--json_output_path", json_output_path);
  CLI11_PARSE(app, argc, argv);

  using element_type = uint64_t;

  auto original_data = generate_data<element_type>(n_local, seed);
  size_t local_seed = seed + xxxxx-3ing::world_rank() + xxxxx-3ing::world_size();
  bool correct = false;
  auto do_run = [&](auto &&algo) {
    if (check) {
      xxxxx-3ing::measurements::timer().synchronize_and_start("warmup_time");
      auto data = original_data;
      algo(MPI_COMM_WORLD, data, local_seed);
      xxxxx-3ing::measurements::timer().stop_and_append();
      correct = globally_sorted(MPI_COMM_WORLD, data, original_data);
    } else if (warmup) {
      xxxxx-3ing::measurements::timer().synchronize_and_start("warmup_time");
      auto data = original_data;
      algo(MPI_COMM_WORLD, data, local_seed);
      xxxxx-3ing::measurements::timer().stop_and_append();
    }
    for (size_t iteration = 0; iteration < iterations; iteration++) {
      auto data = original_data;
      xxxxx-3ing::measurements::timer().synchronize_and_start("total_time");
      algo(MPI_COMM_WORLD, data, local_seed);
      xxxxx-3ing::measurements::timer().stop_and_append();
    }
  };
  if (algorithm == "mpi") {
    do_run(mpi::sort<element_type>);
  } else if (algorithm == "xxxxx-3ing") {
    do_run(xxxxx-3ing::sort<element_type>);
  } else if (algorithm == "xxxxx-3ing_flattened") {
    do_run(xxxxx-3ing_flattened::sort<element_type>);
#if defined(xxxxx-6_EXAMPLES_USE_BOOST)
  } else if (algorithm == "boost") {
    do_run(boost::sort<element_type>);
#endif
  } else if (algorithm == "rwth") {
    do_run(rwth::sort<element_type>);
  } else if (algorithm == "mpl") {
    do_run(mpl::sort<element_type>);
  } else {
    throw std::runtime_error("unsupported algorithm");
  }
  log_results(json_output_path, algorithm, n_local, seed, correct);
  return 0;
}
