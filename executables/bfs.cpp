#if defined(xxxxx-6_EXAMPLES_USE_BOOST)
#include "bfs/bindings/boost.hpp"
#endif

#include <spdlog/fmt/ranges.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/spdlog.h>

#include <CLI/CLI.hpp>
#include <xxxxx-3ing/collectives/allreduce.hpp>
#include <xxxxx-3ing/collectives/alltoall.hpp>
#include <xxxxx-3ing/collectives/reduce.hpp>
#include <xxxxx-3ing/communicator.hpp>
#include <xxxxx-3ing/environment.hpp>
#include <xxxxx-3ing/measurements/printer.hpp>
#include <xxxxx-3ing/measurements/timer.hpp>
#include <xxxxx-3ing/mpi_datatype.hpp>
#include <xxxxx-3ing/plugin/alltoall_grid.hpp>
#include <xxxxx-3ing/plugin/alltoall_sparse.hpp>
#include <memory>
#include <ranges>

#include "bfs/bfs_algorithm.hpp"
#include "bfs/bindings/xxxxx-3ing.hpp"
#include "bfs/bindings/xxxxx-3ing_flattened.hpp"
#include "bfs/bindings/xxxxx-3ing_grid.hpp"
#include "bfs/bindings/xxxxx-3ing_sparse.hpp"
#include "bfs/bindings/mpi.hpp"
#include "bfs/bindings/mpi_neighborhood.hpp"
#include "bfs/bindings/mpi_neighborhood_dynamic.hpp"
#include "bfs/bindings/mpl.hpp"
#include "bfs/bindings/rwth_mpi.hpp"
#include "bfs/common.hpp"
#include "bfs/utils.hpp"

enum class Algorithm {
  boost,
  xxxxx-3ing,
  xxxxx-3ing_flattened,
  xxxxx-3ing_grid,
  xxxxx-3ing_sparse,
  mpi,
  mpi_neighborhood,
  mpi_neighborhood_dynamic,
  mpl,
  rwth_mpi
};

std::string to_string(const Algorithm& algorithm) {
  switch (algorithm) {
    case Algorithm::boost:
      return "boost";
    case Algorithm::xxxxx-3ing:
      return "xxxxx-3ing";
    case Algorithm::xxxxx-3ing_flattened:
      return "xxxxx-3ing_flattened";
    case Algorithm::xxxxx-3ing_grid:
      return "xxxxx-3ing_grid";
    case Algorithm::xxxxx-3ing_sparse:
      return "xxxxx-3ing_sparse";
    case Algorithm::mpi:
      return "mpi";
    case Algorithm::mpi_neighborhood:
      return "mpi_neighborhood";
    case Algorithm::mpi_neighborhood_dynamic:
      return "mpi_neighborhood_dynamic";
    case Algorithm::mpl:
      return "mpl";
    case Algorithm::rwth_mpi:
      return "rwth_mpi";
    default:
      throw std::runtime_error("unsupported algorithm");
  };
}

auto dispatch_bfs_algorithm(Algorithm algorithm) {
  using namespace graph;
  switch (algorithm) {
#if defined(xxxxx-6_EXAMPLES_USE_BOOST)
    case Algorithm::boost: {
      using Frontier = bfs_boost::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
#endif
    case Algorithm::xxxxx-3ing: {
      using Frontier = bfs_xxxxx-3ing::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
    case Algorithm::xxxxx-3ing_flattened: {
      using Frontier = bfs_xxxxx-3ing_flattened::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
    case Algorithm::xxxxx-3ing_grid: {
      using Frontier = bfs_xxxxx-3ing_grid::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
    case Algorithm::xxxxx-3ing_sparse: {
      using Frontier = bfs_xxxxx-3ing_sparse::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
    case Algorithm::mpi: {
      using Frontier = bfs_mpi::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
    case Algorithm::mpi_neighborhood: {
      using Frontier = bfs_mpi_neighborhood::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
    case Algorithm::mpi_neighborhood_dynamic: {
      using Frontier = bfs_mpi_neighborhood_dynamic::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
    case Algorithm::mpl: {
      using Frontier = bfs_mpl::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
    case Algorithm::rwth_mpi: {
      using Frontier = bfs_rwth_mpi::BFSFrontier;
      return bfs::bfs<Frontier>;
    }
    default:
      throw std::runtime_error("unsupported algorithm");
  };
}

void log_results(std::string const& json_output_path, std::size_t iterations,
                 std::string const& algorithm,
                 std::string const& kagen_option_string, size_t max_bfs_level,
                 size_t seed) {
  std::unique_ptr<std::ostream> output_stream;
  if (xxxxx-3ing::comm_world().rank() == 0) {
    if (json_output_path == "stdout") {
      output_stream = std::make_unique<std::ostream>(std::cout.rdbuf());
    } else {
      std::ofstream file_output(json_output_path);
      output_stream = std::make_unique<std::ofstream>(std::move(file_output));
    }
    *output_stream << "{\n";
  }
  xxxxx-3ing::measurements::timer().aggregate_and_print(
      xxxxx-3ing::measurements::SimpleJsonPrinter<>{*output_stream});
  if (mpl::environment::comm_world().rank() == 0) {
    *output_stream << ",\n";
    *output_stream << "\"info\": {\n";
    *output_stream << "  \"iterations\": "
                   << "\"" << iterations << "\",\n";
    *output_stream << "  \"algorithm\": "
                   << "\"" << algorithm << "\",\n";
    *output_stream << "  \"graph\": "
                   << "\"" << kagen_option_string << "\",\n";
    *output_stream << "  \"p\": " << mpl::environment::comm_world().size()
                   << ",\n";
    *output_stream << "  \"max_bfs_level\": " << max_bfs_level << ",\n";
    *output_stream << "  \"seed\": " << seed << "\n";
    *output_stream << "}\n";
    *output_stream << "}";
  }
}

auto main(int argc, char* argv[]) -> int {
  mpl::environment::comm_world();  // this perform MPI_init, MPL has no other
                                   // way to do it and calls it implicitly when
                                   // first accessing a communicator

  auto formatter = std::make_unique<spdlog::pattern_formatter>();
  formatter->add_flag<rank_formatter>('r');
  formatter->add_flag<size_formatter>('s');
  formatter->set_pattern("[%r/%s] [%^%l%$] %v");
  spdlog::set_formatter(std::move(formatter));

  spdlog::default_logger()->set_level(spdlog::level::debug);
  CLI::App app{"BFS"};
  std::string kagen_option_string;
  app.add_option("--kagen_option_string", kagen_option_string, "Kagen options")
      ->required();
  bool permute = false;
  app.add_flag("--permute", permute);
  size_t seed = 42;
  app.add_option("--seed", seed);
  Algorithm algorithm = Algorithm::mpi;
  app.add_option("--algorithm", algorithm, "Algorithm type")
      ->transform(
          CLI::CheckedTransformer(std::unordered_map<std::string, Algorithm>{
              {"boost", Algorithm::boost},
              {"xxxxx-3ing", Algorithm::xxxxx-3ing},
              {"xxxxx-3ing_flattened", Algorithm::xxxxx-3ing_flattened},
              {"xxxxx-3ing_grid", Algorithm::xxxxx-3ing_grid},
              {"xxxxx-3ing_sparse", Algorithm::xxxxx-3ing_sparse},
              {"mpi", Algorithm::mpi},
              {"mpi_neighborhood", Algorithm::mpi_neighborhood},
              {"mpi_neighborhood_dynamic", Algorithm::mpi_neighborhood_dynamic},
              {"mpl", Algorithm::mpl},
              {"rwth_mpi", Algorithm::rwth_mpi}}));
  size_t iterations = 1;
  app.add_option("--iterations", iterations, "Number of iterations");
  std::string json_output_path = "stdout";
  app.add_option("--json_output_path", json_output_path, "Path to JSON output");
  CLI11_PARSE(app, argc, argv);

  auto do_run = [&](auto&& bfs) {
    const auto g = [&]() {
      auto graph = graph::generate_distributed_graph(kagen_option_string);
      if (permute) {
        kagen_option_string += ";permute=true";
        return graph::permute(graph, seed);
      } else {
        kagen_option_string += ";permute=false";
        return graph;
      }
    }();

    const graph::VertexId root = [&]() {
      graph::VertexId r = graph::generate_start_vertex(g, seed);
      if (permute) {
        return graph::permute_vertex(g.global_num_vertices(), seed, r);
      } else {
        return r;
      }
    }();

    const std::vector<size_t> reference_bfs_levels =
        dispatch_bfs_algorithm(Algorithm::mpi)(g, root, MPI_COMM_WORLD);

    std::vector<size_t> bfs_levels;
    for (size_t iteration = 0; iteration < iterations; ++iteration) {
      xxxxx-3ing::measurements::timer().synchronize_and_start("total_time");
      bfs_levels = bfs(g, root, MPI_COMM_WORLD);
      xxxxx-3ing::measurements::timer().stop_and_append();
    }

    if (reference_bfs_levels != bfs_levels) {
      std::runtime_error("bfs level computation is not correct!");
    }
    return bfs_levels;
  };

  auto bfs_levels = do_run(dispatch_bfs_algorithm(algorithm));

  // outputting
  auto reached_levels = bfs_levels | std::views::filter([](auto l) noexcept {
                          return l != graph::unreachable_vertex;
                        });
  auto it = std::ranges::max_element(reached_levels);
  size_t max_bfs_level = it == reached_levels.end() ? 0 : *it;
  xxxxx-3ing::comm_world().allreduce(xxxxx-3ing::send_recv_buf(max_bfs_level),
                                  xxxxx-3ing::op(xxxxx-3ing::ops::max<>{}));
  log_results(json_output_path, iterations, to_string(algorithm),
              kagen_option_string, max_bfs_level, seed);
  return 0;
}
