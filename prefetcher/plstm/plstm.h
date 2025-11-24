#ifndef PREFETCHER_PLSTM_H
#define PREFETCHER_PLSTM_H

#include <cstdint>
#include <queue>
#include <vector>

#include "champsim.h"
#include "modules.h"
#include "lstm.h"
#include "msl/lru_table.h"

class plstm : public champsim::modules::prefetcher
{

  constexpr static int TABLE_WAYS = 128;
  constexpr static int TABLE_SETS = 8;
  LSTM<float> lstm{};
  constexpr static double mean=0.2185, std=16.2121;

  struct tracker_entry {
    champsim::address ip{};
    champsim::address last_address{};
    access_type atype;
    std::deque<float> history{};

    auto index() const {
      using namespace champsim::data::data_literals;
      return ip.slice_upper<48_b>();
    }

    auto tag() const {
      using namespace champsim::data::data_literals;
      return std::make_pair<>(ip.slice_upper<48_b>(), atype);
    }
  };
  champsim::msl::lru_table<tracker_entry> stride_table{TABLE_WAYS, TABLE_SETS};
  float out[LSTM_OUTPUT_SIZE];
  std::vector<float> input;
public:
  using champsim::modules::prefetcher::prefetcher;

  void prefetcher_initialize();
  // void prefetcher_branch_operate(champsim::address ip, uint8_t branch_type, champsim::address branch_target) {}
  uint32_t prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                    uint32_t metadata_in);
  uint32_t prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in);
  // void prefetcher_cycle_operate() {}
  // void prefetcher_final_stats() {}
};

#endif
