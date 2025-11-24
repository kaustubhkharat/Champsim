#include "plstm.h"
#include "cache.h"

void plstm::prefetcher_initialize() {
  input.reserve(LSTM_INPUT_SIZE);
  lstm.initialise_from_folder("lstm_training/lstm_params");
  return;
}

uint32_t plstm::prefetcher_cache_operate(champsim::address addr, champsim::address ip, uint8_t cache_hit, bool useful_prefetch, access_type type,
                                      uint32_t metadata_in)
{
  // assert(addr == ip); // Invariant for instruction prefetchers
  auto found = stride_table.check_hit(tracker_entry{ip, ip, type, {}});
  if (found.has_value()){
    auto stride = champsim::offset(addr, found->last_address);
    float diff = (stride - mean)/std;
    if (found->history.size() < LSTM_INPUT_SIZE) {
      found->history.push_back(diff);
      return metadata_in;
    }
    found->history.pop_front();
    found->history.push_back(diff);
    input.assign(found->history.begin(), found->history.end());
    lstm.predict(input.data(), out);
    for (float &x:out){
      x = x*std + mean;
      x = pow(2, x) - 1;
    }
    champsim::address addr_cp = addr;
    for (auto x:out) {
      addr_cp += champsim::address::difference_type{(int)round(x)};
      prefetch_line(addr_cp, true, metadata_in);
    }
  } else {
    stride_table.fill(tracker_entry{ip, addr, type, {}});
  }
  return metadata_in;
}

uint32_t plstm::prefetcher_cache_fill(champsim::address addr, long set, long way, uint8_t prefetch, champsim::address evicted_addr, uint32_t metadata_in)
{
  return metadata_in;
}
