import lzma
import struct
from collections import defaultdict
import numpy as np

# Struct format: < = little-endian
# Q = uint64 (8 bytes)
# B = uint8  (1 byte)
fmt = "Q" + "BB" + "2B" + "4B" + "2Q" + "4Q"
INSTR_SIZE = struct.calcsize(fmt)
print(f"Instruction size: {INSTR_SIZE} bytes")

def parse_champsim_trace_xz(filename):
    ip_stride_history = defaultdict(list)
    last_addr = defaultdict(lambda: None)
    i = 0
    with lzma.open(filename, "rb") as f:
        while True:
            if i > 5000000:
                break
            i += 1
            chunk = f.read(INSTR_SIZE)
            if len(chunk) < INSTR_SIZE:
                break

            unpacked = struct.unpack(fmt, chunk)

            # unpack according to struct
            # fmt = "<QBB2B4B2Q4Q"
            (ip,
            is_branch,
            branch_taken,
            dst_r0, dst_r1,          # 2B
            src_r0, src_r1, src_r2, src_r3,  # 4B
            dst_mem0, dst_mem1,      # 2Q
            src_mem0, src_mem1, src_mem2, src_mem3  # 4Q
            ) = struct.unpack(fmt, chunk)

            dest_regs = (dst_r0, dst_r1)
            src_regs  = (src_r0, src_r1, src_r2, src_r3)
            dest_mem  = (dst_mem0, dst_mem1)
            src_mem   = (src_mem0, src_mem1, src_mem2, src_mem3)

            # LOAD detection:
            # LOAD = at least one non-zero source_memory AND all dest_memory zero
            has_mem_src = any(addr != 0 for addr in src_mem)
            has_mem_dest = any(addr != 0 for addr in dest_mem)
            if has_mem_src and not has_mem_dest:
                # pick the first memory source as load address
                addr = next(a for a in src_mem if a != 0)
                ip = ip>>16
                # compute stride for this IP
                if last_addr[ip] is not None:
                    stride = addr - last_addr[ip]
                    ip_stride_history[ip].append(stride)

                last_addr[ip] = addr

    return ip_stride_history

if __name__ == "__main__":
    trace_file = "../traces/400.perlbench-41B.champsimtrace.xz"
    dictionary = parse_champsim_trace_xz(trace_file)
    for ip, strides in dictionary.items():
        strides_array = np.array(strides)
        np.save(f"strides/{ip:#x}_strides.npy", strides_array)