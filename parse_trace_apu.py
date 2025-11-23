import py7zr
import struct
from collections import defaultdict
import numpy as np
import os

# Struct format: < = little-endian
# Q = uint64 (8 bytes)
# B = uint8  (1 byte)
fmt = "<QBB2B4B2Q4Q"
INSTR_SIZE = struct.calcsize(fmt)
print(f"Instruction size: {INSTR_SIZE} bytes")

def parse_champsim_trace_7z(filename, max_instr=5000000):
    ip_stride_history = defaultdict(list)
    last_addr = defaultdict(lambda: None)
    i = 0

    with py7zr.SevenZipFile(filename, mode='r') as archive:
        # Assuming the .7z contains a single trace file
        extracted_files = archive.readall()  # dict: {filename: file-like object}
        if not extracted_files:
            raise ValueError("No files found inside the .7z archive")

        # Take the first file inside the archive
        trace_fileobj = next(iter(extracted_files.values()))

        while True:
            if i >= max_instr:
                break
            i += 1
            chunk = trace_fileobj.read(INSTR_SIZE)
            if len(chunk) < INSTR_SIZE:
                break

            unpacked = struct.unpack(fmt, chunk)
            (ip,
             is_branch,
             branch_taken,
             dst_r0, dst_r1,
             src_r0, src_r1, src_r2, src_r3,
             dst_mem0, dst_mem1,
             src_mem0, src_mem1, src_mem2, src_mem3
            ) = unpacked

            dest_regs = (dst_r0, dst_r1)
            src_regs  = (src_r0, src_r1, src_r2, src_r3)
            dest_mem  = (dst_mem0, dst_mem1)
            src_mem   = (src_mem0, src_mem1, src_mem2, src_mem3)

            # LOAD detection
            has_mem_src = any(addr != 0 for addr in src_mem)
            has_mem_dest = any(addr != 0 for addr in dest_mem)
            if has_mem_src and not has_mem_dest:
                addr = next(a for a in src_mem if a != 0)
                ip_reduced = ip >> 16
                if last_addr[ip_reduced] is not None:
                    stride = addr - last_addr[ip_reduced]
                    ip_stride_history[ip_reduced].append(stride)
                last_addr[ip_reduced] = addr

    return ip_stride_history


if __name__ == "__main__":
    trace_file = "/home/apurva/apurva/projnew/resnet_trace_small1.7z"
    dictionary = parse_champsim_trace_7z(trace_file)

    os.makedirs("strides", exist_ok=True)
    for ip, strides in dictionary.items():
        strides_array = np.array(strides)
        np.save(f"strides/{ip:#x}_strides.npy", strides_array)
