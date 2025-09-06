inputs = ["./real_0_list.txt", "./real_1_list.txt", "./real_2_list.txt"]
outputs = ["./real_4_list.txt", "./real_5_list.txt", "./real_6_list.txt"]


out_files = [open(p, "w", encoding="utf-8") for p in outputs]

try:
    for in_path in inputs:
        with open(in_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f, start=1): 
                k = (idx - 1) % 3  
                out_files[k].write(line)
finally:
    for fh in out_files:
        fh.close()