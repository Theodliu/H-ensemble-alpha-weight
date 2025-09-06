with open('./real_list.txt', 'r') as f:
    lines = f.readlines()

# 获取总行数
total_lines = len(lines)

# 每份大约的行数
chunk_size = total_lines // 3

# 分割并写入新文件
for i in range(3):
    start = i * chunk_size
    end = (i + 1) * chunk_size if i < 2 else total_lines  # 最后一份包含剩余所有行
    with open(f'real_{i}_list.txt', 'w') as out_f:
        out_f.writelines(lines[start:end])