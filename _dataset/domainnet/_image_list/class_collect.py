unique_names = set()

with open("./real_list.txt", "r") as f:
    for line in f:
        parts = line.strip().split('/')
        if len(parts) > 3:
            unique_names.add(parts[3])  # 第4个位置就是 'dragon'

result = list(unique_names)
print(result)