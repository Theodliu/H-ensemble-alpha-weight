input_file = "./real.txt"
output_file = "./real_list.txt"

# 要加入的前缀
prefix = "/data/"

# 打开文件并处理
with open(input_file, "r", encoding="utf-8") as infile, open(output_file, "w", encoding="utf-8") as outfile:
    for line in infile:
        # 在每行前加入前缀并写入新文件
        outfile.write(prefix + line)

print("处理完成！每行前已添加内容。")