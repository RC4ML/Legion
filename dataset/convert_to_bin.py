import pandas as pd

# 假设你的文件名是 'data.txt'
file_path = 'data.txt'

# 使用pandas的read_csv函数读取文件
df = pd.read_csv(file_path, header=None, delimiter="\s+")

# 转换为numpy数组
data = df.to_numpy()

print(data)
