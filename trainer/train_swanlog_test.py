import swanlab
import random

# 初始化一个新的 swanlab 实验
swanlab.init(
  # 设置将记录此次实验的项目信息
  project="liang",
  workspace="Moleitau-WorldSaver",
  # 跟踪超参数和实验元数据
  config={
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10
  }
)

# 模拟训练
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
  acc = 1 - 2 ** -epoch - random.random() / epoch - offset
  loss = 2 ** -epoch + random.random() / epoch + offset

  # 向 swanlab 上传训练指标
  swanlab.log({"acc": acc, "loss": loss})

# [可选] 完成实验，这在 notebook 环境中是必要的
swanlab.finish()
