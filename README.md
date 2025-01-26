# 使用方法

## 1. 注册新的模型

编辑 [model_config.py](model_config.py)

```
bc3_33b_3t_1203_all = Model(server='10.5.0.9:8102')
```

## 2. 启动评测

sh run.sh bc3_33b_3t_1203_all


# 添加新的评测集

将评测集以 `pandas.read_csv` 格式放在data目录下

参考：[data/evaluation_set_label_792.csv](data/evaluation_set_label_792.csv)
