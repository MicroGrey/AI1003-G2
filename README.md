# 测试说明
- 架构代码：`basicsr/archs/lspan_arch.py`  
- 训练配置：`options/Train/train_LSPAN_x2.yml`  
- 测试配置：`options/Test/benchmark_LSPAN_x2.yml`  
- 训练产物：`experiments/test_LSPAN_x2_C64B10_L1_PixelAttention_500k`
- 测试产物：`results/test_LSPAN_x2_C64B10_L1_500k_best`
- 最终采用的模型：`experiments/test_LSPAN_x2_C64B10_L1_PixelAttention_500k/models/net_g_375000.pth`

  > 额外说明：  
  > 训练过程存在多段续训，请参考 `experiments/test_LSPAN_x2_C64B10_L1_PixelAttention_500k` 中各个阶段的log了解详细过程。

## 测试命令
```shell
conda activate ai1003-wpq
python basicsr/test.py -opt options/Test/benchmark_LSPAN_x2.yml
```
