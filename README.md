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
conda activate ai1003
python setup.py develop
python basicsr/test.py -opt options/Test/benchmark_LSPAN_x2.yml
```

---

## Dependencies

```bash
conda create -n ai1003 python=3.8 -y
conda activate ai1003
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python setup.py develop
```

## Datasets

Download training and testing datasets and put them into the corresponding folders of `datasets/`. See [datasets](datasets/README.md) for the detail of the directory structure.

## Training

- Run the following scripts. The training configuration is in `options/train/`.

  ```bash
  python basicsr/train.py -opt options/train/train_simpleCNN.yml
  ```

- The training experiment is in `experiments/`.

## Testing

- Run the following scripts. The testing configuration is in `options/test/`.

  Note 1:  You can set `use_chop: True` (default: False) in YML to chop the image for testing.

  ```bash
  python basicsr/test.py -opt options/Test/test_simpleCNN.yml
  ```

- The output is in `results/`.

## Acknowledgements

This code is built on  [BasicSR](https://github.com/XPixelGroup/BasicSR).
