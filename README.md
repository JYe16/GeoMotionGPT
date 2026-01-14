# GeoMotionGPT

**Official Implementation of "GeoMotionGPT: Geometry-Aligned Motion Understanding with Large Language Models"**

[![arXiv](https://img.shields.io/badge/arXiv-2601.07632-b31b1b.svg)](https://arxiv.org/abs/2601.07632)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-GeoMotionGPT-yellow)](https://huggingface.co/zy22b/GeoMotionGPT)

![Teaser](assets/overall-v7.png)

<!-- ## 🔗 Resources

- **Checkpoints:** [Download from SharePoint](https://fsu-my.sharepoint.com/:f:/g/personal/zy22b_fsu_edu/IgDx9QAD6u6PQKwZUMiNmV0zAavwneNHLTuNJDNfwlubH8A?e=yymbpA)

Please place two check point files under checkpoints/ -->

## 🛠️ Setup

### Environment and Dependencies

```bash
conda create -n gmgpt python=3.11
pip install -r requirements.txt
```

After installing the dependencies, please run:
```bash
python -m spacy download en_core_web_sm
bash prepare/download_t2m_evaluators.sh
bash prepare/prepare_gpt2.sh
```



### Dataset

Please download and preprocess data directly from [HumanML3D](https://github.com/EricGuo5513/HumanML3D).

After preprocessing data. Please place the dataset under datasets/humanml3d.

To generate motion tokens using our pretrained DVQ, please run:

```bash
python dvq/data_preprocessing/tokenize_dataset.py 
```

## Inference

After tokenizing the HumanML3D dataset using our DVQ, please run:

```bash
python test.py --cfg configs/test/m2t_o1e-2.yaml 
```

This command will evaluate our pretrained GPT2 model.

## 🖊️ Citation

If you find our work useful for your research, please consider citing:

```bibtex
@misc{ye2026geomotiongptgeometryalignedmotionunderstanding,
      title={GeoMotionGPT: Geometry-Aligned Motion Understanding with Large Language Models}, 
      author={Zhankai Ye and Bofan Li and Yukai Jin and Shuoqiu Li and Wei Wang and Yanfu Zhang and Shangqian Gao and Xin Liu},
      year={2026},
      eprint={2601.07632},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.07632}, 
}
```