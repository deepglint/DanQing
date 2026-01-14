<p align="center">
    <img src="Figures/logo.png" width="50%">
</p>

## DanQing: An Up-to-Date Large-Scale Chinese Vision-Language Pre-training Dataset
Hengyu Shen<sup>∗</sup>,</span>
<a href="https://github.com/GaryGuTC">Tiancheng Gu<sup>∗</sup></a>,</span>
Bin Qin,</span>
Lan Wu,</span>
Yuling Wu,</span>
Shuo Tan,</span>
Zelong Sun,</span>
Jun Wang,</span>
Nan Wu,</span>
<a href="https://github.com/anxiangsir">Xiang An</a>,</span>
<a href="https://weidong-tom-cai.github.io/">Weidong Cai</a>,</span>
Ziyong Feng<sup>‡</sup>,</span>
<a href="https://kaicheng-yang0828.github.io">Kaicheng Yang<sup>†</sup></a> </span> 

∗ Equal Contribution. † Project Leader. ‡ Team Leader.




## 📣 News
- [2026/01/16]:✨We release the [paper]() of DanQing.
- [2026/01/15]:🔥We release the DanQing dataset (images and captions, about 12T) in [<img src="Figures/modelscope.png" alt="示例" width="15" height="12">ModelScope](https://www.modelscope.cn/datasets/deepglint/DanQing)
- [2026/01/13]:✨We release the DanQing dataset (URLs of image and captions) in [🤗Hugging Face](https://huggingface.co/datasets/DeepGlint-AI/DanQing100M)

❗️<font color=#ff7b7a>Note: Due to the storage and transmission limitations of Hugging Face, we only release the URLs corresponding to the images on Hugging Face. To access the complete dataset, please download it from ModelScope.</font>


## 💡 Introduction
<p align="center">
    <img src="Figures/framework.png" width="100%">
</p>
In this paper, we propose DanQing dataset, which contains 100 million image-text pairs collected from Common Crawl. Different from existing datasets, DanQing is curated through a more rigorous selection process, yielding superior data quality. Moreover, DanQing is primarily built from 2024–2025 web data, enabling models to better capture evolving semantic trends and thus offering greater practical utility. We compare DanQing with existing datasets by conduct continual pre-training of the SigLIP2 model. Experimental results show that DanQing consistently achieves superior performance across a range of Chinese downstream tasks, including zero-shot classification, cross-modal retrieval, and LMM-based evaluations. 
To facilitate further research in Chinese vision-language pre-training, we will open-source the DanQing dataset under the Creative Common CC-BY 4.0 license.

## 💻 Dataset Information
### Data Preview
<p align="center">
    <img src="Figures/case.png" width="100%">
</p>

### Topic Assessment
<p align="center">
    <img src="Figures/topic_examples.png" width="100%">
</p>

### Text Length and Image Resolution Distribution

<p align="center">
    <img src="Figures/statistic.png" width="100%">
</p>

### Text Quality

<p align="center">
    <img src="Figures/quality.png" width="100%">
</p>

### Cosine Similarity and Semantic Distribution
<p align="center">
    <img src="Figures/distribution.png" width="100%">
</p>


## 📃 Performance Comparison
### Zero-Shot Classification
<p align="center">
    <img src="Figures/classification.png" width="80%">
</p>

### Cross-Modal Retrieval(short caption)
<p align="center">
    <img src="Figures/short.png" width="100%">
</p>

### Cross-Modal Retrieval(long caption)
<p align="center">
    <img src="Figures/long.png" width="100%">
</p>

### Chinese-Centric Large Multimodal Model Tasks
<p align="center">
    <img src="Figures/LMM.png" width="80%">
</p>

## 🧠 Analysis
### Data and Model Scaling
<p align="center">
    <img src="Figures/scaling.png" width="100%">
</p>

### New Concept Understanding
<p align="center">
    <img src="Figures/new_concept.png" width="100%">
</p>

## License
The DanQing dataset is licensed under [CC-BY-4.0 License](https://creativecommons.org/licenses/by/4.0/).
The full license can be found in the [LICENSE.cc-by-4.0 file](./LICENSE.cc-by-4.0).
The dataset is collected from various sites by analyzing Common Crawl data, an open data web crawling project. 
The collected data is subject to the license to which each content belongs.

## Citation
If you find this repository useful, please use the following BibTeX entry for citation.

```latex


```