<p align="center">
    <img src="Figures/danqing.svg" width="50%">
</p>

## *DanQing*: An Up-to-Date Large-Scale Chinese Vision-Language Pre-training Dataset
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
- [2026/01/15]:🔥We release the DanQing dataset (images and captions, about 12T) in [<img src="Figures/modelscope.png" alt="示例" style="width:16px; height:12px;"/> ModelScope](https://www.modelscope.cn/datasets/deepglint/DanQing)
- [2026/01/13]:✨We release the DanQing dataset (URLs of image and captions) in [🤗 Hugging Face](https://huggingface.co/datasets/DeepGlint-AI/DanQing100M)

❗️<font color=#ff7b7a>Note: Due to the storage and transmission limitations of Hugging Face, we only release the URLs corresponding to the images on Hugging Face. To access the complete dataset, please download it from ModelScope.</font>


## 💡 Highlights
In this paper, we propose DanQing dataset, which contains 100 million image-text pairs collected from Common Crawl. Different from existing datasets, DanQing is curated through a more rigorous selection process, yielding superior data quality. Moreover, DanQing is primarily built from 2024–2025 web data, enabling models to better capture evolving semantic trends and thus offering greater practical utility. We compare DanQing with existing datasets by conduct continual pre-training of the SigLIP2 model. Experimental results show that DanQing consistently achieves superior performance across a range of Chinese downstream tasks, including zero-shot classification, cross-modal retrieval, and LMM-based evaluations. 
<p align="center">
    <img src="Figures/framework.png" width="100%">
</p>


## 💻 Dataset Information
### Data Preview
<p align="center">
    <img src="Figures/case.png" width="100%">
</p>

### Topic Assessment
We implement a topic modeling pipeline based on [BERTopic](https://github.com/MaartenGr/BERTopic). We randomly sample 10M image-text pairs and extract text embeddings using [Chinese-CLIP-L/14](https://github.com/OFA-Sys/Chinese-CLIP). To address high-dimensional clustering, we apply UMAP for dimensionality reduction, followed by HDBSCAN to identify semantic clusters with a minimum cluster size of 1,000 for stability and noise reduction. Finally, we use class-based TF-IDF to extract representative keywords for each topic.
<p align="center">
    <img src="Figures/topic_examples.png" width="100%">
</p>

### Image Resolution and Text Length Distribution
We analyze image resolutions by width, height, and minimum dimension, demonstrating a wide range of visual scales. We also report the distribution of text lengths across 2.2B Chinese words.
<p align="center">
    <img src="Figures/statistic.png" width="100%">
</p>

### Text Quality
We evaluate the text quality of DanQing using two metrics: semantic word density and perplexity (PPL). We randomly sample 10M texts from DanQing, Wukong, and Zero for comparison. Semantic words (nouns, verbs, adjectives) are identified using the jieba toolkit, and their proportion in each sentence is calculated as semantic density. Sentence-level perplexity is computed with a pre-trained Chinese [BERT](https://huggingface.co/google-bert/bert-base-chinese) model.
<p align="center">
    <img src="Figures/quality.png" width="100%">
</p>

### Cosine Similarity and Semantic Distribution
We analyze 10M-sample subsets of DanQing and Wukong by presenting image-text similarity distributions, extracted with [FG-CLIP2-L/16@256](https://huggingface.co/qihoo360/fg-clip2-large). For semantic distribution comparison, 10M images from each dataset are clustered into 10K groups using [FAISS](https://github.com/facebookresearch/faiss), with clusters ranked by sample count.

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
We compare the data and model scaling capabilities of DanQing and Wukong, reporting average zero-shot classification and retrieval (long & short caption) performance in below figure.
<p align="center">
    <img src="Figures/scaling.png" width="100%">
</p>

### New Concept Understanding
We evaluate SigLIP2-L/16 models pre-trained on various Chinese datasets for emergent concept understanding, and find that the model trained on DanQing consistently gives the highest confidence to correct pairs.

<p align="center">
    <img src="Figures/new_concept.png" width="100%">
</p>

## Download
### Huggingface
```
from datasets import load_dataset

ds = load_dataset("DeepGlint-AI/DanQing100M")
```

### ModelScope
```
from modelscope.msdatasets import MsDataset

ds =  MsDataset.load('deepglint/DanQing')
```

## License
The DanQing dataset is licensed under [CC-BY-4.0 License](https://creativecommons.org/licenses/by/4.0/).
The full license can be found in the [LICENSE.cc-by-4.0 file](./LICENSE.cc-by-4.0).
The dataset is collected from various sites by analyzing Common Crawl data, an open data web crawling project. 
The collected data is subject to the license to which each content belongs.

## Citation
If you find this repository useful, please use the following BibTeX entry for citation.

```latex


```

<div align="center">
⭐ Don't forget to star this repository if you find it helpful!
</div>
