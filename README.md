# Self-supervised learning for time series

## Real-time visualization of FITS learning

<video src="NN_learning_visualization.mp4" width="776" height="756" controls>
  Your browser does not support the video tag.
</video>


## Submodules

When cloning the project, use the `--recurse-submodules` to also fetch the content from the submodule *FITS*. If the repository is already initialized, resort to the commands:  
- `git submodule init`
- `git sumbodule update`

## Motivation

Traditional machine learning approaches often rely on labeled data, where the data is annotated with the correct output for each example. However, labeling large amounts of data can be time-consuming and expensive. Self-supervised learning is a type of machine learning that allows models to learn from unlabeled data by using the data itself as the supervision signal. This allows us to learn useful representations from large unlabelled datasets and to use the representations for fine-tuning downstream tasks associated with smaller datasets.
Self-supervised learning has shown a lot of promise in domains such as computer vision [1], and recently, self-supervised learning for time series data has been gaining more attention.

## Project description

A recent method, to be presented at a conference in 2024, has developed a lightweight yet powerful model for time series analysis. The method is interesting since it achieves performance comparable to state-of-the-art models with only 10.000 parameters [2].
This project explores the recently proposed methods, and will therefore include the following steps:

1. Understanding the article and reproducing their results [2].

2. Perform anomaly detection on synthetic time-series data and EEG data.

3. Implementation of your own ideas.

## References

[1] T. Chen, S. Kornblith, M. Norouzi, and G. Hinton, “A simple framework for contrastive learning of visual representations,” in Proceedings of the 37th International Conference on Machine Learning (H. D. III and A. Singh, eds.), vol. 119 of Proceedings of Machine Learning Research, pp. 1597–1607, PMLR, 13–18 Jul 2020.

[2] Zhijian Xu, Ailing Zeng, Qiang Xu, FITS: Modeling Time Series with 10k Parameters, to be presented at The Twelfth International Conference on Learning Representations, 2024.
