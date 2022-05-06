# Time-Series Generative Adversarial Networks for Cyber-Physical Systems

## 1. About
Open-source Cyber-Physical Systems (CPS) dataset is rare, thus having a tool that can generate quality (temporal-preserved) synthetic dataset is needed. TimeGAN is a tool that can generate temporal-preserve dataset. This repository will apply TimeGAN towards two multivariate time-series cyber-physical systems, BATADAL and SWaT, using Tensorflow and PyTorch.

## 2. Explanation
TimeGAN contains four components: Embedding function, Recovery function, Generator and Discriminator. There are three training phases of TimeGAN
- 1. An Autoencoder (contains of Embedding and Recovery functions) is trained with the real dataset to get a very good feature-map function that reconstruct high-dim > low-dim > high-dim data.
- 2. A Supervisor model in latent-space is trained using supervised learning to learn the temporal dependency of the real data in low-level latent-space.
- 3. TimeGAN is then jointly trained with all components train together with three losses: 
  - Reconstruction Loss of Recovery function in high-dim.
  - Supervised Loss of the Supervisor model in latent-space.
  - Classification Loss of the Discriminator as both regression and binary.

![alt-text](https://github.com/mnguyen0226/tsgans_cps/blob/main/imgs/timegan_blocks.png)

## 3. Reproduction
```
python 3.7.13
numpy = 1.21.6
tensorflow = 2.4.1
matplotlib
sklearn = 1.0.2
pip install datagene
pip install pystan
pip install pydot
conda install -c conda-forge google-colab
conda install -c anaconda portpicker
conda install -c anaconda jinja2
conda install -c anaconda requests
conda install -c conda-forge charset-normalizer
conda install zipp
conda install typing-extensions
```

**Training Notebooks:**
- [Sine Wave](https://github.com/mnguyen0226/tsgans_cps/blob/main/src/tf_version/notebooks/time_gans/time_gan_sine_wave.ipynb)
- [BATADAL Dataset](https://github.com/mnguyen0226/tsgans_cps/blob/main/src/tf_version/notebooks/time_gans/time_gan_batadal.ipynb)
- [SWaT Dataset](https://github.com/mnguyen0226/tsgans_cps/blob/main/src/tf_version/notebooks/time_gans/time_gan_swat.ipynb)

**Evaluating Notebooks:**
- [Sine Wave](https://github.com/mnguyen0226/tsgans_cps/blob/main/src/tf_version/notebooks/evaluate/evaluate_synthetic_sine_wave.ipynb)
- [BATADAL Dataset](https://github.com/mnguyen0226/tsgans_cps/blob/main/src/tf_version/notebooks/evaluate/evaluate_synthetic_batadal.ipynb)
- [SWaT Dataset](https://github.com/mnguyen0226/tsgans_cps/blob/main/src/tf_version/notebooks/evaluate/evaluate_synthetic_swat.ipynb)

## 4. Results

![alt-text](https://github.com/mnguyen0226/tsgans_cps/blob/main/imgs/swat_visual.png)
![alt-text](https://github.com/mnguyen0226/tsgans_cps/blob/main/imgs/batadal_visual.png)
![alt-text](https://github.com/mnguyen0226/tsgans_cps/blob/main/imgs/sine_visual.png)

## 5. Report

[Presentation](https://github.com/mnguyen0226/tsgans_cps/blob/main/docs/presentation.pdf)
[Report](https://github.com/mnguyen0226/tsgans_cps/blob/main/docs/report.pdf)

## 6. References
Yoon, Jinsung, Daniel Jarrett, and Mihaela Van der Schaar. "Time-series generative adversarial networks." Advances in Neural Information Processing Systems 32 (2019).

Goh, Jonathan, et al. "A dataset to support research in the design of secure water treatment systems." International conference on critical information infrastructures security. Springer, Cham, 2016.

Taormina, Riccardo, et al. "Battle of the attack detection algorithms: Disclosing cyber attacks on water distribution networks." Journal of Water Resources Planning and Management 144.8 (2018): 04018048.
