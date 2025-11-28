# Music Genre Classification with Deep Learning

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)
![Librosa](https://img.shields.io/badge/Audio-Librosa-green)
![Status](https://img.shields.io/badge/Status-Complete-success)

## Project Overview
As a musician of 25+ years, distinguishing between Heavy Metal and Jazz is instant and intuitive. But can an AI do the same just by "looking" at the sound waves?

This project tackles the problem of classifying audio tracks into 10 distinct genres using Deep Learning. 

What started as a simple comparison between **mathematical feature engineering** (MLP) and **computer vision** (CNN) evolved into a deep dive into optimization, solving the "Small Data" problem via augmentation, and overcoming the vanishing gradient problem using **ResNet architectures**.

---

## The Dataset
**Source:** [GTZAN Genre Collection](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification)  
**Content:** 1,000 audio tracks (30 seconds each).  
**Classes:** 10 Genres (Blues, Classical, Country, Disco, Hiphop, Jazz, Metal, Pop, Reggae, Rock).

---

## Methodology & Evolution

The project followed a scientific method of hypothesis, experimentation, failure analysis, and optimization.

### Phase 1: Multilayer perceptron
I extracted 57 numerical features (MFCCs, Spectral Centroid, Zero Crossing Rate, RMS) using `Librosa` and trained a Multi-Layer Perceptron.
* **Result:** Strong Baseline (~80% Accuracy).
* **Limitation:** The model seemed to be overfitting quite a bit.

### Phase 2: CNN
I used spectrograms from the dataset to train a basic CNN
* **Result:** Failure (~50% Accuracy).
* **Diagnosis:** **Severe Overfitting.** With only 800 training images, the model memorized the data rather than learning features.

### Phase 3: Transfer learning
I attempted to use **MobileNetV2** (pre-trained on ImageNet) to solve the data shortage.
* **Result:** Failure (~62% Accuracy).
* **Diagnosis:** **Domain Mismatch.** MobileNet detects natural objects (eyes, wheels, fur). Spectrograms are heatmaps of frequency. The visual features did not translate.

### Phase 4: Data slicing
To address the lack of data problem, I implemented a simple data augomentation techniques – I sliced spectrograms into 3sec pieces increasing the dataset from **1,000 to 10,000 samples**.
1.  **Result:** 77.17% accuracy. The strongest CNN so far!

### Phase 5: The Solution – ResNet!
Finally, I deployed ResNet architecture with a scheduler and optimized learning rate. 
1.  **Result:** 88.83%
2.  **Diagnosis**: ResNet architecture with skip connections solved the **Vanishing Gradient** problem, paired with a **Learning Rate Scheduler** for precise convergence.

---

## Final Results

| Model Phase | Architecture | Accuracy | Key Finding |
| :--- | :--- | :--- | :--- |
| **Model A** | Multilayer Perceptron | 80.00% | Math features have strong predictive power but hit a ceiling. |
| **Model B** | Simple CNN | 50.75% | Failed due to data scarcity and overfitting. |
| **Model C** | Transfer Learning | 62.81% | Failed due to domain mismatch (Photos vs. Spectrograms). |
| **Model D** | **ResNet + Augmentation** | **88.83%** | **Success.** Slicing solved the data size issue; ResNet solved the depth issue. |

> **Conclusion:** Deep Learning, when supported by sufficient data and correct architecture, outperformed traditional feature engineering by **+8.63%**.

---

## Visualizations

ResNet learning process
<img width="1307" height="544" alt="Screenshot 2025-11-28 at 09 19 25" src="https://github.com/user-attachments/assets/db73b742-4769-4b46-9ab6-aa6b3a551d8c" />

Confusion matrix for ResNet (rock is the msot confusing genres!)
<img width="785" height="699" alt="Screenshot 2025-11-28 at 09 16 34" src="https://github.com/user-attachments/assets/ef401697-a900-4c4d-8540-fa4c24cd6ae7" />


---

## Tech Stack
* **Deep Learning:** TensorFlow / Keras
* **Audio Processing:** Librosa
* **Data Manipulation:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn

## How to Run
1.  Clone the repository:
    ```bash
    git clone [https://github.com/YOUR_USERNAME/music-genre-classification.git](https://github.com/YOUR_USERNAME/music-genre-classification.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Open the Jupyter Notebook:
    ```bash
    jupyter notebook msai_dl_music.ipynb
    ```

## References
* Doshi, K. (2021, February 19). Audio deep learning made simple – Why mel spectrograms perform better. Ketan Doshi. https://ketanhdoshi.github.io/Audio-Mel/
* GeeksforGeeks. (2024). Residual Networks (ResNet) – Deep Learning. Retrieved from https://www.geeksforgeeks.org/deep-learning/residual-networks-resnet-deep-learning/
* McFee, B., Raffel, C., Liang, D., Ellis, D. P., McVicar, M., Battenberg, E., & Nieto, O. (2015). librosa: Audio and music signal analysis in python. Proceedings of the 14th Python in Science Conference, 8, 18-25. doi:10.25080/Majora-7b98e3ed-003
* McFee, B., et al. (2024). Librosa 0.10.2 Documentation. Retrieved from https://librosa.org/doc/latest/index.html
* Sharma, T. (2024, May 6). Detailed Explanation of Residual Network (ResNet50) CNN Model. Medium. Retrieved from https://medium.com/@sharma.tanish096/detailed-explanation-of-residual-network-resnet50-cnn-model-106e0ab9fa9e
* Tzanetakis, G., & Cook, P. (2002). Musical genre classification of audio signals. IEEE Transactions on speech and audio processing, 10(5), 293-302.
* Weirich, A. (2024, June 18). Transfer learning with Keras/TensorFlow: An introduction. Medium. https://medium.com/@alfred.weirich/transfer-learning-with-keras-tensorflow-an-introduction-51d2766c30ca
* Weirich, A. (2024, July 4). Finetuning TensorFlow/Keras networks: Basics using MobileNetV2 as an example. Medium. https://medium.com/@alfred.weirich/finetuning-tensorflow-keras-networks-basics-using-mobilenetv2-as-an-example-8274859dc232
