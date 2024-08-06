# English or Spanish?

Language prediction using various deep learning algorithms, implemented in JAX. I built three different word-level models, each with the task of detecting the language of a given sentence.

 Alongside this repo, I have also written a [blog article](https://asandei.com/posts/lang-detection-jax) with explanations on how to build these models using the JAX framework.

## Models

The following model architectures were implemented, with the respective validation loss values and accuracies.

- MLP: loss of 0.96 and 66.72% accuracy
- RNN: loss of 0.61 and 80.17% accuracy
- LSTM: loss of 0.27 and 95.27% accuracy

## Dataset

Dataset used: [https://www.kaggle.com/datasets/basilb2s/language-detection](https://www.kaggle.com/datasets/basilb2s/language-detection?). Download the CSV file to the root of the repo and name the file `data.csv`.

## License

[GPLv3](LICENSE) © [Asandei Stefan-Alexandru](https://asandei.com) 2024.

