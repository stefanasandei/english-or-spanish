# English or spanish?

Language prediction using various deep learning algorithms, implemented in JAX. I built three different word-level models, each with the task of detecting the language of a given sentence.

 Alongside this repo, I have also written a [blog article](https://asandei.com) with explanations on how to build these models using the JAX framework.

## Models

The following model architectures were implemented, with the respective validation loss values.

- MLP: 0.96
- RNN: 0.54
- LSTM: ???

## Dataset

Dataset used: [https://www.kaggle.com/datasets/basilb2s/language-detection](https://www.kaggle.com/datasets/basilb2s/language-detection?). Download the CSV file to the root of the repo and name the file `data.csv`.

## License

[GPLv3](LICENSE) © [Asandei Stefan-Alexandru](https://asandei.com) 2024.

