# Medical Cost Predictions - Machine Learning

Welcome to the Medical Cost Predictions project! In this project, we aim to predict the cost of medical assistance a person has to pay based on various characteristics. We'll be using a RandomForestRegressor model and exploring concepts like outlier removal, Flask API development, CI/CD pipeline setup, and DVC (Data Version Control) integration.

## Project Overview

- Dataset: The dataset contains information about individuals' characteristics and the medical costs they had to pay.
- Machine Learning Model: We're using a RandomForestRegressor to predict medical costs.
- Concepts Explored: Outlier removal, Flask API, CI/CD pipeline, DVC integration.

## Table of Contents

- [Dataset](#dataset)
- [Machine Learning Model](#machine-learning-model)
- [Concepts Explored](#concepts-explored)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Dataset

Our dataset contains information about individuals' characteristics, including sex, smoking status, and more. This data is used to predict the medical costs individuals have to pay.

## Machine Learning Model

We've built a RandomForestRegressor model to predict medical costs. The model is trained on the medical cost datapoints and is used for predictions.

## Concepts Explored

In this project, we explore the following concepts:

- **Outlier Removal:** We delve into the importance of outlier removal in improving the model's accuracy and reliability.
- **Flask API Development:** We develop a Flask-based API to deploy our trained model, allowing users to make predictions via API requests.
- **CI/CD Pipeline:** We've set up a continuous integration and continuous deployment (CI/CD) pipeline to automate the testing and deployment process.
- **DVC Integration:** Data Version Control (DVC) is integrated to manage and version data effectively.

## Getting Started

To get started with the project, follow these steps:

1. Clone the repository: `git clone https://github.com/your-username/medical-cost-predictions.git`
2. Install the required dependencies: `pip install -r requirements.txt`


## Usage

1. Train the model: Run `python train_model.py` to train the RandomForestRegressor model.
2. Run the Flask API: Start the Flask API by running `python app.py`.
3. Make Predictions: Use API requests to make predictions based on the provided characteristics.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to submit a pull request.
![Screenshot (97)](https://github.com/pras-ops/Medical_Cost_Predictions/assets/56476064/797db57b-e85f-4178-bc11-9852221b7d47)

## License

This project is licensed under the [MIT License](LICENSE).
