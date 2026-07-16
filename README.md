Taobao user behavior analysis

This project is an industrial-grade AI recommendation system built on a massive real-world Taobao e-commerce dataset (comprising billions of interactions). It covers the complete end-to-end R&D lifecycle: from feature engineering, deep learning model training, and ablation studies, to backend microservice API deployment and an interactive frontend dashboard.

Key Features

 Strict Point-in-Time Architecture (Data Leakage Prevention):
Completely abandoned the flawed "global statistical wide table" approach that causes inflated offline metrics. Utilized Time-based Sliding Windows and shift() mechanisms to construct a 100% pure Out-of-Time (OOT) validation set, accurately simulating real-world online inference scenarios.

Wide-DIN Deep Attention Network:
Combined Wide & Deep concepts to design and implement a custom Target Attention mechanism. Compared to traditional RNN/GRU, this model dynamically captures the local relevance between candidate items and users' historical behaviors, boosting the pure OOT GAUC to 0.479 (significantly outperforming LightGBM/XGBoost baselines).

Extreme Class Imbalance Strategies:
Conducted exhaustive ablation studies against the extreme dataset imbalance (89% Views vs. 2% Purchases). Proved via empirical data that SMOTE spatial interpolation is unsuitable for high-cardinality discrete sequence features. Ultimately adopted a Gradient Class Weight Penalty (1:1.5:2:3) to successfully achieve effective recall for rare intent classes (Purchases).

Frontend-Backend Decoupled Microservices:
Moved beyond local blocking inference by building a sub-20ms response backend inference cluster using FastAPI. Crafted a highly interactive frontend dashboard with Streamlit incorporating native Taobao UI designs, delivering an enterprise-level deployment and demonstration experience.

 Repository Structure

├── data/
│   ├── raw/                # Directory for the raw Taobao dataset
│   └── processed/          # Pure datasets after Point-in-Time truncation
├── models/                 # Trained Wide-DIN .pth model weights
├── figures/                # GAUC comparisons & Feature Importance visualizations
├── src/
│   ├── pit_feature_builder.py    # Feature Eng: Time-sliced processing without leakage
│   ├── oot_pipeline.py           # Model Training: Wide-DIN architecture & pipeline
│   ├── baseline_experiments.py   # Baselines: LR, XGBoost, LightGBM comparisons
│   ├── sampling_experiments.py   # Sampling: Deep parameter testing for SMOTE & Under-sampling
│   ├── visualization.py          # Scripts: Matplotlib chart generation
│   └── api.py                    # Backend Service: FastAPI real-time inference interface
├── app.py                        # Frontend Dashboard: Streamlit full-stack Taobao UI
├── requirements.txt              # Project environment dependencies
└── evaluation_report.md          # Special Report: Metrics & ablation study results


 Quick Start

1. Environment Setup

Clone this repository and install the required Python dependencies:

git clone [https://github.com/your-username/taobao-ai-recsys.git](https://github.com/your-username/taobao-ai-recsys.git)
cd taobao-ai-recsys
pip install -r requirements.txt


2. Data Preparation

The open-source Taobao dataset used in this project has been zipped and uploaded to Google Drive. Please download the data before running the code:

📥 Dataset Download Link: Google Drive Backup

Placement Path: Please place the downloaded raw data files (e.g., UserBehavior.csv) in the data/raw/ directory of the project.

3. Training & Evaluation

Run the following commands sequentially to process data, train the model, run baseline experiments, and generate evaluation charts:

python src/pit_feature_builder.py  # 1. Build leakage-free PiT features
python src/oot_pipeline.py         # 2. Train the Wide-DIN model
python src/baseline_experiments.py # 3. Run multi-model baseline comparisons
python src/sampling_experiments.py # 4. Run deep tests on sampling strategies
python src/visualization.py        # 5. Generate visual evaluation charts


4. Launch Microservices & Dashboard

This project uses a decoupled frontend-backend architecture. You need to open two separate terminals to run them.

Terminal 1 (Start the Backend AI Inference Engine):

python src/api.py


(The API will run on http://127.0.0.1:8000 by default. Visit /docs to access the built-in Swagger UI for interactive testing.)

Terminal 2 (Start the Frontend Commercial Dashboard):

streamlit run app.py


(The dashboard will automatically open in your browser, providing a highly realistic view for monitoring and debugging the Taobao E-commerce Recommendation System.)

Experimental Results

Under strictly time-sliced OOT (Out-of-Time) datasets, traditional tree models (LightGBM) only reached a GAUC of 0.4013.

The Wide-DIN architecture incorporating the Target Attention mechanism achieved a GAUC of 0.4791, proving the absolute dominance of attention mechanisms in modeling sequential behaviors.

For detailed experimental data, SMOTE analysis, and feature selection logic, please refer to the evaluation_report.md.
