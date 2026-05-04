🎯 OKCupid: Generation Predictor AI
Deep Learning Based Demographic & Psychographic Analysis System
This project was developed to analyze user profiles on the popular dating platform OKCupid to predict which generation (Gen X, Millennial, or Gen Z) an individual belongs to based on their demographic data and profile essays.

🔗 Project Links
Hugging Face Application: OKCupid-Gen-Predictor

GitHub Repository: OKCupid-Generation-Analysis

📖 Project Summary
This study predicts generational affiliation by analyzing physical attributes, lifestyle choices (alcohol, smoking, diet), and specific word choices in profile essays. By processing a massive feature set of over 10,000 variables, the project provides a data-driven look at how digital footprints vary characteristically across different age groups[cite: 2].

🧠 Methods Used
Advanced Algorithm (Deep Learning): A multi-layer Artificial Neural Network (ANN) was implemented. The model consists of hidden layers with 64, 32, and 16 neurons, utilizing Dropout mechanisms to achieve stable classification performance[cite: 2].

Performance Stability: To prevent overfitting within a high-dimensional feature space (10,219 features), regularization techniques were applied during training, ensuring strong generalization capacity for real-world data[cite: 2].

Large-Scale Data Processing: Metadata from user profiles and text-mined keywords were integrated to create a comprehensive feature universe.

🛠️ Data Preprocessing & Feature Engineering
NLP & Feature Extraction: User essays were processed to identify generation-specific keywords such as 'yoga', 'writing', 'working', and 'world', which were then converted into numerical inputs[cite: 2].

Categorical Encoding & Normalization: Categorical variables like gender, body type, and occupation were encoded, while all data was normalized to a 0-1 range for optimal model training[cite: 2].

Input Shape Management: The model's requirement for a 10,219-input vector is managed through a dynamic alignment system during the inference stage to ensure error-free predictions[cite: 2].

📊 Key Results
Linguistic Divergence: Analysis revealed that Gen Z and Millennials use distinct jargons and keywords in their profiles that are statistically separable.

Lifestyle Correlations: Habits regarding alcohol, smoking, and specific career sectors were proven to have high predictive power in determining a user's generation.

🎨 Application Features
Interactive Interface: The Streamlit-based UI accepts physical traits, occupation, and text inputs to generate a generation risk analysis and probability scores within seconds.

Production Deployment: The model was packaged in .keras format with normalization layers embedded, allowing for end-to-end deployment on Hugging Face Spaces[cite: 2].

Prepared By: Zeynep Tekin