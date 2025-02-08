🏷️ Sales Price Prediction using Machine Learning & Streamlit

📌 Overview

This project utilizes **Linear Regression** to predict the **Total Price** of a product based on multiple input features. It includes a **Streamlit-based web application** for user interaction.

 📊 Features
- **ML Model**: Linear Regression trained using scikit-learn.
- **Inputs (X)**:
  - `beverage_category`
  - `product_cat`
  - `customer_cat`
  - `region_cat`
  - `Unit_Price`
  - `Quantity`
  - `Discount`
- **Output (Y)**: `Total_Price`
- **Frontend**: Built with Streamlit for real-time prediction.

 🛠️ Tech Stack
- Python (Pandas, NumPy, Scikit-Learn)
- Streamlit (Web UI)
- Jupyter Notebook (for model training)

 🚀 How to Run
1. **Clone the Repository**
   ```sh
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
2. vscode terminal
   python -m streamlit run salary.py
   
📊 Model Training
The dataset is preprocessed and split into training and testing sets.
The Linear Regression model is trained using sklearn.linear_model.LinearRegression.
Model evaluation metrics include Mean Squared Error (MSE) and R² Score.

📌 Future Improvements
Implement hyperparameter tuning for better accuracy.
Support for multiple regression models.
Deploy the model using Docker / Cloud platforms.

📝 Author
Developed by Praveen R
