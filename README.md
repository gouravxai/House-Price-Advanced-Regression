# House Price Prediction: From Basic to Advanced

This project marks my second attempt at solving the classic house price prediction problem. My first version was a basic implementation, but after diving deeper into **feature engineering**, **data distributions**, and **model selection**, I decided to rebuild it from scratch. This new approach led to a significant jump in accuracy and a much cleaner workflow.

### The Goal
The project predicts the final sale price of a home based on variables like square footage, build quality, location, and age. To make the model accessible, I deployed it as a web app where users can input house details and receive an instant price estimate.

**[Live Demo: Click here to try the app]((https://house-price-advanced-regression-bhssn8ejs5dkhptewmhb4x.streamlit.app/))**
### What I Did Differently
The biggest breakthrough came from realizing that the `SalePrice` was heavily right-skewed. Since most machine learning models perform better when data is normally distributed, I applied a **log transformation**. This simple shift made the distribution nearly normal and immediately boosted the performance of every model I tested.

I also moved beyond the raw dataset by engineering several new features:
* **Total Area:** A combined metric of basement and above-ground living space.
* **Property Age:** Calculated by comparing the year built with the year the house was sold.
* **Total Bathrooms:** A weighted combination of full and half baths.
* **Ratios:** Created features like "rooms per square foot" to capture the efficiency of a home's layout.

### Model Comparison & Results
I compared three different models by measuring their **Root Mean Squared Error (RMSE)**. Surprisingly, the simplest model came out on top:

| Model | RMSE |
| :--- | :--- |
| **Linear Regression** | **0.1267** |
| XGBoost | 0.1406 |
| Random Forest | 0.1477 |

While XGBoost and Random Forest are more powerful on paper, **Linear Regression** performed best here. By log-transforming the data, I made the relationship between the features and the target approximately linear.

### Key Takeaways
This project taught me that **feature engineering** is often more impactful than model complexity. By creating meaningful features like `HouseAge` and `TotalArea`, I provided the model with better signals than the raw data ever could.

### Tech Stack
* **Data & ML:** Python, Pandas, NumPy, Scikit-learn, XGBoost, Scipy
* **Deployment:** Streamlit, Joblib
* **Visualization:** Matplotlib
