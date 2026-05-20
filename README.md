## House Price Prediction — From Basic to Advanced

Live Demo: [Try the predictor here](https://house-price-advanced-regression-bhssn8ejs5dkhptewmhb4x.streamlit.app/)

### Overview

This is my second attempt at house price prediction. The first version was basic - just throw data into a model and hope. This time I focused on feature engineering and understanding why certain models work better than others.

It turned out the simplest model won. That taught me something important.

### The Key Insight

When I first explored the data, I noticed the sale prices were heavily skewed - most houses were cheap but a few were very expensive. Most machine learning models assume normally distributed data.

I applied a log transformation to the target variable and suddenly everything made more sense. The relationship between features and price became linear.

This is why I built the model this way, and why it works so well.

### Feature Engineering

Rather than using raw variables, I created features that actually mean something:

- Total Area: Combines basement + above-ground space (more relevant than either alone)
- Property Age: Year sold minus year built (recency matters)
- Total Bathrooms: Weighted combination of full and half baths
- Space Efficiency: Rooms per square foot (tells you about layout quality)
- Quality Score: Interaction of building quality and condition

These features capture the business logic of home value better than raw numbers.

### Model Comparison

I tested three approaches:

| Model | RMSE |
|-------|------|
| Linear Regression | 0.1267 |
| XGBoost | 0.1406 |
| Random Forest | 0.1477 |

Linear Regression won. This surprised me at first - XGBoost and Random Forest are more "powerful" models. But they were overfitting.

The lesson: after log transformation and good feature engineering, the relationship between features and price is largely linear. Complex models just fit noise.

### Technical Stack

- Data Processing: Python, Pandas, NumPy
- Machine Learning: Scikit-learn, XGBoost
- Deployment: Streamlit
- Visualization: Matplotlib

### What I Did Differently

Most house price projects skip feature engineering and jump to modeling. I spent time understanding the data first.

I also didn't just optimize for accuracy. I looked at residuals to understand where the model failed. Some houses have weird characteristics that make them outliers - the model doesn't predict those well, and that's okay.

### What I Learned

Feature engineering beats model complexity, at least for this problem. You can't machine-learn your way out of bad features.

Also learned that log transformation of skewed targets is really powerful. It's a simple technique that gets overlooked.

Finally realized that the best model isn't always the most sophisticated. Simpler models are easier to explain, faster to run, and less likely to overfit.

### Deployment

The Streamlit app lets you adjust features (square footage, year built, quality) and see the predicted price in real-time. Useful for understanding how different factors affect value.

### Limitations

- Trained on house prices from a specific region (may not generalize)
- Doesn't account for unique features like location-specific amenities
- Model struggles with very old or very new houses (limited training data)
- Market cycles affect prices (training data from one period may not work for another)

### Future Improvements

- Include location-based features (zip code, proximity to schools)
- Add temporal features to account for market changes
- Ensemble multiple models with different feature sets
- Incorporate external data (school ratings, crime statistics)
