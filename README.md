# Review Submission and Prediction App

This is a Flask application for submitting reviews and predicting star ratings and tags for the reviews. It uses a pretrained BERT-GRU sentiment analysis model to predict star ratings and a Word2Vec-based model to predict tags.

## Features

- **Review Submission**: Users can submit reviews with their personal star rating and text. This allows businesses to gather customer feedback directly through the application.
  
- **Star Rating Prediction**: Utilize a pretrained BERT-GRU model to predict star ratings for submitted reviews. This helps businesses understand the potential sentiment of a review even if the user does not provide a specific star rating.

- **Tag Prediction**: Use a Word2Vec-based model to automatically generate tags from review text. Tags help categorize feedback themes, making it easier for businesses to analyze common concerns or praises.

- **Review Analytics**: View all submitted reviews with their original and predicted star ratings, along with generated tags. This feature allows businesses to monitor and analyze customer feedback trends over time.

- **Business Insights Dashboard**: Navigate to specific views (`/reviews` and `/center`) to see structured outputs of review data. These pages can act as dashboards where businesses can get quick insights into customer sentiment and feedback categories.

## Requirements
- Python 3.7+
- Flask
- PyTorch
- Transformers
- Scikit-learn
- Matplotlib
- Seaborn
## Models
- There should be a folder named 'models' under 'FlaskAPP', but it was too big to upload. Here is the download link:'https://drive.google.com/drive/folders/10_sh6Vp9ldTPM4AWSaWYkBgQhJxIL8Kf?usp=share_link', You need to install it and put it under the FlaskAPP to run the whole application.
