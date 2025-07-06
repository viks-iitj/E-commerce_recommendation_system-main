from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from django.conf import settings

def home(request):
    """
    Main view for the recommendation system
    """
    recommendations = None
    search_query = None
    error_message = None

    if request.method == 'POST':
        product_title = request.POST.get('product_title', '').strip()

        if product_title:
            search_query = product_title
            try:
                # Get recommendations using your ML model
                recommendations = get_recommendations(product_title)

                # If no recommendations found
                if not recommendations:
                    error_message = f"No recommendations found for '{product_title}'. Please try a different product name."

            except Exception as e:
                error_message = f"Error getting recommendations: {str(e)}"
                print(f"Recommendation error: {e}")

    return render(request, 'index.html', {
        'recommendations': recommendations,
        'search_query': search_query,
        'error_message': error_message
    })

def get_recommendations(product_title, num_recommendations=12):
    """
    Get product recommendations based on the input product title
    """
    try:
        # Load your preprocessed data and models
        # Adjust these paths based on your project structure
        data_path = os.path.join(settings.BASE_DIR, 'data.pkl')  # or wherever your data file is
        model_path = os.path.join(settings.BASE_DIR, 'kmeans_model.pkl')  # adjust path

        # Load data
        if os.path.exists(data_path):
            df = pd.read_pickle(data_path)
        else:
            # If pickle file doesn't exist, create sample data for testing
            df = create_sample_data()

        # Clean and preprocess the input
        product_title_clean = product_title.lower().strip()

        # Method 1: Direct title matching (simple approach)
        recommendations = get_recommendations_by_similarity(df, product_title_clean, num_recommendations)

        # Method 2: If you have ML models, use them
        # recommendations = get_recommendations_by_ml(df, product_title_clean, num_recommendations)

        return recommendations

    except Exception as e:
        print(f"Error in get_recommendations: {e}")
        return []

def get_recommendations_by_similarity(df, product_title, num_recommendations=12):
    """
    Get recommendations using text similarity
    """
    try:
        # Ensure required columns exist
        required_columns = ['title', 'price', 'rating', 'total_ratings', 'category']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            print(f"Missing columns: {missing_columns}")
            return create_sample_recommendations(product_title, num_recommendations)

        # Clean the dataframe
        df_clean = df.dropna(subset=['title']).copy()
        df_clean['title_lower'] = df_clean['title'].astype(str).str.lower()

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)

        # Fit and transform the titles
        tfidf_matrix = vectorizer.fit_transform(df_clean['title_lower'])

        # Transform the input query
        query_vector = vectorizer.transform([product_title.lower()])

        # Calculate cosine similarity
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Get top recommendations
        top_indices = similarity_scores.argsort()[-num_recommendations-1:-1][::-1]

        recommendations = []
        for idx in top_indices:
            if similarity_scores[idx] > 0:  # Only include if there's some similarity
                product = df_clean.iloc[idx]
                recommendations.append({
                    'title': str(product['title']),
                    'price': format_price(product.get('price', 0)),
                    'rating': float(product.get('rating', 0)),
                    'total_ratings': int(product.get('total_ratings', 0)),
                    'category': str(product.get('category', 'General')),
                    'subcategory': str(product.get('subcategory', '')),
                    'similarity_score': float(similarity_scores[idx])
                })

        return recommendations[:num_recommendations]

    except Exception as e:
        print(f"Error in similarity calculation: {e}")
        return create_sample_recommendations(product_title, num_recommendations)

def get_recommendations_by_ml(df, product_title, num_recommendations=12):
    """
    Get recommendations using your trained ML models (K-means, etc.)
    """
    try:
        # Load your trained models
        model_path = os.path.join(settings.BASE_DIR, 'kmeans_model.pkl')
        preprocessor_path = os.path.join(settings.BASE_DIR, 'preprocessor.pkl')

        if os.path.exists(model_path) and os.path.exists(preprocessor_path):
            kmeans_model = joblib.load(model_path)
            preprocessor = joblib.load(preprocessor_path)

            # Your ML-based recommendation logic here
            # This is where you'd use your trained models

            # For now, fall back to similarity-based approach
            return get_recommendations_by_similarity(df, product_title, num_recommendations)
        else:
            # Fall back to similarity-based approach
            return get_recommendations_by_similarity(df, product_title, num_recommendations)

    except Exception as e:
        print(f"Error in ML-based recommendations: {e}")
        return get_recommendations_by_similarity(df, product_title, num_recommendations)

def format_price(price):
    """
    Format price for display
    """
    try:
        if pd.isna(price) or price == 0:
            return "N/A"

        # Remove currency symbols and convert to float
        price_str = str(price).replace('$', '').replace(',', '').replace('₹', '')
        price_float = float(price_str)

        return f"{price_float:.2f}"
    except:
        return "N/A"

def create_sample_data():
    """
    Create sample data for testing when actual data file is not available
    """
    sample_data = {
        'title': [
            'Apple iPhone 13 Pro Max 128GB',
            'Samsung Galaxy S21 Ultra 5G',
            'Sony WH-1000XM4 Wireless Headphones',
            'MacBook Pro 14-inch M1 Pro',
            'Dell XPS 13 Laptop',
            'iPad Pro 11-inch',
            'AirPods Pro with MagSafe',
            'Samsung 65" 4K Smart TV',
            'PlayStation 5 Console',
            'Xbox Series X',
            'Nintendo Switch OLED',
            'Apple Watch Series 7',
            'Fitbit Versa 3',
            'Canon EOS R5 Camera',
            'Sony A7III Mirrorless Camera',
            'Bose QuietComfort 35 II',
            'JBL Charge 5 Bluetooth Speaker',
            'Logitech MX Master 3 Mouse',
            'Mechanical Gaming Keyboard',
            'Gaming Monitor 27-inch 4K'
        ],
        'price': [1099.99, 999.99, 279.99, 1999.99, 1299.99, 799.99, 179.99, 899.99, 499.99, 499.99,
                  349.99, 399.99, 199.99, 3899.99, 1999.99, 299.99, 149.99, 99.99, 159.99, 399.99],
        'rating': [4.5, 4.3, 4.7, 4.8, 4.4, 4.6, 4.2, 4.1, 4.9, 4.7, 4.5, 4.3, 4.0, 4.8, 4.6, 4.4, 4.2, 4.5, 4.3, 4.4],
        'total_ratings': [15420, 8930, 12450, 5670, 7890, 9870, 18900, 3450, 25670, 18900, 12340, 11230, 6780, 1230, 2340, 8900, 4560, 7890, 3450, 5670],
        'category': ['Electronics', 'Electronics', 'Audio', 'Computers', 'Computers', 'Electronics', 'Audio', 'Electronics', 'Gaming', 'Gaming',
                     'Gaming', 'Wearables', 'Wearables', 'Cameras', 'Cameras', 'Audio', 'Audio', 'Accessories', 'Gaming', 'Computers']
    }

    return pd.DataFrame(sample_data)

def create_sample_recommendations(query, num_recommendations=12):
    """
    Create sample recommendations for testing
    """
    sample_products = [
        {'title': f'Related Product to {query} - Premium Model', 'price': '299.99', 'rating': 4.5, 'total_ratings': 1250, 'category': 'Electronics'},
        {'title': f'{query} - Professional Edition', 'price': '459.99', 'rating': 4.7, 'total_ratings': 890, 'category': 'Professional'},
        {'title': f'Best {query} Alternative', 'price': '199.99', 'rating': 4.3, 'total_ratings': 2340, 'category': 'Popular'},
        {'title': f'{query} - Budget Friendly', 'price': '89.99', 'rating': 4.1, 'total_ratings': 567, 'category': 'Budget'},
        {'title': f'Premium {query} with Warranty', 'price': '699.99', 'rating': 4.8, 'total_ratings': 1890, 'category': 'Premium'},
        {'title': f'{query} - Latest Model 2024', 'price': '399.99', 'rating': 4.6, 'total_ratings': 1456, 'category': 'Latest'},
        {'title': f'Top Rated {query}', 'price': '249.99', 'rating': 4.9, 'total_ratings': 3450, 'category': 'Top Rated'},
        {'title': f'{query} - Bestseller', 'price': '179.99', 'rating': 4.4, 'total_ratings': 2890, 'category': 'Bestseller'},
        {'title': f'Upgraded {query} Pro', 'price': '549.99', 'rating': 4.7, 'total_ratings': 1123, 'category': 'Pro'},
        {'title': f'{query} - Customer Choice', 'price': '129.99', 'rating': 4.2, 'total_ratings': 2567, 'category': 'Popular'},
        {'title': f'Advanced {query} System', 'price': '799.99', 'rating': 4.8, 'total_ratings': 890, 'category': 'Advanced'},
        {'title': f'{query} - Special Edition', 'price': '349.99', 'rating': 4.5, 'total_ratings': 1678, 'category': 'Special'}
    ]

    return sample_products[:num_recommendations]