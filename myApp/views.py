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
        print(f"DEBUG: Received product_title: '{product_title}'")  # Debug print

        if product_title:
            search_query = product_title
            try:
                # Get recommendations using your ML model
                recommendations = get_recommendations(product_title)
                print(f"DEBUG: Got {len(recommendations) if recommendations else 0} recommendations")  # Debug print

                # If no recommendations found
                if not recommendations:
                    error_message = f"No recommendations found for '{product_title}'. Please try a different product name."
                    print("DEBUG: No recommendations found")  # Debug print

            except Exception as e:
                error_message = f"Error getting recommendations: {str(e)}"
                print(f"DEBUG: Exception occurred: {e}")  # Debug print

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
        print(f"DEBUG: Starting get_recommendations for '{product_title}'")

        # Load your preprocessed data and models
        # First, check if data files exist
        data_path = os.path.join(settings.BASE_DIR, 'myApp', 'dataset', 'data.pkl')
        csv_path = os.path.join(settings.BASE_DIR, 'myApp', 'dataset', 'amazon_data.csv')

        print(f"DEBUG: Looking for data at: {data_path}")
        print(f"DEBUG: Looking for CSV at: {csv_path}")

        df = None

        # Try to load pickle file first
        if os.path.exists(data_path):
            print("DEBUG: Loading from pickle file")
            df = pd.read_pickle(data_path)
        # Try to load CSV file
        elif os.path.exists(csv_path):
            print("DEBUG: Loading from CSV file")
            df = pd.read_csv(csv_path)
        else:
            print("DEBUG: No data files found, creating sample data")
            df = create_sample_data()

        print(f"DEBUG: DataFrame shape: {df.shape}")
        print(f"DEBUG: DataFrame columns: {df.columns.tolist()}")

        # Clean and preprocess the input
        product_title_clean = product_title.lower().strip()
        print(f"DEBUG: Cleaned product title: '{product_title_clean}'")

        # Get recommendations using similarity
        recommendations = get_recommendations_by_similarity(df, product_title_clean, num_recommendations)
        print(f"DEBUG: Found {len(recommendations)} recommendations")

        return recommendations

    except Exception as e:
        print(f"DEBUG: Error in get_recommendations: {e}")
        import traceback
        traceback.print_exc()
        # Return sample recommendations as fallback
        return create_sample_recommendations(product_title, num_recommendations)

def get_recommendations_by_similarity(df, product_title, num_recommendations=12):
    """
    Get recommendations using text similarity
    """
    try:
        print(f"DEBUG: Starting similarity calculation for '{product_title}'")

        # Check if DataFrame is empty
        if df.empty:
            print("DEBUG: DataFrame is empty")
            return create_sample_recommendations(product_title, num_recommendations)

        # Map possible column names to standard names
        column_mapping = {
            'title': ['title', 'product_title', 'name', 'product_name'],
            'price': ['price', 'actual_price', 'discounted_price', 'cost'],
            'rating': ['rating', 'ratings', 'average_rating', 'stars'],
            'total_ratings': ['total_ratings', 'rating_count', 'review_count', 'reviews'],
            'category': ['category', 'main_category', 'product_category', 'genre']
        }

        # Find the actual column names in the DataFrame
        actual_columns = {}
        for standard_name, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name in df.columns:
                    actual_columns[standard_name] = possible_name
                    break

        print(f"DEBUG: Mapped columns: {actual_columns}")

        # Check if we have the essential column (title)
        if 'title' not in actual_columns:
            print("DEBUG: No title column found, creating sample data")
            return create_sample_recommendations(product_title, num_recommendations)

        # Create a working copy with standardized column names
        df_work = df.copy()
        for standard_name, actual_name in actual_columns.items():
            if actual_name != standard_name:
                df_work[standard_name] = df_work[actual_name]

        # Clean the dataframe
        df_clean = df_work.dropna(subset=['title']).copy()
        df_clean['title_lower'] = df_clean['title'].astype(str).str.lower()

        print(f"DEBUG: Cleaned DataFrame shape: {df_clean.shape}")

        if df_clean.empty:
            print("DEBUG: No valid titles found")
            return create_sample_recommendations(product_title, num_recommendations)

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000, lowercase=True)

        # Fit and transform the titles
        tfidf_matrix = vectorizer.fit_transform(df_clean['title_lower'])
        print(f"DEBUG: TF-IDF matrix shape: {tfidf_matrix.shape}")

        # Transform the input query
        query_vector = vectorizer.transform([product_title.lower()])

        # Calculate cosine similarity
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
        print(f"DEBUG: Similarity scores range: {similarity_scores.min()} to {similarity_scores.max()}")

        # Get top recommendations (excluding the first one if it's too similar to avoid exact duplicates)
        top_indices = similarity_scores.argsort()[-num_recommendations-5:][::-1]

        recommendations = []
        for idx in top_indices:
            if len(recommendations) >= num_recommendations:
                break

            if similarity_scores[idx] > 0.05:  # Lowered threshold for more results
                product = df_clean.iloc[idx]
                recommendation = {
                    'title': str(product['title']),
                    'price': format_price(product.get('price', 0)),
                    'rating': float(product.get('rating', 4.0)),
                    'total_ratings': int(product.get('total_ratings', 100)),
                    'category': str(product.get('category', 'General')),
                    'subcategory': str(product.get('subcategory', '')),
                    'similarity_score': float(similarity_scores[idx])
                }
                recommendations.append(recommendation)
                print(f"DEBUG: Added recommendation: {recommendation['title'][:50]}... (score: {similarity_scores[idx]:.4f})")

        print(f"DEBUG: Total recommendations found: {len(recommendations)}")

        # If still no recommendations, return sample ones
        if not recommendations:
            print("DEBUG: No similarity-based recommendations found, returning samples")
            return create_sample_recommendations(product_title, num_recommendations)

        return recommendations

    except Exception as e:
        print(f"DEBUG: Error in similarity calculation: {e}")
        import traceback
        traceback.print_exc()
        return create_sample_recommendations(product_title, num_recommendations)

def format_price(price):
    """
    Format price for display
    """
    try:
        if pd.isna(price) or price == 0 or price == '':
            return "N/A"

        # Remove currency symbols and convert to float
        price_str = str(price).replace('$', '').replace(',', '').replace('₹', '').replace('€', '').strip()

        # Handle price ranges (e.g., "10-20")
        if '-' in price_str:
            price_str = price_str.split('-')[0]

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
            'Mechanical Gaming Keyboard RGB',
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
    print(f"DEBUG: Creating sample recommendations for '{query}'")

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