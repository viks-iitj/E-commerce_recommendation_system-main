from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from django.conf import settings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def home(request):
    """
    Main view for the recommendation system
    """
    recommendations = None
    search_query = None
    error_message = None

    if request.method == 'POST':
        product_title = request.POST.get('product_title', '').strip()
        logger.info(f"Received product_title: '{product_title}'")

        if product_title:
            search_query = product_title
            try:
                # Get recommendations using ML models
                recommendations = get_recommendations(product_title)
                logger.info(f"Got {len(recommendations) if recommendations else 0} recommendations")

                # If no recommendations found
                if not recommendations:
                    error_message = f"No recommendations found for '{product_title}'. Please try a different product name."
                    logger.info("No recommendations found")

            except Exception as e:
                error_message = f"Error getting recommendations: {str(e)}"
                logger.error(f"Exception occurred: {e}")

    return render(request, 'index.html', {
        'recommendations': recommendations,
        'search_query': search_query,
        'error_message': error_message
    })

def get_recommendations(product_title, num_recommendations=12):
    """
    Get product recommendations using the trained ML models
    """
    try:
        logger.info(f"Starting get_recommendations for '{product_title}'")

        # Define paths for data and models
        base_path = os.path.join(settings.BASE_DIR, 'myApp')
        data_path = os.path.join(base_path, 'dataset', 'data.pkl')
        csv_path = os.path.join(base_path, 'dataset', 'amazon_data.csv')

        # Model paths
        kmeans_path = os.path.join(base_path, 'models', 'kmeans_model.joblib')
        dbscan_path = os.path.join(base_path, 'models', 'dbscan_model.joblib')
        preprocessor_path = os.path.join(base_path, 'models', 'preprocessor.joblib')

        # Load data
        df = load_data(data_path, csv_path)
        if df is None or df.empty:
            logger.warning("No valid data found, using sample data")
            return create_sample_recommendations(product_title, num_recommendations)

        # Load models if they exist
        models = load_models(kmeans_path, dbscan_path, preprocessor_path)

        # Get recommendations
        if models['kmeans'] is not None and models['preprocessor'] is not None:
            logger.info("Using trained models for recommendations")
            recommendations = get_ml_recommendations(df, product_title, models, num_recommendations)
        else:
            logger.info("Using similarity-based recommendations")
            recommendations = get_similarity_recommendations(df, product_title, num_recommendations)

        return recommendations

    except Exception as e:
        logger.error(f"Error in get_recommendations: {e}")
        import traceback
        traceback.print_exc()
        return create_sample_recommendations(product_title, num_recommendations)

def load_data(data_path, csv_path):
    """
    Load data from pickle or CSV file
    """
    try:
        # Try to load pickle file first
        if os.path.exists(data_path):
            logger.info("Loading from pickle file")
            return pd.read_pickle(data_path)

        # Try to load CSV file
        elif os.path.exists(csv_path):
            logger.info("Loading from CSV file")
            df = pd.read_csv(csv_path)
            # Save as pickle for faster loading next time
            try:
                df.to_pickle(data_path)
                logger.info("Saved CSV data as pickle file")
            except:
                pass
            return df

        else:
            logger.warning("No data files found")
            return None

    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None

def load_models(kmeans_path, dbscan_path, preprocessor_path):
    """
    Load trained ML models
    """
    models = {
        'kmeans': None,
        'dbscan': None,
        'preprocessor': None
    }

    try:
        if os.path.exists(kmeans_path):
            models['kmeans'] = joblib.load(kmeans_path)
            logger.info("Loaded KMeans model")
    except Exception as e:
        logger.error(f"Error loading KMeans model: {e}")

    try:
        if os.path.exists(dbscan_path):
            models['dbscan'] = joblib.load(dbscan_path)
            logger.info("Loaded DBSCAN model")
    except Exception as e:
        logger.error(f"Error loading DBSCAN model: {e}")

    try:
        if os.path.exists(preprocessor_path):
            models['preprocessor'] = joblib.load(preprocessor_path)
            logger.info("Loaded preprocessor")
    except Exception as e:
        logger.error(f"Error loading preprocessor: {e}")

    return models

def get_ml_recommendations(df, product_title, models, num_recommendations):
    """
    Get recommendations using trained ML models
    """
    try:
        # Standardize column names
        df_clean = standardize_dataframe(df)

        if df_clean.empty:
            return get_similarity_recommendations(df, product_title, num_recommendations)

        # Find the product in the dataset
        product_title_clean = product_title.lower().strip()
        df_clean['title_lower'] = df_clean['title'].astype(str).str.lower()

        # Find similar products by title matching
        title_matches = df_clean[df_clean['title_lower'].str.contains(product_title_clean, na=False)]

        if not title_matches.empty:
            # Use the first matching product
            query_product = title_matches.iloc[0]

            # Prepare features for the model
            if models['preprocessor'] is not None:
                # Create feature vector for the query product
                features = prepare_features(query_product, df_clean.columns)

                # Transform features using the preprocessor
                query_features = models['preprocessor'].transform([features])

                # Get cluster prediction
                if models['kmeans'] is not None:
                    cluster = models['kmeans'].predict(query_features)[0]
                    logger.info(f"Predicted cluster: {cluster}")

                    # Get all products in the same cluster
                    df_features = df_clean.apply(lambda row: prepare_features(row, df_clean.columns), axis=1)
                    all_features = models['preprocessor'].transform(df_features.tolist())
                    all_clusters = models['kmeans'].predict(all_features)

                    # Filter products in the same cluster
                    cluster_products = df_clean[all_clusters == cluster].copy()

                    # Calculate similarity within cluster
                    cluster_features = all_features[all_clusters == cluster]
                    similarities = cosine_similarity(query_features, cluster_features).flatten()

                    # Get top recommendations
                    top_indices = similarities.argsort()[-num_recommendations-1:][::-1]

                    recommendations = []
                    for idx in top_indices:
                        if len(recommendations) >= num_recommendations:
                            break

                        if similarities[idx] > 0.1:  # Threshold for similarity
                            product = cluster_products.iloc[idx]
                            if product['title'].lower() != product_title.lower():  # Avoid exact match
                                recommendation = format_recommendation(product, similarities[idx])
                                recommendations.append(recommendation)

                    return recommendations

        # Fallback to similarity-based recommendations
        return get_similarity_recommendations(df, product_title, num_recommendations)

    except Exception as e:
        logger.error(f"Error in ML recommendations: {e}")
        return get_similarity_recommendations(df, product_title, num_recommendations)

def get_similarity_recommendations(df, product_title, num_recommendations):
    """
    Get recommendations using text similarity
    """
    try:
        # Standardize dataframe
        df_clean = standardize_dataframe(df)

        if df_clean.empty:
            return create_sample_recommendations(product_title, num_recommendations)

        # Prepare text data
        df_clean['search_text'] = (
                df_clean['title'].astype(str) + ' ' +
                df_clean['category'].astype(str) + ' ' +
                df_clean.get('subcategory', '').astype(str)
        ).str.lower()

        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=5000,
            ngram_range=(1, 2),
            min_df=1
        )

        # Fit and transform
        tfidf_matrix = vectorizer.fit_transform(df_clean['search_text'])
        query_vector = vectorizer.transform([product_title.lower()])

        # Calculate similarities
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Get top recommendations
        top_indices = similarities.argsort()[-num_recommendations-5:][::-1]

        recommendations = []
        for idx in top_indices:
            if len(recommendations) >= num_recommendations:
                break

            if similarities[idx] > 0.01:  # Lower threshold for more results
                product = df_clean.iloc[idx]
                recommendation = format_recommendation(product, similarities[idx])
                recommendations.append(recommendation)

        return recommendations

    except Exception as e:
        logger.error(f"Error in similarity recommendations: {e}")
        return create_sample_recommendations(product_title, num_recommendations)

def standardize_dataframe(df):
    """
    Standardize dataframe column names and clean data
    """
    try:
        # Column mapping
        column_mapping = {
            'title': ['title', 'product_title', 'name', 'product_name', 'Title'],
            'price': ['price', 'actual_price', 'discounted_price', 'cost', 'Price'],
            'rating': ['rating', 'ratings', 'average_rating', 'stars', 'Rating'],
            'total_ratings': ['total_ratings', 'rating_count', 'review_count', 'reviews', 'Total_Ratings'],
            'category': ['category', 'main_category', 'product_category', 'Category'],
            'subcategory': ['subcategory', 'sub_category', 'SubCategory']
        }

        # Find actual column names
        actual_columns = {}
        for standard_name, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name in df.columns:
                    actual_columns[standard_name] = possible_name
                    break

        # Create standardized dataframe
        df_clean = df.copy()
        for standard_name, actual_name in actual_columns.items():
            if actual_name != standard_name:
                df_clean[standard_name] = df_clean[actual_name]

        # Clean data
        df_clean = df_clean.dropna(subset=['title']).copy()

        # Clean price column
        if 'price' in df_clean.columns:
            df_clean['price'] = df_clean['price'].apply(clean_price)

        # Clean rating column
        if 'rating' in df_clean.columns:
            df_clean['rating'] = pd.to_numeric(df_clean['rating'], errors='coerce').fillna(4.0)

        # Clean total_ratings column
        if 'total_ratings' in df_clean.columns:
            df_clean['total_ratings'] = pd.to_numeric(df_clean['total_ratings'], errors='coerce').fillna(100)

        # Fill missing categories
        if 'category' in df_clean.columns:
            df_clean['category'] = df_clean['category'].fillna('General')

        return df_clean

    except Exception as e:
        logger.error(f"Error standardizing dataframe: {e}")
        return pd.DataFrame()

def prepare_features(product, available_columns):
    """
    Prepare feature vector for ML model
    """
    features = []

    # Add numerical features
    if 'price' in available_columns:
        price = clean_price(product.get('price', 0))
        features.append(float(price) if price != 'N/A' else 0.0)

    if 'rating' in available_columns:
        rating = float(product.get('rating', 4.0))
        features.append(rating)

    if 'total_ratings' in available_columns:
        total_ratings = float(product.get('total_ratings', 100))
        features.append(total_ratings)

    # Add categorical features (you might need to encode these properly)
    if 'category' in available_columns:
        # Simple hash encoding for category
        category_hash = hash(str(product.get('category', 'General'))) % 1000
        features.append(category_hash)

    return features

def clean_price(price):
    """
    Clean and format price
    """
    try:
        if pd.isna(price) or price == '' or price == 0:
            return 'N/A'

        # Convert to string and clean
        price_str = str(price)

        # Remove currency symbols and extra characters
        for symbol in ['$', '€', '£', '₹', '¥', ',']:
            price_str = price_str.replace(symbol, '')

        # Handle price ranges
        if '-' in price_str:
            price_str = price_str.split('-')[0]

        # Extract numeric value
        import re
        numbers = re.findall(r'\d+\.?\d*', price_str)

        if numbers:
            return float(numbers[0])
        else:
            return 'N/A'

    except:
        return 'N/A'

def format_recommendation(product, similarity_score=0.0):
    """
    Format product data for recommendation display
    """
    return {
        'title': str(product.get('title', 'Unknown Product')),
        'price': clean_price(product.get('price', 0)),
        'rating': float(product.get('rating', 4.0)),
        'total_ratings': int(product.get('total_ratings', 100)),
        'category': str(product.get('category', 'General')),
        'subcategory': str(product.get('subcategory', '')),
        'similarity_score': float(similarity_score),
        'product_id': getattr(product, 'name', 0) if hasattr(product, 'name') else 0
    }

def create_sample_recommendations(query, num_recommendations=12):
    """
    Create sample recommendations when no data or models are available
    """
    logger.info(f"Creating sample recommendations for '{query}'")

    # More realistic sample products
    sample_products = [
        {
            'title': f'Premium {query} - Professional Grade',
            'price': 299.99,
            'rating': 4.5,
            'total_ratings': 1250,
            'category': 'Electronics',
            'subcategory': 'Premium',
            'similarity_score': 0.95,
            'product_id': 1
        },
        {
            'title': f'{query} - Latest Model 2024',
            'price': 459.99,
            'rating': 4.7,
            'total_ratings': 890,
            'category': 'Electronics',
            'subcategory': 'Latest',
            'similarity_score': 0.92,
            'product_id': 2
        },
        {
            'title': f'Best {query} for Professionals',
            'price': 199.99,
            'rating': 4.3,
            'total_ratings': 2340,
            'category': 'Professional',
            'subcategory': 'Pro',
            'similarity_score': 0.88,
            'product_id': 3
        },
        {
            'title': f'{query} - Budget Friendly Option',
            'price': 89.99,
            'rating': 4.1,
            'total_ratings': 567,
            'category': 'Budget',
            'subcategory': 'Value',
            'similarity_score': 0.85,
            'product_id': 4
        },
        {
            'title': f'Premium {query} with Extended Warranty',
            'price': 699.99,
            'rating': 4.8,
            'total_ratings': 1890,
            'category': 'Premium',
            'subcategory': 'Warranty',
            'similarity_score': 0.82,
            'product_id': 5
        },
        {
            'title': f'{query} - Customer\'s Choice',
            'price': 399.99,
            'rating': 4.6,
            'total_ratings': 1456,
            'category': 'Popular',
            'subcategory': 'Choice',
            'similarity_score': 0.79,
            'product_id': 6
        },
        {
            'title': f'Top Rated {query} 2024',
            'price': 249.99,
            'rating': 4.9,
            'total_ratings': 3450,
            'category': 'Top Rated',
            'subcategory': 'Bestseller',
            'similarity_score': 0.76,
            'product_id': 7
        },
        {
            'title': f'{query} - Amazon\'s Choice',
            'price': 179.99,
            'rating': 4.4,
            'total_ratings': 2890,
            'category': 'Amazon Choice',
            'subcategory': 'Recommended',
            'similarity_score': 0.73,
            'product_id': 8
        },
        {
            'title': f'Upgraded {query} Pro Max',
            'price': 549.99,
            'rating': 4.7,
            'total_ratings': 1123,
            'category': 'Pro',
            'subcategory': 'Upgraded',
            'similarity_score': 0.70,
            'product_id': 9
        },
        {
            'title': f'{query} - Highly Recommended',
            'price': 129.99,
            'rating': 4.2,
            'total_ratings': 2567,
            'category': 'Recommended',
            'subcategory': 'Popular',
            'similarity_score': 0.67,
            'product_id': 10
        },
        {
            'title': f'Advanced {query} System',
            'price': 799.99,
            'rating': 4.8,
            'total_ratings': 890,
            'category': 'Advanced',
            'subcategory': 'System',
            'similarity_score': 0.64,
            'product_id': 11
        },
        {
            'title': f'{query} - Limited Edition',
            'price': 349.99,
            'rating': 4.5,
            'total_ratings': 1678,
            'category': 'Limited',
            'subcategory': 'Special',
            'similarity_score': 0.61,
            'product_id': 12
        }
    ]

    return sample_products[:num_recommendations]

def product_details(request, product_id):
    """
    View for product details (new function)
    """
    try:
        # This is a placeholder - you would fetch real product details from your data
        product = {
            'id': product_id,
            'title': f'Product {product_id}',
            'price': 299.99,
            'rating': 4.5,
            'total_ratings': 1250,
            'category': 'Electronics',
            'description': 'This is a detailed description of the product.',
            'features': ['Feature 1', 'Feature 2', 'Feature 3'],
            'specifications': {
                'Brand': 'Sample Brand',
                'Model': f'Model-{product_id}',
                'Weight': '1.5 kg',
                'Dimensions': '30x20x10 cm'
            }
        }

        return JsonResponse(product)

    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)