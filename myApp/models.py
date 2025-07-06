from django.db import models
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils import timezone
import uuid

class Category(models.Model):
    """Category model for organizing products"""
    name = models.CharField(max_length=100, unique=True)
    slug = models.SlugField(max_length=100, unique=True)
    description = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name_plural = "Categories"
        ordering = ['name']

    def __str__(self):
        return self.name

class Product(models.Model):
    """Main product model for storing product information"""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=500)
    description = models.TextField(blank=True)
    price = models.DecimalField(max_digits=10, decimal_places=2, validators=[MinValueValidator(0)])
    discounted_price = models.DecimalField(max_digits=10, decimal_places=2, null=True, blank=True)
    rating = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(5)], default=0)
    total_ratings = models.PositiveIntegerField(default=0)
    category = models.ForeignKey(Category, on_delete=models.CASCADE, related_name='products')
    subcategory = models.CharField(max_length=200, blank=True)
    brand = models.CharField(max_length=100, blank=True)
    image_url = models.URLField(blank=True)
    product_url = models.URLField(blank=True)
    availability = models.CharField(max_length=50, default='In Stock')
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Additional fields for ML features
    features = models.JSONField(default=dict, blank=True)  # Store ML features
    cluster_id = models.IntegerField(null=True, blank=True)  # For clustering results

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['category', 'rating']),
            models.Index(fields=['price']),
            models.Index(fields=['is_active', 'availability']),
        ]

    def __str__(self):
        return f"{self.title[:50]}..."

    @property
    def discount_percentage(self):
        """Calculate discount percentage if discounted price exists"""
        if self.discounted_price and self.discounted_price < self.price:
            return round(((self.price - self.discounted_price) / self.price) * 100, 2)
        return 0

    @property
    def final_price(self):
        """Return the final price (discounted if available, otherwise regular price)"""
        return self.discounted_price if self.discounted_price else self.price

class SearchHistory(models.Model):
    """Track user search history for analytics and personalization"""
    search_query = models.CharField(max_length=255)
    search_date = models.DateTimeField(auto_now_add=True)
    results_count = models.PositiveIntegerField(default=0)
    user_ip = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=500, blank=True)

    class Meta:
        ordering = ['-search_date']
        verbose_name_plural = "Search Histories"

    def __str__(self):
        return f"{self.search_query} - {self.search_date.strftime('%Y-%m-%d %H:%M')}"

class Recommendation(models.Model):
    """Store recommendation results for caching and analytics"""
    query_product = models.CharField(max_length=255)
    recommended_products = models.JSONField()  # Store list of recommended product IDs
    algorithm_used = models.CharField(max_length=100, default='similarity')
    similarity_threshold = models.FloatField(default=0.0)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Recommendations for: {self.query_product}"

class UserFeedback(models.Model):
    """Store user feedback on recommendations for model improvement"""
    RATING_CHOICES = [
        (1, 'Poor'),
        (2, 'Fair'),
        (3, 'Good'),
        (4, 'Very Good'),
        (5, 'Excellent')
    ]

    search_query = models.CharField(max_length=255)
    recommended_product = models.ForeignKey(Product, on_delete=models.CASCADE)
    rating = models.IntegerField(choices=RATING_CHOICES)
    feedback_text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    user_ip = models.GenericIPAddressField(null=True, blank=True)

    class Meta:
        ordering = ['-created_at']
        unique_together = ['search_query', 'recommended_product', 'user_ip']

    def __str__(self):
        return f"Feedback: {self.rating}/5 for {self.recommended_product.title[:30]}"