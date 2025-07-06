from django.contrib import admin
from django.utils.html import format_html
from django.db.models import Count, Avg
from django.urls import reverse
from django.utils.safestring import mark_safe
from .models import Product, Category, SearchHistory, Recommendation, UserFeedback

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ['name', 'slug', 'product_count', 'created_at']
    list_filter = ['created_at']
    search_fields = ['name', 'description']
    prepopulated_fields = {'slug': ('name',)}
    readonly_fields = ['created_at', 'updated_at']

    def product_count(self, obj):
        return obj.products.count()
    product_count.short_description = 'Products'

    def get_queryset(self, request):
        return super().get_queryset(request).annotate(
            product_count=Count('products')
        )

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = [
        'title_short', 'category', 'price', 'discounted_price',
        'rating', 'total_ratings', 'availability', 'is_active', 'created_at'
    ]
    list_filter = [
        'category', 'availability', 'is_active', 'created_at',
        'rating', 'brand'
    ]
    search_fields = ['title', 'description', 'brand']
    readonly_fields = ['id', 'created_at', 'updated_at', 'discount_percentage_display']
    list_editable = ['price', 'discounted_price', 'is_active', 'availability']
    list_per_page = 50
    date_hierarchy = 'created_at'

    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'description', 'category', 'subcategory', 'brand')
        }),
        ('Pricing', {
            'fields': ('price', 'discounted_price', 'discount_percentage_display')
        }),
        ('Ratings & Reviews', {
            'fields': ('rating', 'total_ratings')
        }),
        ('Links & Media', {
            'fields': ('image_url', 'product_url')
        }),
        ('Status', {
            'fields': ('availability', 'is_active')
        }),
        ('ML Features', {
            'fields': ('features', 'cluster_id'),
            'classes': ('collapse',)
        }),
        ('Metadata', {
            'fields': ('id', 'created_at', 'updated_at'),
            'classes': ('collapse',)
        })
    )

    def title_short(self, obj):
        return obj.title[:50] + "..." if len(obj.title) > 50 else obj.title
    title_short.short_description = 'Title'

    def discount_percentage_display(self, obj):
        discount = obj.discount_percentage
        if discount > 0:
            return format_html(
                '<span style="color: green; font-weight: bold;">{}%</span>',
                discount
            )
        return "No discount"
    discount_percentage_display.short_description = 'Discount %'

    actions = ['activate_products', 'deactivate_products', 'mark_out_of_stock']

    def activate_products(self, request, queryset):
        updated = queryset.update(is_active=True)
        self.message_user(request, f'{updated} products activated.')
    activate_products.short_description = "Activate selected products"

    def deactivate_products(self, request, queryset):
        updated = queryset.update(is_active=False)
        self.message_user(request, f'{updated} products deactivated.')
    deactivate_products.short_description = "Deactivate selected products"

    def mark_out_of_stock(self, request, queryset):
        updated = queryset.update(availability='Out of Stock')
        self.message_user(request, f'{updated} products marked out of stock.')
    mark_out_of_stock.short_description = "Mark as out of stock"

@admin.register(SearchHistory)
class SearchHistoryAdmin(admin.ModelAdmin):
    list_display = ['search_query', 'results_count', 'search_date', 'user_ip']
    list_filter = ['search_date', 'results_count']
    search_fields = ['search_query']
    readonly_fields = ['search_date']
    date_hierarchy = 'search_date'
    list_per_page = 100

    def has_add_permission(self, request):
        return False  # Prevent manual addition

    def has_change_permission(self, request, obj=None):
        return False  # Make read-only

@admin.register(Recommendation)
class RecommendationAdmin(admin.ModelAdmin):
    list_display = ['query_product', 'algorithm_used', 'similarity_threshold', 'created_at']
    list_filter = ['algorithm_used', 'created_at']
    search_fields = ['query_product']
    readonly_fields = ['created_at']
    date_hierarchy = 'created_at'

    def has_add_permission(self, request):
        return False  # Prevent manual addition

@admin.register(UserFeedback)
class UserFeedbackAdmin(admin.ModelAdmin):
    list_display = ['search_query', 'recommended_product_title', 'rating', 'created_at']
    list_filter = ['rating', 'created_at']
    search_fields = ['search_query', 'recommended_product__title']
    readonly_fields = ['created_at']
    date_hierarchy = 'created_at'

    def recommended_product_title(self, obj):
        return obj.recommended_product.title[:50] + "..." if len(obj.recommended_product.title) > 50 else obj.recommended_product.title
    recommended_product_title.short_description = 'Recommended Product'

    def has_add_permission(self, request):
        return False  # Prevent manual addition

# Custom admin site header
admin.site.site_header = "Recommendation System Admin"
admin.site.site_title = "Recommendation System"
admin.site.index_title = "Welcome to Recommendation System Administration"