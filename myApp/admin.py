from django.contrib import admin
from .models import Category, SubCategory, Product

@admin.register(Category)
class CategoryAdmin(admin.ModelAdmin):
    list_display = ['id', 'name']
    search_fields = ['name']

@admin.register(SubCategory)
class SubCategoryAdmin(admin.ModelAdmin):
    list_display = ['id', 'name', 'category']
    search_fields = ['name']
    list_filter = ['category']

@admin.register(Product)
class ProductAdmin(admin.ModelAdmin):
    list_display = ['id', 'title', 'category', 'sub_category', 'price', 'rating', 'reviews']
    search_fields = ['title']
    list_filter = ['category', 'sub_category']
