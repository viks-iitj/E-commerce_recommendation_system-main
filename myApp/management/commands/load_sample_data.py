from django.core.management.base import BaseCommand
from django.utils.text import slugify
from myApp.models import Category, Product
from decimal import Decimal

class Command(BaseCommand):
    help = 'Load sample Amazon-like product data'

    def handle(self, *args, **options):
        self.stdout.write('Loading sample data...')

        # Create categories
        categories_data = [
            {'name': 'Electronics', 'description': 'Electronic devices and accessories'},
            {'name': 'Computers', 'description': 'Laptops, desktops, and computer accessories'},
            {'name': 'Mobile Phones', 'description': 'Smartphones and mobile accessories'},
            {'name': 'Audio', 'description': 'Headphones, speakers, and audio equipment'},
            {'name': 'Gaming', 'description': 'Gaming consoles and accessories'},
            {'name': 'Cameras', 'description': 'Digital cameras and photography equipment'},
            {'name': 'Wearables', 'description': 'Smart watches and fitness trackers'},
            {'name': 'Home & Kitchen', 'description': 'Home appliances and kitchen gadgets'},
            {'name': 'Sports & Outdoors', 'description': 'Sports equipment and outdoor gear'},
            {'name': 'Books', 'description': 'Books and e-readers'},
        ]

        categories = {}
        for cat_data in categories_data:
            category, created = Category.objects.get_or_create(
                name=cat_data['name'],
                defaults={
                    'slug': slugify(cat_data['name']),
                    'description': cat_data['description']
                }
            )
            categories[cat_data['name']] = category
            if created:
                self.stdout.write(f'Created category: {category.name}')

        # Sample products data
        products_data = [
            # Electronics
            {
                'title': 'Apple iPhone 15 Pro Max 256GB Natural Titanium',
                'description': 'Latest iPhone with A17 Pro chip, titanium design, and advanced camera system',
                'price': Decimal('1199.99'),
                'discounted_price': Decimal('1099.99'),
                'rating': 4.7,
                'total_ratings': 15420,
                'category': 'Electronics',
                'subcategory': 'Smartphones',
                'brand': 'Apple',
                'image_url': 'https://via.placeholder.com/300x300?text=iPhone+15+Pro+Max',
            },
            {
                'title': 'Samsung Galaxy S24 Ultra 512GB Phantom Black',
                'description': 'Premium Android smartphone with S Pen, 200MP camera, and AI features',
                'price': Decimal('1299.99'),
                'discounted_price': Decimal('1199.99'),
                'rating': 4.5,
                'total_ratings': 12340,
                'category': 'Electronics',
                'subcategory': 'Smartphones',
                'brand': 'Samsung',
                'image_url': 'https://via.placeholder.com/300x300?text=Galaxy+S24+Ultra',
            },
            {
                'title': 'Google Pixel 8 Pro 128GB Obsidian',
                'description': 'AI-powered camera, pure Android experience, and advanced computational photography',
                'price': Decimal('999.99'),
                'discounted_price': Decimal('899.99'),
                'rating': 4.4,
                'total_ratings': 8930,
                'category': 'Electronics',
                'subcategory': 'Smartphones',
                'brand': 'Google',
                'image_url': 'https://via.placeholder.com/300x300?text=Pixel+8+Pro',
            },
            # Computers
            {
                'title': 'MacBook Pro 16-inch M3 Max 36GB RAM 1TB SSD',
                'description': 'Professional laptop with M3 Max chip, stunning Liquid Retina XDR display',
                'price': Decimal('3999.99'),
                'discounted_price': Decimal('3799.99'),
                'rating': 4.8,
                'total_ratings': 5670,
                'category': 'Computers',
                'subcategory': 'Laptops',
                'brand': 'Apple',
                'image_url': 'https://via.placeholder.com/300x300?text=MacBook+Pro+16',
            },
            {
                'title': 'Dell XPS 13 Plus Intel Core i7 32GB RAM 1TB SSD',
                'description': 'Ultra-portable laptop with stunning InfinityEdge display and premium build',
                'price': Decimal('1899.99'),
                'discounted_price': Decimal('1699.99'),
                'rating': 4.6,
                'total_ratings': 7890,
                'category': 'Computers',
                'subcategory': 'Laptops',
                'brand': 'Dell',
                'image_url': 'https://via.placeholder.com/300x300?text=Dell+XPS+13',
            },
            {
                'title': 'ASUS ROG Zephyrus G16 Gaming Laptop RTX 4070',
                'description': 'High-performance gaming laptop with RTX 4070, 165Hz display, and RGB lighting',
                'price': Decimal('2299.99'),
                'discounted_price': Decimal('1999.99'),
                'rating': 4.7,
                'total_ratings': 4560,
                'category': 'Computers',
                'subcategory': 'Gaming Laptops',
                'brand': 'ASUS',
                'image_url': 'https://via.placeholder.com/300x300?text=ASUS+ROG+G16',
            },
            # Audio
            {
                'title': 'Sony WH-1000XM5 Wireless Noise Canceling Headphones',
                'description': 'Industry-leading noise cancellation with premium sound quality and 30-hour battery',
                'price': Decimal('399.99'),
                'discounted_price': Decimal('299.99'),
                'rating': 4.6,
                'total_ratings': 12450,
                'category': 'Audio',
                'subcategory': 'Headphones',
                'brand': 'Sony',
                'image_url': 'https://via.placeholder.com/300x300?text=Sony+WH-1000XM5',
            },
        ]

        for product_data in products_data:
            category = categories.get(product_data['category'])
            if not category:
                self.stdout.write(self.style.WARNING(f"Category not found: {product_data['category']}"))
                continue

            Product.objects.create(
                name=product_data['title'],
                description=product_data['description'],
                price=product_data['price'],
                discounted_price=product_data['discounted_price'],
                rating=product_data['rating'],
                total_ratings=product_data['total_ratings'],
                category=category,
                subcategory=product_data['subcategory'],
                brand=product_data['brand'],
                image_url=product_data['image_url']
            )
            self.stdout.write(self.style.SUCCESS(f"Added product: {product_data['title']}"))
