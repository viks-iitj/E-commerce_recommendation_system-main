from django.db import models

class Category(models.Model):
    name = models.CharField(max_length=100)

    def __str__(self):
        return self.name

class SubCategory(models.Model):
    name = models.CharField(max_length=100)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)

    def __str__(self):
        return self.name

class Product(models.Model):
    title = models.CharField(max_length=300)
    category = models.ForeignKey(Category, on_delete=models.CASCADE)
    sub_category = models.ForeignKey(SubCategory, on_delete=models.CASCADE)
    image_url = models.URLField(max_length=500, blank=True, null=True)
    price = models.FloatField()
    rating = models.FloatField()
    reviews = models.IntegerField()
    about = models.TextField(blank=True, null=True)

    def __str__(self):
        return self.title
