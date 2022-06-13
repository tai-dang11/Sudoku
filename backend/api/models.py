from django.db import models
from django.contrib.postgres.fields import ArrayField

# Create your models here.
class Post(models.Model):

    title = "solution"
    image = models.ImageField(upload_to='post_images')

    def __str__(self):
        return self.title
