# Generated by Django 4.0.1 on 2022-01-28 03:02

import api.models
from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Image',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('title', models.CharField(default='', max_length=8, unique=True)),
                ('cover', models.ImageField(blank=True, null=True, upload_to=api.models.upload_image)),
            ],
        ),
    ]