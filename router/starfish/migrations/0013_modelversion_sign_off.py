from django.db import migrations, models
from django.db.models import F


def approve_existing(apps, schema_editor):
    # Versions published before manual sign-off were already released to labs
    ModelVersion = apps.get_model('starfish', 'ModelVersion')
    ModelVersion.objects.update(status='approved', reviewed_by='before manual sign-off',
                                reviewed_at=F('created_at'))


class Migration(migrations.Migration):

    dependencies = [
        ('starfish', '0012_site_tokens'),
    ]

    operations = [
        migrations.AddField(
            model_name='modelversion',
            name='status',
            field=models.CharField(choices=[('pending', 'Pending'), ('approved', 'Approved'), ('rejected', 'Rejected')], default='pending', max_length=10),
        ),
        migrations.AddField(
            model_name='modelversion',
            name='reviewed_by',
            field=models.CharField(blank=True, default='', max_length=200),
        ),
        migrations.AddField(
            model_name='modelversion',
            name='reviewed_at',
            field=models.DateTimeField(blank=True, null=True),
        ),
        migrations.AddField(
            model_name='modelversion',
            name='review_note',
            field=models.CharField(blank=True, default='', max_length=500),
        ),
        migrations.RunPython(approve_existing, migrations.RunPython.noop),
    ]
