from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('starfish', '0006_run_cur_seq_alter_run_site_uid_alter_run_status'),
    ]

    operations = [
        migrations.AddField(
            model_name='project',
            name='agent_config',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name='project',
            name='agent_log',
            field=models.JSONField(blank=True, default=list),
        ),
        migrations.AddField(
            model_name='run',
            name='agent_advice',
            field=models.JSONField(blank=True, default=dict),
        ),
        migrations.AddField(
            model_name='run',
            name='agent_diagnosis',
            field=models.JSONField(blank=True, default=dict),
        ),
    ]
