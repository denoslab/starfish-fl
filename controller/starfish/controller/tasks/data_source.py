"""Local data sources: tasks that read data already on the site.

A task config may declare ``"data_source": {"type": ...}``. Such a run needs
no dataset upload; each site reads its own data from a location set by its
own environment. Config may never carry a path, so a coordinator cannot make
a participant read arbitrary files.
"""

from starfish.controller.tasks.babel_brain_fno.store import BUCKETS_HZ

BABELBRAIN_STORE = 'babelbrain_store'

# type -> keys allowed in the data_source block
LOCAL_DATA_SOURCES = {
    BABELBRAIN_STORE: {'type', 'bucket_hz'},
}


def get_data_source(config):
    """Return the ``data_source`` block of a task config, or None."""
    if not isinstance(config, dict):
        return None
    return config.get('data_source')


def validate_data_source(data_source):
    """Return an error message, or None when ``data_source`` is valid."""
    if not isinstance(data_source, dict):
        return 'data_source must be a key-value map'
    source_type = data_source.get('type')
    if source_type not in LOCAL_DATA_SOURCES:
        return 'unknown data_source type {!r}'.format(source_type)
    extra = sorted(set(data_source) - LOCAL_DATA_SOURCES[source_type])
    if extra:
        return 'data_source {} does not accept {}; the location comes from the site'.format(
            source_type, ', '.join(extra))
    if source_type == BABELBRAIN_STORE and data_source.get('bucket_hz') not in BUCKETS_HZ:
        return 'data_source bucket_hz must be one of {}'.format(
            ', '.join(str(b) for b in BUCKETS_HZ))
    return None
