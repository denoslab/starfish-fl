"""
Where a site keeps its settings and files.

Everything lives in one folder, ``~/.starfish-site`` unless ``--home`` or
``STARFISH_SITE_HOME`` names another one::

    site.env        the settings, readable by the owner only
    router-ca.pem   the router's certificate authority, when enrolment got one
    local/          run files, the controller's /starfish-controller/local
    logs/           starfish-site.log, rotated

``site.env`` holds the controller's own environment variables, so a task
reads the same names on a native site as in a container.
"""

import os
import re
import tempfile
from pathlib import Path

from dotenv import dotenv_values

HOME_ENV = 'STARFISH_SITE_HOME'
CONFIG_NAME = 'site.env'
CA_NAME = 'router-ca.pem'
API_PATH = '/starfish/api/v1'

# Settings a site may hold, in the order site.env lists them, with a note each
KNOWN = (
    ('SITE_UID', 'This site, as the router knows it'),
    ('ROUTER_URL', 'The router API'),
    ('ROUTER_TOKEN', 'This site\'s token from enrolment, SF-09. Keep it secret'),
    ('REQUESTS_CA_BUNDLE', 'Certificate authority that signed the router\'s HTTPS certificate'),
    ('BABELBRAIN_FL_STORE', 'This site\'s BabelBrain sample store, read-only'),
    ('BABELBRAIN_FL_EVAL_STORE', 'Coordinator only, the held-out test set for the evaluation gate'),
    ('BABELBRAIN_FL_DEVICE', 'cuda, mps or cpu to override the automatic choice'),
    ('STARFISH_DISABLE_AGENTS', '1 keeps all LLM agent code off on this site'),
)
_SAFE_VALUE = re.compile(r"^[^'\n\r]*$")


def site_home(path=None):
    """The site's folder, from ``path``, then ``STARFISH_SITE_HOME``, then the default."""
    return Path(path or os.environ.get(HOME_ENV) or Path.home() / '.starfish-site').expanduser()


def config_path(home):
    return Path(home) / CONFIG_NAME


def local_folder(home):
    return Path(home) / 'local'


def log_folder(home):
    return Path(home) / 'logs'


def api_url(url):
    """The router API URL, with ``/starfish/api/v1`` added when only a host is given."""
    url = url.strip().rstrip('/')
    scheme, _, rest = url.partition('://')
    if not rest:
        raise ValueError('The router URL needs a scheme, for example https://host:8443')
    if '/' not in rest:
        url += API_PATH
    return url


def read_config(home):
    """The settings in ``site.env``, empty when the file does not exist."""
    path = config_path(home)
    if not path.is_file():
        return {}
    return {k: v for k, v in dotenv_values(path, interpolate=False).items() if v is not None}


def write_config(home, values):
    """
    Write ``site.env`` atomically, readable by the owner only.

    Known settings come first, in a fixed order with a note each, and any
    other setting follows. Empty values are left out.
    """
    home = Path(home)
    home.mkdir(parents=True, exist_ok=True)
    known = [k for k, _ in KNOWN]
    notes = dict(KNOWN)
    lines = ['# starfish-site settings. Written by starfish-site enrol, safe to edit.', '']
    for key in known + sorted(k for k in values if k not in known):
        value = values.get(key)
        if value in (None, ''):
            continue
        value = str(value)
        if not _SAFE_VALUE.match(value):
            raise ValueError('{} cannot hold a quote or a line break'.format(key))
        if key in notes:
            lines.append('# ' + notes[key])
        lines.append("{}='{}'".format(key, value))
    fd, tmp = tempfile.mkstemp(dir=home, prefix='.site.env.')
    try:
        if hasattr(os, 'fchmod'):
            os.fchmod(fd, 0o600)
        # Windows has no os.fchmod in Python 3.11, and the ACL of the user's
        # profile folder keeps the file private there
        with os.fdopen(fd, 'w') as f:
            f.write('\n'.join(lines) + '\n')
        os.replace(tmp, config_path(home))
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return config_path(home)


def apply_config(values, environ=None):
    """Put the settings into the environment, where the controller code reads them."""
    environ = os.environ if environ is None else environ
    for key, value in values.items():
        if value not in (None, ''):
            environ[key] = value
