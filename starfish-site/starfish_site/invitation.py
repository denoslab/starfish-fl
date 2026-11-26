"""
Invitation files, which carry everything a new site needs to enrol.

The router's admin writes one per site with ``starfish-site invite``. It holds
the router's address as the sites reach it, a single-use enrolment code with
its expiry, the project the site joins, a name for the testbed, a note for the
site and, when the router's certificate comes from the admin's own certificate
authority, that authority's certificate. Starfish Site opens the file in its
setup wizard. The code works once, and the file is as private as a password
until the site has used it.

The file is UTF-8 JSON::

    {"kind": "starfish-invitation", "version": 1,
     "router": "https://10.0.0.5:8443/starfish/api/v1", "code": "...",
     "expires_at": "2026-11-29T10:00:00+00:00", "project": 4,
     "testbed": "...", "note": "...", "ca_cert": "-----BEGIN CERTIFICATE-----..."}
"""

import datetime
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

from starfish_site import config as C

SUFFIX = '.starfish-invitation'
KIND = 'starfish-invitation'
VERSION = 1
# A certificate chain fits many times over, and a wrong file of gigabytes is refused unread
MAX_BYTES = 256 * 1024
PEM_START = '-----BEGIN CERTIFICATE-----'


class InvitationError(ValueError):
    """The file is no invitation, a broken one, or the router refused to make one."""


@dataclass
class Invitation:
    router: str
    code: str
    expires_at: datetime.datetime = None
    project: int = None
    testbed: str = ''
    note: str = ''
    ca_cert: str = None

    def expired(self, now=None):
        if self.expires_at is None:
            return False
        now = now or datetime.datetime.now(datetime.timezone.utc)
        return self.expires_at <= now

    def to_dict(self):
        return {'kind': KIND, 'version': VERSION, 'router': self.router, 'code': self.code,
                'expires_at': self.expires_at.isoformat() if self.expires_at else None,
                'project': self.project, 'testbed': self.testbed, 'note': self.note,
                'ca_cert': self.ca_cert}


def parse_time(text):
    """A timestamp from the router or a file, with UTC for a trailing Z, as an aware datetime."""
    if not text:
        return None
    text = str(text).strip()
    if text.endswith('Z'):
        text = text[:-1] + '+00:00'
    try:
        value = datetime.datetime.fromisoformat(text)
    except ValueError:
        raise InvitationError('The time {} cannot be read'.format(text))
    if value.tzinfo is None:
        value = value.replace(tzinfo=datetime.timezone.utc)
    return value


def describe_time(value):
    """A timestamp in this computer's time zone, to the minute, or "no date"."""
    if value is None:
        return 'no date'
    return value.astimezone().strftime('%Y-%m-%d %H:%M')


def parse(text):
    """An :class:`Invitation` from the text of a file, or :class:`InvitationError` saying what is wrong."""
    try:
        data = json.loads(text)
    except ValueError:
        raise InvitationError('This file is not an invitation from Starfish.')
    if not isinstance(data, dict) or data.get('kind') != KIND:
        raise InvitationError('This file is not an invitation from Starfish.')
    if data.get('version') != VERSION:
        raise InvitationError('This invitation was made by another version of Starfish, which this '
                              'version cannot read. Ask the admin for a new one.')
    router, code = data.get('router'), data.get('code')
    if not isinstance(router, str) or not isinstance(code, str) or not code.strip():
        raise InvitationError('This invitation lacks the address of the main device or its code.')
    try:
        router = C.api_url(router)
    except ValueError as e:
        raise InvitationError(str(e))
    ca_cert = data.get('ca_cert') or None
    if ca_cert is not None and (not isinstance(ca_cert, str) or PEM_START not in ca_cert):
        raise InvitationError('The certificate in this invitation is broken. Ask the admin for a new one.')
    project = data.get('project')
    if project is not None and not isinstance(project, int):
        raise InvitationError('The project in this invitation is not a number.')
    return Invitation(router=router, code=code.strip(), expires_at=parse_time(data.get('expires_at')),
                      project=project, testbed=str(data.get('testbed') or ''),
                      note=str(data.get('note') or ''), ca_cert=ca_cert)


def read(path):
    """The invitation in the file at ``path``."""
    path = Path(path)
    try:
        if path.stat().st_size > MAX_BYTES:
            raise InvitationError('This file is too large for an invitation.')
        text = path.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        raise InvitationError('This file is not an invitation from Starfish.')
    except OSError as e:
        raise InvitationError('The file cannot be read: {}'.format(e.strerror or e))
    return parse(text)


def write(path, invitation):
    """Write ``invitation`` to ``path``, readable by the owner only where the system allows it."""
    path = Path(path)
    if path.suffix != SUFFIX:
        path = path.with_name(path.name + SUFFIX)
    folder = path.parent if str(path.parent) else Path('.')
    fd, tmp = tempfile.mkstemp(dir=folder, prefix='.invitation.')
    try:
        if hasattr(os, 'fchmod'):
            os.fchmod(fd, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(invitation.to_dict(), f, indent=2)
            f.write('\n')
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    return path


def create(router, user, password, project=None, note='', valid_hours=72, testbed='', ca_cert=None):
    """
    Ask the router for a new enrolment code and return the invitation that carries it.

    Needs an admin user of the router. ``ca_cert`` is the file of the
    certificate authority that signed the router's certificate. It also checks
    the router's certificate on this request and travels in the invitation.
    """
    import requests

    try:
        url = C.api_url(router)
    except ValueError as e:
        raise InvitationError(str(e))
    pem = None
    if ca_cert:
        try:
            pem = Path(ca_cert).read_text(encoding='ascii')
        except (OSError, UnicodeDecodeError) as e:
            raise InvitationError('The CA file {} cannot be read: {}'.format(ca_cert, e))
        if PEM_START not in pem:
            raise InvitationError('{} holds no certificate in PEM form'.format(ca_cert))
    body = {'valid_hours': valid_hours, 'note': note or ''}
    if project:
        body['project'] = project
    try:
        response = requests.post(url + '/enrolment-codes/', json=body, auth=(user, password), timeout=30,
                                 verify=str(ca_cert) if ca_cert else True)
    except requests.exceptions.SSLError as e:
        raise InvitationError('The router\'s HTTPS certificate is not trusted: {}\n'
                              'Pass the CA file that signed it with --ca-cert.'.format(e))
    except requests.RequestException as e:
        raise InvitationError('The router at {} cannot be reached: {}'.format(url, e))
    if response.status_code == 401:
        raise InvitationError('The router refused the user {} or its password (401).'.format(user))
    if response.status_code == 403:
        raise InvitationError('The user {} is not an admin of the router (403).'.format(user))
    if response.status_code == 404 and project:
        raise InvitationError('The router has no project {}.'.format(project))
    if response.status_code != 201:
        raise InvitationError('The router answered {}: {}'.format(response.status_code, response.text[:300]))
    answer = response.json()
    return Invitation(router=url, code=answer['code'], expires_at=parse_time(answer.get('expires_at')),
                      project=answer.get('project'), testbed=testbed or '', note=note or '', ca_cert=pem)
