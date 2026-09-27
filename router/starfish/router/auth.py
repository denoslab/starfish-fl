"""Per-site tokens and enrolment codes, SF-09.

A site enrols once with a single-use code and gets a token. It then sends
``Authorization: Token <token>`` with every request. Tokens and codes are
stored only as SHA-256 hashes; a token can be revoked. A request made with
a token acts as that site: it can read and write only its own runs, and a
coordinator's run can also read its batch's runs. Superuser Basic auth
still works, for the admin and for older controllers.
"""

import hashlib
import secrets

from django.utils import timezone
from rest_framework.authentication import BaseAuthentication, get_authorization_header
from rest_framework.exceptions import AuthenticationFailed, PermissionDenied

KEYWORD = b'token'


def hash_secret(value):
    return hashlib.sha256(value.encode()).hexdigest()


def new_secret():
    return secrets.token_urlsafe(32)


class SiteIdentity:
    """``request.user`` for a request made with a site token."""
    is_authenticated = True
    is_anonymous = False
    is_active = True
    is_staff = False
    is_superuser = False

    def __init__(self, site, token):
        self.site = site
        self.token = token
        self.username = 'site:{}'.format(site.uid)
        self.pk = None

    def __str__(self):
        return self.username


class SiteTokenAuthentication(BaseAuthentication):

    def authenticate(self, request):
        from starfish.router.models import SiteToken
        parts = get_authorization_header(request).split()
        if not parts or parts[0].lower() != KEYWORD:
            return None
        if len(parts) != 2:
            raise AuthenticationFailed('Token header must be "Token <token>"')
        try:
            token = parts[1].decode()
        except UnicodeError:
            raise AuthenticationFailed('Token is not valid text')
        record = SiteToken.objects.select_related(
            'site').filter(token_hash=hash_secret(token)).first()
        if record is None or record.revoked_at is not None:
            raise AuthenticationFailed('Invalid or revoked token')
        SiteToken.objects.filter(pk=record.pk).update(
            last_used_at=timezone.now())
        return SiteIdentity(record.site, record), record

    def authenticate_header(self, request):
        return 'Token'


def request_site(request):
    """The site a token request acts for, or None for superuser and Basic auth."""
    return getattr(request.user, 'site', None)


def check_run_access(request, run):
    """A token request may touch only its own site's run."""
    site = request_site(request)
    if site is not None and str(run.site_uid) != str(site.uid):
        raise PermissionDenied('this run belongs to another site')


def check_site_uid(request, uid):
    site = request_site(request)
    if site is not None and str(uid) != str(site.uid):
        raise PermissionDenied('a site can act only as itself')
