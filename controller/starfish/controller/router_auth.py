"""How this controller authenticates to the router, SF-09.

With ``ROUTER_TOKEN`` set, from enrolment, every request carries
``Authorization: Token <token>`` and acts as this site only. Without it,
the controller uses ``ROUTER_USERNAME`` and ``ROUTER_PASSWORD`` Basic auth,
as before.
"""

import logging
import os

import requests

logger = logging.getLogger(__name__)
_warned_http = False


class TokenAuth(requests.auth.AuthBase):

    def __init__(self, token):
        self.token = token

    def __call__(self, request):
        request.headers['Authorization'] = 'Token ' + self.token
        return request


def router_auth():
    """The ``auth`` argument for requests to the router."""
    global _warned_http
    token = os.getenv('ROUTER_TOKEN')
    if token:
        if not _warned_http and os.getenv('ROUTER_URL', '').startswith('http://'):
            logger.warning('ROUTER_TOKEN is sent over plain HTTP; use an https ROUTER_URL outside '
                           'a local workbench')
            _warned_http = True
        return TokenAuth(token)
    return (os.getenv('ROUTER_USERNAME'), os.getenv('ROUTER_PASSWORD'))
