import atexit
import json
import logging
import os

import requests
from dotenv import load_dotenv
from starfish.controller.router_auth import router_auth

load_dotenv()
logger = logging.getLogger(__name__)

site_uid = os.getenv('SITE_UID')
router_url = os.getenv('ROUTER_URL')
router_username = os.getenv('ROUTER_USERNAME')
router_password = os.getenv('ROUTER_PASSWORD')


def report_alive():
    send_status(1)


def report_exit():
    send_status(0)


def send_status(status):
    if site_uid:
        data = dict()
        data['uid'] = site_uid
        data['status'] = status
        response = requests.post('{0}/sites/heartbeat/'.format(router_url),
                                 headers={'Content-Type': 'application/json'},
                                 auth=router_auth(),
                                 data=json.dumps(data))
        if not response or not response.ok:
            logger.debug(
                'Failed to report status {}  of site {}'.format(status, site_uid))
    else:
        logger.debug(
            "Site info is absent, will not report status {}".format(status))


atexit.register(report_exit)
