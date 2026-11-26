"""
What the router says about this site, in words for the people at the site.

The app asks the router the same questions as the site's own process, with
the site's token, and turns the answers into sentences. Nothing here changes
anything on the router.
"""

from dataclasses import dataclass, field

TIMEOUT = 10

# A run's state from the router, as a line for the person at the site
STATE_TEXT = {
    'Standby': 'Waiting for the round to start',
    'Preparing': 'Getting the cases ready',
    'Running': 'Training on this computer',
    'Pending Success': 'Sending what this computer learned',
    'Pending Aggregating': 'Waiting for the other sites',
    'Aggregating': 'Combining what the sites learned',
    'Success': 'Finished',
    'Pending Failed': 'Stopping after an error',
    'Failed': 'Stopped by an error',
    'Sitting Out': 'Sitting out this round',
}
TRAINING = ('Preparing', 'Running', 'Pending Success')
ROLE_TEXT = {'CO': 'Coordinator', 'PA': 'Participant', 'coordinator': 'Coordinator',
             'participant': 'Participant'}


def role_text(role):
    return ROLE_TEXT.get(role, str(role or '').capitalize())


@dataclass
class RunInfo:
    run_id: int
    project_id: int
    state: str
    role: str = ''
    round: int = None
    rounds: int = None

    @property
    def state_text(self):
        return STATE_TEXT.get(self.state, self.state)

    @property
    def round_text(self):
        if self.round and self.rounds:
            return '{} of {}'.format(self.round, self.rounds)
        return ''


@dataclass
class ProjectInfo:
    project_id: int
    name: str
    role: str = ''
    description: str = ''


@dataclass
class RouterStatus:
    reached: bool = False
    error: str = ''
    site_name: str = ''
    site_id: int = None
    runs: list = field(default_factory=list)
    projects: list = field(default_factory=list)

    def project_name(self, project_id):
        for project in self.projects:
            if project.project_id == project_id:
                return project.name
        return 'Project {}'.format(project_id)


def run_info(run):
    """A :class:`RunInfo` from one run of ``runs/active/``."""
    current = total = None
    tasks = run.get('tasks') or []
    seq = run.get('cur_seq') or 1
    if 0 < seq <= len(tasks):
        config = (tasks[seq - 1] or {}).get('config') or {}
        current, total = config.get('current_round'), config.get('total_round')
    project = run.get('project')
    if isinstance(project, dict):
        project = project.get('id')
    return RunInfo(run_id=run.get('id'), project_id=project, state=run.get('status') or '',
                   role=role_text(run.get('role')), round=current, rounds=total)


def fetch(values, get=None):
    """
    Ask the router who this site is, which projects it takes part in and which of its runs are active.

    Returns a :class:`RouterStatus`, with ``reached`` false and ``error`` set
    when the router cannot be reached or refuses the site's token.
    """
    import requests

    from starfish.controller.router_auth import TokenAuth

    get = get or requests.get
    url, uid, token = values.get('ROUTER_URL'), values.get('SITE_UID'), values.get('ROUTER_TOKEN')
    status = RouterStatus()
    if not (url and uid and token):
        status.error = 'This computer is not set up'
        return status
    options = {'auth': TokenAuth(token), 'timeout': TIMEOUT,
               'verify': values.get('REQUESTS_CA_BUNDLE') or True}
    try:
        response = get('{}/sites/lookup/'.format(url), params={'uid': uid}, **options)
        if response.status_code in (401, 403):
            status.error = 'The main device refused this computer\'s key'
            return status
        if response.status_code == 404:
            status.error = 'The main device does not know this computer'
            return status
        response.raise_for_status()
        site = response.json()
        status.reached = True
        status.site_name, status.site_id = site.get('name', ''), site.get('id')
        response = get('{}/projects/lookup/'.format(url), params={'site_id': status.site_id}, **options)
        if response.ok:
            for participant in response.json() or []:
                project = participant.get('project') or {}
                status.projects.append(ProjectInfo(
                    project_id=project.get('id'), name=project.get('name', ''),
                    role=role_text(participant.get('role')),
                    description=project.get('description') or ''))
        response = get('{}/runs/active/'.format(url), **options)
        if response.ok:
            status.runs = [run_info(run) for run in response.json() or []
                           if run.get('site_uid') in (None, uid)]
    except requests.exceptions.SSLError:
        status.reached = False
        status.error = 'The main device\'s certificate is not trusted'
    except requests.RequestException as e:
        status.reached = False
        status.error = 'The main device cannot be reached ({})'.format(e.__class__.__name__)
    except ValueError:
        status.reached = False
        status.error = 'The main device gave an answer this app cannot read'
    return status


def headline(running, stopping, status):
    """
    The big line and the line under it on the Status page.

    ``running`` is whether this site's process runs, ``stopping`` the stop
    request it has not acted on, and ``status`` the router's last answer.
    """
    if not running:
        return ('Starfish is stopped',
                'This computer takes no part in training until you start Starfish again.')
    if stopping:
        return ('Starfish is stopping',
                'Starfish stops when the step in progress ends.')
    runs = status.runs if status else []
    training = [r for r in runs if r.state in TRAINING]
    current = training[0] if training else (runs[0] if runs else None)
    if current is None:
        if status and not status.reached:
            return ('Starfish is running',
                    'It tries to reach the main device again every few seconds.')
        return ('Starfish is running',
                'There is no training at the moment. Starfish waits for the admin to start a round.')
    name = status.project_name(current.project_id)
    if current.round_text:
        return ('{}, round {}'.format(current.state_text, current.round_text),
                'Project {}. You can keep using this computer as usual.'.format(name))
    return (current.state_text, 'Project {}. You can keep using this computer as usual.'.format(name))
