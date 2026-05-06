import base64
import json
import logging
import os

import requests
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.http import HttpResponseRedirect
from django.http import JsonResponse
from django.shortcuts import redirect
from django.template import loader
from dotenv import load_dotenv

from .file.file_utils import gen_logs_url, gen_dataset_url
from .forms import SiteForm, ProjectJoinForm, ProjectNewForm, ProjectLeaveForm
from .tasks_validator import TaskValidator
from .templatetags.fl_tag import download_actions

# take environment variables from .env.
load_dotenv()
logger = logging.getLogger(__name__)

# read vars from env
site_uid = os.getenv('SITE_UID')
router_url = os.getenv('ROUTER_URL')
router_username = os.getenv('ROUTER_USERNAME')
router_password = os.getenv('ROUTER_PASSWORD')


# Create your views here.
def index(request):
    # if this is a POST request we need to process the form data
    if request.method == "POST":
        # create a form instance and populate it with data from the request:
        site_form = SiteForm(request.POST)
        # check if form is valid:
        if site_form.is_valid():
            site_name = site_form.cleaned_data['name']
            site_description = site_form.cleaned_data['description']
            # get current site info
            response = requests.get('{0}/sites/lookup/?uid={1}'.format(router_url, site_uid),
                                    auth=(router_username, router_password))
            current_site = None
            if response.ok:
                current_site = response.json()

            if current_site:
                # site already exists
                if 'deregister_site' in request.POST:
                    # delete site with DELETE
                    requests.delete('{0}/sites/{1}/'.format(router_url, current_site['id']),
                                    auth=(router_username, router_password))
                else:
                    # update site with PUT
                    current_site['name'] = site_name
                    current_site['description'] = site_description
                    requests.put('{0}/sites/{1}/'.format(router_url, current_site['id']),
                                 headers={'Content-Type': 'application/json'},
                                 auth=(router_username, router_password),
                                 data=json.dumps(current_site))
            else:
                # site does not exist, create site with POST
                current_site = dict()
                current_site['uid'] = site_uid
                current_site['name'] = site_name
                current_site['description'] = site_description
                # register new site
                requests.post('{0}/sites/'.format(router_url),
                              headers={'Content-Type': 'application/json'},
                              auth=(router_username, router_password),
                              data=json.dumps(current_site))
        # redirect to the same page
        return HttpResponseRedirect("./")
    # if a GET, load the form
    else:
        response = requests.get('{0}/sites/lookup/?uid={1}'.format(router_url, site_uid),
                                auth=(router_username, router_password))
        # if current site exists, store it for use
        current_site = None
        if response.ok:
            current_site = response.json()

        project_participants = None

        if current_site:
            # site already exists, init form with existing values
            site_form = SiteForm(initial={
                'name': current_site['name'],
                'description': current_site['description'],
            })
            # get all projects this site is involved
            response_project_participants = requests \
                .get('{0}/projects/lookup/?site_id={1}'.format(router_url, current_site['id']),
                     auth=(router_username, router_password))
            project_participants = response_project_participants.json()
        else:
            # site does not exist, init blank form
            site_form = SiteForm()

        project_leave_form = ProjectLeaveForm()
        # render template
        template = loader.get_template(
            "controller/index.html")
        context = {
            # built-in uid of site
            "site_uid": site_uid,
            # if available, current site info from router
            "site_detail": current_site,
            # site form
            "site_form": site_form,
            # project leave form
            "project_leave_form": project_leave_form,
            # projects the site is involved
            "project_participants": project_participants,
        }
        return HttpResponse(template.render(context, request))


def project_leave(request):
    if request.method == "POST":
        # create a form instance and populate it with data from the request:
        project_leave_form = ProjectLeaveForm(request.POST)
        # check if form is valid:
        print(project_leave_form)
        if project_leave_form.is_valid():
            pp_id = project_leave_form.cleaned_data['participant_id']
            # get current site info
            rr = requests.delete('{0}/project-participants/{1}/'.format(router_url, pp_id),
                                 auth=(router_username, router_password))
            print(rr)
    return redirect('index')


def project_new(request):
    if request.method == "POST":
        # create a form instance and populate it with data from the request:
        project_new_form = ProjectNewForm(request.POST)

        # check if form is valid:
        if project_new_form.is_valid():
            project_name = project_new_form.cleaned_data['name']
            description = project_new_form.cleaned_data['description']
            tasks = project_new_form.cleaned_data['tasks']
            validator = TaskValidator(tasks)
            task_list = validator.get_validated_tasks()
            if task_list is None or len(task_list) == 0:
                return JsonResponse(
                    {'success': False,
                     'msg': 'Tasks provided is not valid due to {}'.format(validator.get_error_msg())})
            # get current site info
            response = requests.get('{0}/sites/lookup/?uid={1}'.format(router_url, site_uid),
                                    auth=(router_username, router_password))
            current_site = None
            if response.ok:
                current_site = response.json()
            else:
                return JsonResponse(
                    {'success': False,
                     'msg': 'Failed to get current site information with site id {}'.format(site_uid)})

            if current_site:
                # create new Project
                project = dict()
                project['name'] = project_name
                project['description'] = description
                project['site'] = current_site['id']
                project['tasks'] = task_list
                requests.post('{0}/projects/'.format(router_url),
                              headers={'Content-Type': 'application/json'},
                              auth=(router_username, router_password),
                              data=json.dumps(project))
                if response.ok:
                    return JsonResponse(
                        {'success': True,
                         'msg': 'Successfully create new project with name {}'.format(project_name)})
                else:
                    return JsonResponse(
                        {'success': False,
                         'msg': 'Failed to create new project due to {}'.format(response.text)})

        # redirect to the home page
        return JsonResponse(
            {'success': False,
             'msg': 'Project info provided is not valid'})
    # if a GET, load the form
    else:
        project_new_form = ProjectNewForm(initial={'tasks': '{}'})
    # render template
    template = loader.get_template(
        "controller/project_new.html")
    context = {
        "project_new_form": project_new_form,
    }
    return HttpResponse(template.render(context, request))


def project_join(request):
    # if this is a POST request we need to process the form data
    if request.method == "POST":
        # create a form instance and populate it with data from the request:
        project_join_form = ProjectJoinForm(request.POST)
        # check if form is valid:
        if project_join_form.is_valid():
            project_name = project_join_form.cleaned_data['name']
            notes = project_join_form.cleaned_data['notes']
            response = requests.get('{0}/projects/lookup/?name={1}'.format(router_url, project_name),
                                    auth=(router_username, router_password))
            project_to_join = None
            if response.ok:
                project_to_join = response.json()
            # retrieve site info
            response = requests.get('{0}/sites/lookup/?uid={1}'.format(router_url, site_uid),
                                    auth=(router_username, router_password))
            current_site = None
            if response.ok:
                current_site = response.json()
            if project_to_join:
                # create new ProjectParticipant to join the project
                project_participant = dict()
                project_participant['site'] = current_site['id']
                project_participant['project'] = project_to_join['id']
                project_participant['role'] = 'PA'
                project_participant['notes'] = notes
                requests.post('{0}/project-participants/'.format(router_url),
                              headers={'Content-Type': 'application/json'},
                              auth=(router_username, router_password),
                              data=json.dumps(project_participant))
        # redirect to the home page
        return HttpResponseRedirect("/controller/")
    # if a GET, load the form
    else:
        # site does not exist, init blank form
        project_join_form = ProjectJoinForm()

    # render template
    template = loader.get_template(
        "controller/project_join.html")
    context = {
        "project_join_form": project_join_form,
    }
    return HttpResponse(template.render(context, request))


def project_detail(request, project_id, site_id):
    project_response = requests.get('{0}/projects/{1}/'.format(router_url, project_id),
                                    auth=(router_username, router_password))
    current_project = None
    all_participants = None
    all_runs = None
    can_start_runs = False
    if project_response.ok:
        current_project = project_response.json()
        if site_id == current_project["site"]:
            participants_response = requests.get(
                '{0}/project-participants/lookup/?project={1}'.format(
                    router_url, project_id),
                auth=(router_username, router_password))
            if participants_response.ok:
                all_participants = participants_response.json()
                can_start_runs = True

    runs_response = requests.get('{0}/runs/lookup/?project={1}&site_uid={2}'.format(router_url, project_id, site_uid),
                                 auth=(router_username, router_password))
    if runs_response.ok:
        all_runs = runs_response.json()
    
    # Check if there are any active runs (not Completed or Failed)
    has_active_runs = False
    active_statuses = ['Preparing', 'Running', 'Pending Aggregating', 'Pending Success', 'Pending Failed', 'Aggregating']
    current_run = all_runs[-1] if all_runs else None
    if current_run:
        status = current_run.get('status')
        cur_seq = current_run.get('cur_seq')
        cur_round = current_run.get('tasks')[0].get('config').get('current_round')
        if status in active_statuses:
            has_active_runs = True
        elif status == 'Standby':
            # Include ALL other Standby states (between tasks/rounds)
            # Exclude only the very first Standby (waiting for dataset upload)
            if not (cur_seq == 1 and cur_round == 1):
                has_active_runs = True                    
    
    # render template
    template = loader.get_template(
        "controller/project_detail.html")
    context = {
        "project_id": project_id,
        "project_details": current_project,
        "participants": all_participants,
        "runs": all_runs,
        "site_id": site_id,
        "can_start_runs": can_start_runs,
        "has_active_runs": has_active_runs
    }
    return HttpResponse(template.render(context, request))


def run_detail(request, batch, project_id, site_id):
    runs_response = requests.get(
        '{0}/runs/detail/?batch={1}&project={2}&site={3}'.format(
            router_url, batch, project_id, site_id),
        auth=(router_username, router_password))
    # if current site exists, store it for use
    dic = {}
    if runs_response.ok:
        dic = runs_response.json()
    # run participants
    # render template
    template = loader.get_template(
        "controller/run_detail.html")
    context = {
        "runs": dic['runs'] if dic else [],
        "participant": dic['participant'] if dic else -1
    }
    return HttpResponse(template.render(context, request))


def upload_dataset(request):
    run_id = request.POST.get('run_id', None)
    has_dataset = request.POST.get('has_dataset', None)
    dataset = request.FILES.get('dataset')
    if not run_id:
        return JsonResponse({
            'success': False, 'msg': 'Run info is not provided to perform action'
        })
    if has_dataset and not dataset:
        return JsonResponse({
            'success': False, 'msg': 'Dataset is not provided to perform action'
        })
    if dataset:
        url = gen_dataset_url(run_id)
        try:
            fs = FileSystemStorage(url)
            # Always replace the canonical dataset file used by task loaders.
            canonical_path = os.path.join(url, 'dataset')
            if os.path.exists(canonical_path):
                os.remove(canonical_path)
            name = fs.save('dataset', dataset)
            if not name:
                return JsonResponse({
                    'success': False, 'msg': 'Dataset saving error'
                })
        except:
            return JsonResponse({
                'success': False, 'msg': 'Dataset saving error'
            })

    param = dict()
    param['status'] = 3
    requests.put('{0}/runs/{1}/status/'.format(router_url, run_id),
                 headers={'Content-type': 'application/json'},
                 auth=(router_username, router_password),
                 data=json.dumps(param))

    return JsonResponse({
        'success': True, 'msg': 'Dataset save successfully'
    })


def start_runs(request, project_id, site_id):
    if project_id and site_id:
        data = dict()
        data['project'] = project_id
        response = requests.post('{0}/runs'.format(router_url),
                                 headers={'Content-Type': 'application/json'},
                                 auth=(router_username, router_password),
                                 data=json.dumps(data))
        if not response.ok:
            return JsonResponse(
                {'success': False, 'msg': 'Failed to start runs of new round due to {}'.format(response.text)})
        else:
            return JsonResponse(
                {'success': True, 'msg': 'Successfully start runs of new round'})
    else:
        return JsonResponse(
            {'success': False, 'msg': 'Project info not provided to start runs of new round'})


def perform_run_action(request, run_id, project_id, batch, role, action):
    if run_id and project_id and batch and role and action:
        file_type = None
        if action in download_actions:
            file_type = action.split()[1]
            logger.debug('will download {} files'.format(file_type))
            response = requests.get(
                '{0}/runs-action/download/?run={1}&all_runs={2}&type={3}'.format(router_url,
                                                                                 run_id,
                                                                                 1,
                                                                                 file_type),
                auth=(router_username, router_password))
        else:
            data = dict()
            data['run'] = run_id
            data['project'] = project_id
            data['batch'] = batch
            data['role'] = role
            data['action'] = action
            response = requests.put('{0}/runs-action/update/'.format(router_url),
                                    headers={
                                        'Content-Type': 'application/json'},
                                    auth=(router_username, router_password),
                                    data=json.dumps(data))
        if not response.ok:
            return JsonResponse(
                {'success': False,
                 'msg': 'Failed to perform {} action on run with id : {} due to {}'.format(action, run_id,
                                                                                           response.text)})
        else:
            if action in download_actions:
                return JsonResponse(
                    {'success': True, 'msg': 'Successfully get run artifacts or logs', 'file_type': file_type,
                     'content': base64.b64encode(response.content).decode('utf-8')})
            return JsonResponse({'success': True, 'msg': 'Successfully update run status'})
    else:
        return JsonResponse(
            {'success': False, 'msg': 'Run info not provided to perform action'})


def fetch_logs(request):
    run_id = request.GET.get('run_id')
    task_seq = request.GET.get('task_seq')
    round_seq = request.GET.get('round_seq')
    line = int(request.GET.get('line'))
    logs_url = gen_logs_url(run_id, task_seq, round_seq)
    try:
        # with open(logs_url, 'r') as log_file:
        with open(logs_url, 'r') as file:
            lines = file.readlines()

            # Calculate the index for the start line
            start_index = max(line, 0)

            # Extract the lines from the start index
            selected_lines = lines[start_index:]

            # Return the selected lines as a list
            return JsonResponse({'success': True, 'content': selected_lines})
    except FileNotFoundError:
        msg = "No logs yet"

    return JsonResponse(
        {'success': False, 'msg': msg})
