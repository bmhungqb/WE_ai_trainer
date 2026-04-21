from label_studio_sdk import Client
from label_studio_sdk import LabelStudio

def pull_data_from_label_studio(url: str, api_key: str, project_id: int):
    client = Client(url, api_key)
    project = client.get_project(project_id)
    tasks = project.get_tasks()
    for task in tasks:
        
        