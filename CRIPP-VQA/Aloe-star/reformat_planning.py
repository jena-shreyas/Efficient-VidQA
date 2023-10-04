import pickle
import json
import os

def create_files(data: list, dir_path: str):
    video_qa_files = {}
    
    for qn_dict in data:
        video_id = qn_dict['example_file'].split('_')[-1]
        if video_id not in video_qa_files:
            video_qa_files[video_id] = {}
            video_qa_files[video_id]['questions'] = ([], [])
            video_qa_files[video_id]['answers'] = []

        qn_id = qn_dict['question_id']
        q = qn_dict['question']
        a = qn_dict['answers']

        video_qa_files[video_id]['questions'][0].append(q)
        video_qa_files[video_id]['questions'][1].append(qn_id)
        video_qa_files[video_id]['answers'].append(a)

    for video_id in video_qa_files:
        video_file = 'example_'+ video_id + '.pkl'
        video_dict = video_qa_files[video_id]

        os.makedirs(dir_path, exist_ok=True)

        with open(os.path.join(dir_path, video_file), 'wb') as f:
            pickle.dump(video_dict, f)


with open('data/planning_gt.json', 'r') as f:
    data = json.load(f)

root_path = 'data/planning_gt'

data_paths = {
    'add': 'planning_add_flt_v6', 
    'remove': 'planning_remove_flt_v6',
    'replace': 'planning_replace_flt_v6'
}

for key in data_paths:
    create_files(data[key], os.path.join(root_path, data_paths[key]))