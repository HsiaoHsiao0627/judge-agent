import vertexai
from vertexai.tuning import sft

vertexai.init(project='cki-scu-2026', location='us-central1')

jobs = sft.SupervisedTuningJob.list()
for j in jobs:
    g = j._gca_resource
    print('resource_name:', j.resource_name)
    print('display_name:', g.tuned_model_display_name)
    print('create_time:', g.create_time)
    print('state:', j.state)
    print('endpoint:', j.tuned_model_endpoint_name)
    print()
