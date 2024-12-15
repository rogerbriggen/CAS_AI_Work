# https://huggingface.co/nvidia/NV-Embed-v2
# https://huggingface.co/nvidia/NV-Embed-v2/blob/main/README.md

# very large model (16GB), quite slow, but runs on my machine.
# #1 in the leaderboard for massive text embeddings (https://huggingface.co/spaces/mteb/leaderboard)
# english only

# pip uninstall -y transformer-engine
# pip install torch==2.5.1+cu124   # install torch with gpu support... according to pyorch site and not according to the huggingface site
# pip install transformers==4.42.4
# pip install flash-attn==2.2.0
# pip install sentence-transformers==2.7.0


import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel  # pylint: disable=unused-import


# Each query needs to be accompanied by an corresponding instruction describing the task.
task_name_to_instruct = {"example": "Given a question, retrieve passages that answer the question",}

QUERY_PREFIX = "Instruct: "+task_name_to_instruct["example"]+"\nQuery: "
queries = [
    'are judo throws allowed in wrestling?', 
    'how to become a radiology technician in michigan?'
    ]

# No instruction needed for retrieval passages
PASSAGE_PREFIX = ""
passages = [
    "Since you're reading this, you are probably someone from a judo background or someone who is just wondering how judo techniques can be applied under wrestling rules. So without further ado, let's get to the question. Are Judo throws allowed in wrestling? Yes, judo throws are allowed in freestyle and folkstyle wrestling. You only need to be careful to follow the slam rules when executing judo throws. In wrestling, a slam is lifting and returning an opponent to the mat with unnecessary force.",
    "Below are the basic steps to becoming a radiologic technologist in Michigan:Earn a high school diploma. As with most careers in health care, a high school education is the first step to finding entry-level employment. Taking classes in math and science, such as anatomy, biology, chemistry, physiology, and physics, can help prepare students for their college studies and future careers.Earn an associate degree. Entry-level radiologic positions typically require at least an Associate of Applied Science. Before enrolling in one of these degree programs, students should make sure it has been properly accredited by the Joint Review Committee on Education in Radiologic Technology (JRCERT).Get licensed or certified in the state of Michigan."
]

# load model with tokenizer
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
print(f'Using device: {device}')
model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True).to(device)

start_time = time.time()

# get the embeddings
MAX_LENGTH = 32768
query_embeddings = model.encode(queries, instruction=QUERY_PREFIX, max_length=MAX_LENGTH, device=device)
passage_embeddings = model.encode(passages, instruction=PASSAGE_PREFIX, max_length=MAX_LENGTH, device=device)

# normalize embeddings
query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
passage_embeddings = F.normalize(passage_embeddings, p=2, dim=1)

# get the embeddings with DataLoader (spliting the datasets into multiple mini-batches)
# batch_size=2
# query_embeddings = model._do_encode(queries, batch_size=batch_size, instruction=query_prefix, max_length=max_length, num_workers=32, return_numpy=True)
# passage_embeddings = model._do_encode(passages, batch_size=batch_size, instruction=passage_prefix, max_length=max_length, num_workers=32, return_numpy=True)

print(f'Time taken: {time.time()-start_time:.2f} seconds')

scores = (query_embeddings @ passage_embeddings.T) * 100
print(f'Scores: {scores.tolist()}')
print (f'query_embeddings: {query_embeddings.tolist()}')
