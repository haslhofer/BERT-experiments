from bert_serving.client import BertClient
import numpy as np
prefix_q = '##### **Q:** '
with open('C:\\Users\\gerhas\\code\\BERT-experiments\\collections.txt') as fp:
    questions_raw = [v.replace(prefix_q, '').strip() for v in fp if v.strip() and v.startswith(prefix_q)]
    print('%d questions loaded, avg. len of %d' % (len(questions_raw), np.mean([len(d.split()) for d in questions_raw])))

questions_cat = [v[:4] for v in questions_raw]
questions = [v[5:] for v in questions_raw]

count = len(questions)
bc = BertClient()
doc_vecs = bc.encode(questions)

while True:
    query = input('Enter title of new article:')
    query_vec = bc.encode([query])[0]
    # compute normalized dot product as score
    score = np.sum(query_vec * doc_vecs, axis=1) / np.linalg.norm(doc_vecs, axis=1)
    topk_idx = np.argsort(score)[::-1][:count]
    for idx in topk_idx:
        print('> %s\t%s\t%s' % (score[idx], questions_cat[idx], questions[idx]))
        
print("end")