import json
from crux.tools.mds.ir_utils import load_topic, load_subtopics

## CRUX-MDS
for subset in ['duc04', 'multi_news']:
    subtopics = load_subtopics(subset)
    topics = load_topic(subset)

    # filter the unused subtopics
    subtopics = {k: v for k, v in subtopics.items() if k in topics}

    with open(f'crux-mds-{subset}.oracle.subquestions.jsonl', 'w') as f_out:
        for topic_id, subtopic_list in subtopics.items():
            for i, subtopic in enumerate(subtopic_list):
                entry = {
                    'query_id': f"{topic_id}:::{i}",
                    'query_text': subtopic
                }
                f_out.write(json.dumps(entry) + '\n')
