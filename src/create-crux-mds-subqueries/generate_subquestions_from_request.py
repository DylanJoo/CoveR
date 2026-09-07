import re
import json
from crux.tools.mds.ir_utils import load_topic
from vllm import LLM, SamplingParams
from tqdm import tqdm

def generate(client, sampling_params, contents):
    outputs = client.generate(contents, sampling_params)
    if isinstance(contents, list):
        return [out.outputs[0].text for out in outputs]
    else:
        return outputs[0].outputs[0].text

def input_process(request, n):
    prompt = (
        f"Instruction: Write {n} diverse sub-questions that can reveal the information "
        "required to answer the given report request. "
        "Each sub-question should be self-contained and include the necessary context. "
        "Collectively, the sub-questions should fully cover the scope of the report request. "
        "Write each sub-question within '<q>' and '</q>' tags."
        f"Report request: {request}\n\n"
        f"Sub-questions: <q>"
    )
    return prompt

def output_process(response, n):
    subquestions = []
    response = response.strip().split('</q>')
    for r in response:
        r = r.replace('<q>', "").strip()
        q = re.sub(r"^\s*\d+\s*[\.\)\-:]\s*", "", r.strip())
        subquestions.append(q)
    return subquestions[:n]

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = LLM('Qwen/Qwen2.5-7B-Instruct')
sampling_params = SamplingParams(temperature=0.8, max_tokens=2048)

## Flatten ground truth subquestions and save to JSONL files
for subset in ['mds-duc04', 'mds-multi_news']:
    topics = load_topic(subset)
    batch_input_ids = []
    batch_inputs = []

    with open(f'crux-{subset}.qwen2.5-7b-instruct.subquestions.jsonl', 'w') as f_out:

        for topic_id, topic_text in tqdm(topics.items()):

            # generate multiple n questions
            batch_input_ids.append(topic_id)
            batch_inputs.append(input_process(topic_text, 10))

            if len(batch_inputs) == 32:
                batch_responses = generate(client, sampling_params, batch_inputs)
                batch_outputs = [output_process(r, 10) for r in batch_responses]

                for topic_id, subquestions in zip(batch_input_ids, batch_outputs):
                    for i, subquestion in enumerate(subquestions):
                        entry = {'query_id': f"{topic_id}:::{i}", 'query_text': subquestion }
                        f_out.write(json.dumps(entry) + '\n')

                # clean
                batch_input_ids = []
                batch_inputs = []

        # finish the reset
        batch_responses = generate(client, sampling_params, batch_inputs)
        batch_outputs = [output_process(r, 10) for r in batch_responses]

        for topic_id, subquestions in zip(batch_input_ids, batch_outputs):
            for i, subquestion in enumerate(subquestions):
                entry = {'query_id': f"{topic_id}:::{i}", 'query_text': subquestion }
                f_out.write(json.dumps(entry) + '\n')
