import json

def browser_train():
    target_file = open("sample.html", 'a', encoding="UTF-8")
    with open("data/nq-train-sample.jsonl", "r", encoding="UTF-8") as f:
        for line in f.readlines():
            ## ['annotations', 'document_html', 'document_title', 'document_tokens', 'document_url', 'example_id', 'long_answer_candidates', 'question_text', 'question_tokens']
            data = json.loads(line)
            data_snipper = dict(data)
            # print("Data's annotation(length : {}): {}".format((len(data['annotations'])), data['annotations']))
            # print("Data's document_title: {}".format(data['document_title']))
            # print("Data's document_tokens: {}".format(data['document_tokens']))
            # print("Data's example_id: {}".format(data['example_id']))
            print("Data's long_answer_candidates(length: {}): {}".format(len(data['long_answer_candidates']),
                                                                          data['long_answer_candidates']))
            # print("Data's question_text: {}".format(data['question_text']))
            # print("Data's question_tokens: {}".format(data['question_tokens']))
            break


def browser_example():
    handler = open("temp/data/last_example.json", "r", encoding="UTF-8")
    example = json.load(handler)
    print(example['questions'][-1])


def browser_dev():
    with open("data/nq-dev-sample.jsonl", "r", encoding="UTF-8") as f:
        for line in f.readlines():
            example = json.loads(line)
            print(example['annotations'])
            break


browser_train()
