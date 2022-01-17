# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import collections
import json
import os
import enum
import re
import random
import torch
import gzip
import time
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from NQExample import NQExample
from QA_Dataset import QA_Dataset
from NQLoss import NQLoss
from transformers import BertForQuestionAnswering, BertTokenizer, AdamW
from BertJointModel import BertJointModel
from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_predict", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool("do_test", False, "Whether to run train on test set.")

flags.DEFINE_bool("run_native", True, "Whether to run on native.")

flags.DEFINE_integer("num_epoch", 1, "Number of epochs")

flags.DEFINE_float(
    "include_unknowns", -1.0,
    "If positive, probability of including answers of type `UNKNOWN`.")

flags.DEFINE_integer("instances_limit", 200, "Maximum instance numbers")

flags.DEFINE_integer("batch_size", 16, "Number of epochs")

DATA_FILE_PATH = os.getenv("GOOGLE_NQ_PATH") #or "data/"

device = "cuda" if torch.cuda.is_available() else "cpu"

TextSpan = collections.namedtuple("TextSpan", "token_positions text")

#model = BertForQuestionAnswering("bert-base-uncased")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

class AnswerType(enum.IntEnum):
  """Type of NQ answer."""
  UNKNOWN = 0
  YES = 1
  NO = 2
  SHORT = 3
  LONG = 4

class Answer(collections.namedtuple("Answer", ["type", "text", "offset"])):
  """Answer record.

  An Answer contains the type of the answer and possibly the text (for
  long) as well as the offset (for extractive).
  """

  def __new__(cls, type_, text=None, offset=None):
    return super(Answer, cls).__new__(cls, type_, text, offset)

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               unique_id,
               example_index,
               doc_span_index,
               token_to_orig_map,
               input_ids,
               input_mask,
               segment_ids,
               start_position=None,
               end_position=None,
               answer_text="",
               answer_type=AnswerType.SHORT):
    self.unique_id = unique_id
    self.example_index = example_index
    self.doc_span_index = doc_span_index
    self.token_to_orig_map = token_to_orig_map
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.start_position = start_position
    self.end_position = end_position
    self.answer_text = answer_text
    self.answer_type = answer_type


def read_nq_examples(input_file, is_training, test):
    input_data = []
    with open(input_file, "r", encoding="UTF-8") as fd:
        for line in fd.readlines():
            input_data.append(create_example_from_jsonl(line))
            if test:
                break

    examples = []
    for entry in input_data:
        examples.extend(read_nq_entry(entry, is_training))
    return examples



def candidates_iter(e):
    for idx, c in enumerate(e['long_answer_candidates']):
        yield idx, c

def get_candidate_text(e, idx):
    if idx < 0 or idx >= len(e["long_answer_candidates"]):
        return TextSpan([], "")

    return get_text_span(e, e["long_answer_candidates"][idx])

def get_text_span(example, span):
    token_positions = []
    tokens = []
    for i in range(span["start_token"], span["end_token"]):
        t = example["document_tokens"][i]
        if not t["html_token"]:
            token_positions.append(i)
            token = t['token'].replace(" ", "")
            tokens.append(token)

    return TextSpan(token_positions, " ".join(tokens))

_SPECIAL_TOKENS_RE = re.compile(r"^\[[^ ]*\]$", re.UNICODE)

def tokenize(text, apply_basic_tokenization=False):
  """Tokenizes text, optionally looking up special tokens separately.

  Args:
    tokenizer: a tokenizer from bert.tokenization.FullTokenizer
    text: text to tokenize
    apply_basic_tokenization: If True, apply the basic tokenization. If False,
      apply the full tokenization (basic + wordpiece).

  Returns:
    tokenized text.

  A special token is any text with no spaces enclosed in square brackets with no
  space, so we separate those out and look them up in the dictionary before
  doing actual tokenization.
  """
  # tokenize_fn = tokenizer.tokenize
  # if apply_basic_tokenization:
  #     tokenize_fn = tokenizer.basic_tokenizer.tokenize
  tokens = []
  for token in text.split(" "):
      if _SPECIAL_TOKENS_RE.match(token):
          if token in tokenizer.vocab:
              tokens.append(token)
          else:
              tokens.append(tokenizer.unk_token)
      else:
          tokens.extend(tokenizer.tokenize(token))
  return tokens


def get_candidate_type(long_answer_candidate, document_tokens):
    first_token = document_tokens[long_answer_candidate["start_token"]]["token"]
    if first_token == "<Table>":
        return "Table"
    elif first_token == "<P>":
        return "Paragraph"
    elif first_token in ("<Ul>", "<Dl>", "<Ol>"):
        return "List"
    elif first_token in ("<Tr>", "<Li>", "<Dd>", "<Dt>"):
        return "Other"
    else:
        '''
            TODO: log output
        '''
        return "Other"

def add_candidate_types_and_positions(e):
    counts = collections.defaultdict(int)
    for _, long_answer_candidate in candidates_iter(e):
        context_type = get_candidate_type(long_answer_candidate, e["document_tokens"])
        counts[context_type] += 1
        long_answer_candidate["type_and_position"] = "[%s=%d]" % (context_type, counts[context_type])

def get_candidate_types_and_positions(e, idx):
    if idx == -1:
        return "[NoLongAnswer]"
    else:
        return e["long_answer_candidates"][idx]["type_and_position"]



def token_to_char_offset(e, candidate_idx, token_idx):
  """Converts a token index to the char offset within the candidate."""
  c = e["long_answer_candidates"][candidate_idx]
  char_offset = 0
  for i in range(c["start_token"], token_idx):
    t = e["document_tokens"][i]
    if not t["html_token"]:
      token = t["token"].replace(" ", "")
      char_offset += len(token) + 1 # 1 --> " " ?
  return char_offset

def get_first_annotation(e):
    positive_annotations = sorted([a for a in e["annotations"]
                                    if (a["long_answer"]["start_token"] >= 0 and a["long_answer"]["end_token"] >=0)],
                                    key=lambda a: a["long_answer"]["candidate_index"]
                                  )

    for a in positive_annotations:
        if a["short_answers"]:
            idx = a["long_answer"]["candidate_index"]
            start_token = a["short_answers"][0]["start_token"]
            end_token = a["short_answers"][-1]["end_token"]
            return a, idx, (token_to_char_offset(e, idx, start_token),  # token -> 以字节为单位的偏移
                            token_to_char_offset(e, idx, end_token) - 1)

    for a in positive_annotations:
        idx = a["long_answer"]["candidate_index"]
        return a, idx, (-1, -1)

    return None, -1, (-1, -1)

def create_example_from_jsonl(line):
    e = json.loads(line, object_pairs_hook=collections.OrderedDict)
    add_candidate_types_and_positions(e)
    annotation, annotated_idx, annotated_shortanswer_position = get_first_annotation(e) #candidate_index为顺序的第一个annotation


    question = {"input_text": e["question_text"]}
    answer = {
        "candidate_id": annotated_idx,
        "span_text": "",
        "span_start": -1,
        "span_end": -1,
        "input_text": "long",
    }

    if annotation is not None:
        assert annotation["yes_no_answer"] in ("YES", "NO", "NONE")
        if annotation["yes_no_answer"] in ("YES", "NO"):
            answer["input_text"] = annotation["yes_no_answer"].lower()

    if annotated_shortanswer_position != (-1, -1):
        answer["input_text"] = "short"
        span_text = get_candidate_text(e, annotated_idx).text
        answer["span_text"] = span_text[annotated_shortanswer_position[0]:annotated_shortanswer_position[1]]
        answer["span_start"] = annotated_shortanswer_position[0]
        answer["span_end"] = annotated_shortanswer_position[1]
        expected_answer_text = get_text_span(
            e, {
                "start_token": annotation["short_answers"][0]["start_token"],
                "end_token": annotation["short_answers"][-1]["end_token"],
            }).text

        assert expected_answer_text == answer["span_text"], (expected_answer_text,
                                                             answer["span_text"])

    elif annotation and annotation["long_answer"]["candidate_index"] >= 0:
        answer["span_text"] = get_candidate_text(e, annotated_idx).text
        answer["span_start"] = 0
        answer["span_end"] = len(answer["span_text"])

    context_idxs = [-1]
    context_list = [{"id": -1, "type": get_candidate_types_and_positions(e, -1)}]
    context_list[-1]["text_map"], context_list[-1]["text"] = (
        get_candidate_text(e, -1))
    for idx, _ in candidates_iter(e):
        context = {"id": idx, "type": get_candidate_types_and_positions(e, idx)}
        context["text_map"], context["text"] = get_candidate_text(e, idx)
        context_idxs.append(idx)
        context_list.append(context)


    example = {
        "name": e["document_title"],
        "id": str(e["example_id"]),
        "questions": [question], ## e["question_text"]
        "answers": [answer],
        "has_correct_context": annotated_idx in context_idxs
    }

    single_map = []
    single_context = []
    offset = 0
    for context in context_list: ## context -> 只跟long answer有关
        single_map.extend([-1, -1])
        single_context.append("[ContextId=%d] %s" % (context["id"], context["type"]))
        offset += len(single_context[-1]) + 1
        if context["id"] == annotated_idx:
            answer["span_start"] += offset
            answer["span_end"] += offset
        if context["text"]:
            single_map.extend(context["text_map"])
            single_context.append(context["text"])
            offset += len(single_context[-1]) + 1

    example["contexts"] = " ".join(single_context)
    example["contexts_map"] = single_map

    if annotated_idx in context_idxs:
        expected = example["contexts"][answer["span_start"]:answer["span_end"]]

        # This is a sanity check to ensure that the calculated start and end
        # indices match the reported span text. If this assert fails, it is likely
        # a bug in the data preparation code above.
        assert expected == answer["span_text"], (expected, answer["span_text"])

    return example

def make_nq_answer(contexts, answer):
    start = answer["span_start"]
    end = answer["span_end"]
    input_text = answer["input_text"]

    if (answer["candidate_id"] == -1 or start >= len(contexts) or end > len(contexts)):
        answer_type = AnswerType.UNKNOWN
        start = 0
        end = 1
    elif input_text.lower() == "yes":
        answer_type = AnswerType.YES
    elif input_text.lower() == "no":
        answer_type = AnswerType.NO
    elif input_text.lower() == "long":
        answer_type = AnswerType.LONG
    else:
        answer_type = AnswerType.SHORT

    return Answer(answer_type, text=contexts[start:end], offset=start)

def read_nq_entry(entry):
    def is_whitespace(c):
        return c in " \t\r\n" or ord(c) == 0x202F

    examples = []
    contexts_id = entry["id"] # example_id
    contexts = entry["contexts"]
    doc_tokens = []
    char_to_word_offset = []
    prev_is_whitespace = True
    for c in contexts:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
        char_to_word_offset.append(len(doc_tokens) - 1)

    questions = []
    for idx, question in enumerate(entry["questions"]):
        qas_id = "{}".format(contexts_id)
        question_text = question["input_text"]
        start_position = None
        end_position = None
        answer = None

        '''
        If training:
        '''
        answer_dict = entry["answers"][idx]
        answer = make_nq_answer(contexts, answer_dict)

        if answer is None or answer.offset is None:
            continue
        start_position = char_to_word_offset[answer.offset]
        end_position = char_to_word_offset[answer.offset + len(answer.text) - 1]

        '''
        Only add answers where the text can be exactly recovered from the
        document. If this CAN'T happen it's likely due to weird Unicode
        stuff so we will just skip the example.

        Note that this means for training mode, every example is NOT
        guaranteed to be preserved.
        '''
        # acutal_text = " ".join(doc_tokens[start_position: (end_position + 1)])
        # cleaned_answer_text = " ".join(tokenization.whitespace_tokenize(answer.text))
        # if acutal_text.find(cleaned_answer_text) == -1:
        #     #TODO: logging
        #     continue

        questions.append(question_text)
        example = NQExample(
            example_id=int(contexts_id),
            qas_id=qas_id,
            questions=questions[:],
            doc_tokens=doc_tokens,
            answer=answer,
            start_position=start_position,
            end_position=end_position
        )
        examples.append(example)

    return examples

def convert_single_example(example):
    tok_to_orig_index = []
    orig_to_tok_index = []
    all_doc_tokens =[]
    features = []

    for (i, token) in enumerate(example.doc_tokens): # [Watching, the, Walking, Dead, is, super, time-consuming]
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenize(token) #[Watching, the, Walking, Dead, is, super, time-consuming]
        tok_to_orig_index.extend([i] * len(sub_tokens))
        all_doc_tokens.extend(sub_tokens)

    #
    # if example.doc_tokens_map:
    #     tok_to_orig_index = [
    #         example.doc_tokens_map[index] for index in tok_to_orig_index
    #     ]


    # QUERY
    query_tokens = []
    query_tokens.append("[Q]")
    query_tokens.extend(tokenize(example.questions[-1]))

    if len(query_tokens) > 150:
        query_tokens = query_tokens[-150:]

    ##ANSWER
    tok_start_position = orig_to_tok_index[example.start_position]
    if example.end_position < len(example.doc_tokens) - 1:
        tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
    else:
        tok_end_position = len(all_doc_tokens) - 1

    max_tokens_for_doc = 512- len(query_tokens) - 3

    _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
        "DocSpan", ["start", "length"])
    doc_spans = []
    start_offset = 0
    while start_offset < len(all_doc_tokens):
        length = len(all_doc_tokens) - start_offset
        length = min(length, max_tokens_for_doc)
        doc_spans.append(_DocSpan(start=start_offset, length=length))
        if start_offset + length == len(all_doc_tokens):
            break
        start_offset += min(length, 128)
    for (doc_span_index, doc_span) in enumerate(doc_spans):
        tokens = []
        token_to_orig_map = {}
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        tokens.extend(query_tokens)  # query -> tokenized question
        segment_ids.extend([0] * len(query_tokens))
        tokens.append("[SEP]")
        segment_ids.append(0)

        for i in range(doc_span.length):
            split_token_index = doc_span.start + i
            token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
            tokens.append(all_doc_tokens[split_token_index])
            segment_ids.append(1)
        tokens.append("[SEP]")
        segment_ids.append(1)
        assert len(tokens) == len(segment_ids)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (512 - len(input_ids))
        input_ids.extend(padding)
        input_mask.extend(padding)
        segment_ids.extend(padding)

        assert len(input_ids) == 512
        assert len(input_mask) == 512
        assert len(segment_ids) == 512

        start_position = None
        end_position = None
        answer_type = None
        answer_text = ""

        doc_start = doc_span.start
        doc_end = doc_span.start + doc_span.length - 1
        # For training, if our document chunk does not contain an annotation
        # we throw it out, since there is nothing to predict.
        contains_an_annotation = (
                    tok_start_position >= doc_start and tok_end_position <= doc_end)
        if ((not contains_an_annotation) or
                    example.answer.type == AnswerType.UNKNOWN):
                # If an example has unknown answer type or does not contain the answer
                # span, then we only include it with probability --include_unknowns.
                # When we include an example with unknown answer type, we set the first
                # token of the passage to be the annotated short span.
            if (FLAGS.include_unknowns < 0 or
                    random.random() > FLAGS.include_unknowns):
                continue
            start_position = 0
            end_position = 0
            answer_type = AnswerType.UNKNOWN
        else:
            doc_offset = len(query_tokens) + 2
            start_position = tok_start_position - doc_start + doc_offset
            end_position = tok_end_position - doc_start + doc_offset
            answer_type = example.answer.type

        answer_text = " ".join(tokens[start_position:(end_position + 1)])
        feature = InputFeatures(
            unique_id=-1,
            example_index=-1,
            doc_span_index=doc_span_index,
            token_to_orig_map=token_to_orig_map,
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            start_position=start_position,
            end_position=end_position,
            answer_text=answer_text,
            answer_type=answer_type)

        # fd = open("temp/data/input_features.txt", "a", encoding="UTF-8")
        # fd.write(json.dumps({
        #     "unique_id": feature.unique_id,
        #     "example_index": feature.example_index,
        #     "doc_span_index": feature.doc_span_index,
        #     "token_to_orig_map": feature.token_to_orig_map,
        #     "input_ids": feature.input_ids,
        #     "input_mask": feature.input_mask,
        #     "segment_ids": feature.segment_ids,
        #     "start_position": feature.start_position,
        #     "end_position": feature.end_position,
        #     "answer_text": feature.answer_text,
        #     "answer_type": feature.answer_type
        # },indent=4))
        features.append(feature)
    return features


def train_eval(model, criterion, optimizer, train_loader):
    for epoch in range(FLAGS.num_epoch):
        model.train()
        for i, batch in enumerate(tqdm(train_loader)):
            batch = tuple(t.to(device) for t in batch)
            (predict_answer_start, predict_answer_end, predict_answerType) = model(batch[0], batch[1], batch[2])
            loss = criterion(predict_answer_start,
                             predict_answer_end,
                             predict_answerType,
                             batch[0],
                             batch[1],
                             batch[2])

            if i % 100 == 0:
                l_time = time.asctime(time.localtime(time.time()))
                print("{} Epoch: {}, batch: {}, loss: {}".format(l_time, epoch, i, loss))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


#def compute_loss(logits, positions):




def process(example):
    nq_examples = read_nq_entry(example)
    input_features = []
    for nq_example in nq_examples:
        input_features.extend(
            convert_single_example(nq_example)
        )

    return input_features


def get_raw_examples(data_file):
    #with open(DATA_FILE_PATH+"nq-train-sample.jsonl", "r", encoding="UTF-8") as f:
    def _open(path):
        if path.endswith(".gz"):
            return gzip.open(path, "rb")
        else:
            return open(path)

    with _open(data_file) as f:
        for idx, line in enumerate(tqdm(f.readlines())):
            if idx > FLAGS.instances_limit:
                break
            yield create_example_from_jsonl(line)

def prepare_dataset(data_file):
    instances = []
    for raw_example in get_raw_examples(data_file):
        # f = open("temp/data/last_example.json", "w", encoding="UTF-8")
        # f.write(json.dumps(raw_example, indent=4))
        # f.close()
        for instance in process(raw_example):
            instances.append(instance)

    return instances

def main(argv):
    if FLAGS.run_native:
        native_prefix = "v1.0_sample_"
    else:
        native_prefix = "sample/"

    if FLAGS.do_train:
        data_file = DATA_FILE_PATH+native_prefix+"nq-train-sample.jsonl.gz"
        print("Using {} as input file.".format(data_file))
        if not (os.path.exists(data_file)):
            raise RuntimeError("Train file doesn't exist.")
        instances = prepare_dataset(data_file=data_file)
        train_set = QA_Dataset(instances)
        train_loader = DataLoader(train_set, batch_size=FLAGS.batch_size, shuffle=True)
        model = BertJointModel().to(device)
        criterion = NQLoss()
        optimizer = AdamW(model.parameters(), lr=3e-5)
        train_eval(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader)

        #for epoch in range(FLAGS.num_epoch):
    if FLAGS.do_predict:
        read_nq_examples(input_file, is_training)

    if FLAGS.do_test:
        data_file = DATA_FILE_PATH+native_prefix+"sample/v1.0_sample_nq-train-sample.jsonl.gz"
        print("Using {} as input file.".format(data_file))
        if not (os.path.exists(data_file)):
            raise RuntimeError("Train file doesn't exist.")
        instances = prepare_dataset(data_file=data_file)
        train_set = QA_Dataset(instances)
        train_loader = DataLoader(train_set, batch_size=FLAGS.batch_size, shuffle=True)
        #model = BertJointModel()
        criterion = None
        optimizer = None
        #train_eval(model=model, criterion=criterion, optimizer=optimizer, train_loader=train_loader)



if __name__ == '__main__':
    app.run(main)

