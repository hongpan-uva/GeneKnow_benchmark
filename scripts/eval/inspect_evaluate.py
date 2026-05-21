from openai import OpenAI
import os
import pandas as pd
from pathlib import Path
import json
from jsonschema import validate
from token_tracker import TokenTracker
from verify_claims import answer_questions_batch
from generate_questions import generate_questions


def generate_prompt_insert(g_primary_name, g_secondary_names, ct_all_names):
    if len(g_secondary_names) == 0:
        g_insert = g_primary_name
    else:
        g_insert = g_primary_name + " (" + "aliases: " + \
            ", ".join(g_secondary_names) + ")"

    if len(ct_all_names) == 1:
        ct_insert = ct_all_names[0]
    else:
        ct_insert = ct_all_names[0] + \
            " (" + "aliases: " + ", ".join(ct_all_names[1:]) + ")"

    return ({"g_insert": g_insert, "ct_insert": ct_insert})


def extract_claims(passage: str, tracker: TokenTracker, prompt_paramters: dict, model="gpt-5"):
    """
    Extract claims from a paragraph
    """
    gene = prompt_paramters["g_insert"]
    constraint = prompt_paramters["ct_insert"]
    client = prompt_paramters["client"]

    output_schema = {
        "title": "ClaimsObject",
        "type": "array",
        "items": {
            "type": "object",
            "required": ["id", "claim"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "integer"},
                "claim": {"type": "string"}
            }
        }
    }

    response = client.responses.create(
        model=model,
        input=[{"role": "system",
                "content": "You are a claim extractor. Extract atomic, checkable claims from scientific summary."},
               {"role": "developer",
               "content": f"""
A claim is defined as one proposition anchored to a single subject-predicate relationship.

Instructions:
- Extract atomic, checkable claims that exactly reflect the source—no external facts, inference, or speculation.
- Preserve hedging/negation/conditions, entity names/aliases as written, and any numbers, units, directions, or temporal cues.
- Replace all pronouns and demonstratives such as "this population", "these cells" with the full explicit entity name described in the source text.
- Split long or compound sentences into multiple claims.
- Each claim must be self-contained and interpretable in isolation, restate necessary constraints, context, and entity names explicitly
- Use the fewest non-overlapping claims needed to fully cover the text.
- Return the extracted claims with sequential ids in a JSON array, example:
[
        {{
            "id" : 1,
            "claim" : "claim1"
        }},
        {{
            "id" : 2,
            "claim" : "claim2"
        }},
        {{
            "id" : 3,
            "claim" : "claim3"
        }}
]

- Retrun the following blank JSON array, if no claims was extracted or the source text is blank:
[]
"""},
               {"role": "user",
                "content": [
                    {"type": "input_text", "text": f"Extract checkable claims from the following text"
                     },
                    {"type": "input_text", "text": passage}
                ]
                }]
    )

    tracker.update(response.usage)
    structured_output = json.loads(response.output_text)

    validate(instance=structured_output, schema=output_schema)

    return structured_output



def generate_questions_from_passage(passage: str, tracker: TokenTracker, prompt_paramters: dict, model="gpt-5", max_questions=5):
    """
    Generate questions from a passage.
    """
    gene = prompt_paramters["g_insert"]
    constraint = prompt_paramters["ct_insert"]
    client = prompt_paramters["client"]

    output_schema = {
        "title": "QuestionObject",
        "type": "array",
        "items": {
            "type": "object",
            "required": ["id", "question"],
            "additionalProperties": False,
            "properties": {
                "id": {"type": "integer"},
                "question": {"type": "string"}
            }
        }
    }

    response = client.responses.create(
        model=model,
        input=[{"role": "system",
                "content": f"You will receive a text passage. Your job is to generate at most {max_questions} YES/NO questions."},
               {"role": "developer",
               "content": f"""
Definition of “YES/NO question”:
- The question must be answerable strictly with "yes" or "no" without requiring any extra information.
- It should be a single question (no "and/or", no two-part questions).
- Avoid open-ended forms like “explain”, “describe”, “why”, “how”.

Instructions:
- Read in the text passage between <passage> and </passage>
- Ask questions based on the important claims about the functional role of {gene} in the context of {constraint} (e.g. performs specific biological actions, regulates other genes, contributes to a cellular process...)
- Ask at most {max_questions} questions, you can ask less if claims are lacked
- Only ask YES/ONE questions, example:
Does SOX9 specify astrocytes?
- Keep the question concise (ideally ≤ 20 words).
- Return the questions with sequential ids in a JSON array, template:
[
  {{"id": <integer>, "question": <string>}},
  ...
]

- Retrun the following blank JSON array, if there's no question can be asked:
[]
"""},
               {"role": "user",
                "content": [
                    {"type": "input_text", "text": f"Input passage is provided below. Generate at most {max_questions} close-ended questions."
                     },
                    {"type": "input_text", "text": "<passage>" +
                        passage + "</passage>"}
                ]
                }]
    )

    tracker.update(response.usage)
    structured_output = json.loads(response.output_text)

    validate(instance=structured_output, schema=output_schema)

    return structured_output


def evaluate_summary(summary, evidence, prompt_paramters, output_dir, tracker, e_question_file=None):
    if summary == "no function revealed":
        return [{"alignment": 0,
                "coverage": 0, "F1_score": 0}]

    claims = extract_claims(summary, tracker,
                            prompt_paramters, model="gpt-5")

    if e_question_file is None:
        print("No evidence questions were provided, will generate new ones.")
        e_questions = generate_questions_from_passage(evidence, tracker,
                                                      prompt_paramters, model="gpt-5", max_questions=5)
    elif e_question_file.is_file():
        # read in questions is the file exists
        print("Using existed evidence questions.")
        e_questions = pd.read_csv(e_question_file).to_dict(orient='records')
    else:
        print("No evidence questions were provided, will generate new ones.")
        e_questions = generate_questions_from_passage(evidence, tracker,
                                                      prompt_paramters, model="gpt-5", max_questions=5)

    # summary-based answers to evidence-based questions
    se_answers = answer_questions_batch(e_questions, summary,
                                        model="gpt-5", client=prompt_paramters["client"], tracker=tracker)
    # evidence-based answers to evidence-based questions
    ee_answers = answer_questions_batch(e_questions, evidence,
                                        model="gpt-5", client=prompt_paramters["client"], tracker=tracker)

    # Save claims to CSV and generate questions
    claims_file = output_dir / "claims.csv"
    pd.DataFrame(claims).to_csv(claims_file, index=False)
    
    # Generate questions using the shared tracker
    generate_questions(str(claims_file), model="gpt-5", tracker=tracker)
    
    # Read back the questions
    s_questions_df = pd.read_csv(claims_file)
    s_questions = s_questions_df[["id", "question"]].to_dict(orient='records')

    # summary-based answers to summary-based questions
    ss_answers = answer_questions_batch(s_questions, summary,
                                        model="gpt-5", client=prompt_paramters["client"], tracker=tracker)
    # evidence-based answers to summary-based questions
    es_answers = answer_questions_batch(s_questions, evidence,
                                        model="gpt-5", client=prompt_paramters["client"], tracker=tracker)

    # calculate coverage (recall)
    n_valid_qestions = 0
    n_matched_answers = 0
    for item1, item2 in zip(se_answers, ee_answers):
        if item1["id"] != item2["id"]:
            raise Exception("Unmatched answer ids")
        if item2["answer"] != "Idk":
            n_valid_qestions += 1
            if item1["answer"] == item2["answer"]:
                n_matched_answers += 1

    if n_valid_qestions == 0:
        coverage = 0
    else:
        coverage = n_matched_answers/n_valid_qestions

    # calculate alignment (precision)
    n_valid_qestions = 0
    n_matched_answers = 0
    for item1, item2 in zip(es_answers, ss_answers):
        if item1["id"] != item2["id"]:
            raise Exception("Unmatched answer ids")
        if item2["answer"] != "Idk":
            n_valid_qestions += 1
            if item1["answer"] == item2["answer"]:
                n_matched_answers += 1

    if n_valid_qestions == 0:
        alignment = 0
    else:
        alignment = n_matched_answers/n_valid_qestions

    F1_score = 2*alignment*coverage/(alignment + coverage)

    metrics = [{"alignment": alignment,
                "coverage": coverage, "F1_score": F1_score}]

    # save assessment questions and answers
    pd.DataFrame(s_questions).to_csv(
        output_dir / "summary_based_questions.csv", index=False)
    pd.DataFrame(e_questions).to_csv(
        output_dir / "evidence_based_questions.csv", index=False)
    pd.DataFrame(ss_answers).to_csv(
        output_dir / "summary_based_questions_s_answers.csv", index=False)
    pd.DataFrame(es_answers).to_csv(
        output_dir / "summary_based_questions_e_answers.csv", index=False)
    pd.DataFrame(ee_answers).to_csv(
        output_dir / "evidence_based_questions_e_answers.csv", index=False)
    pd.DataFrame(se_answers).to_csv(
        output_dir / "evidence_based_questions_s_answers.csv", index=False)
    pd.DataFrame(metrics).to_csv(
        output_dir / "metrics.csv", index=False)

    return (metrics)


def main():
    # initiate GPT client
    client1 = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    tracker = TokenTracker()

    # test data input
    g_primary_name = "SOX9"
    ct_all_names = ["astrocyte", "astrocytic"]

    # test data output
    data_dir = Path("output/inspect/41271638/")
    report_dict_list = pd.read_csv(
        data_dir / 'GeneKnow_report.csv').to_dict(orient='records')
    passage_df_list = pd.read_csv(
        data_dir / 'evidence_passages/SOX9_PMID41271638_evidence_passages.csv').to_dict(orient='records')

    # set folder to save assessment output
    output_dir = Path(data_dir / "assessment")
    os.makedirs(output_dir, exist_ok=True)

    article_summary = report_dict_list[0]["Summary"]
    evidence = ""
    for item in passage_df_list:
        evidence += item["text"]
        evidence += "\n"

    inserts = generate_prompt_insert(g_primary_name, [], ct_all_names)

    prompt_paramters = {"g_insert": inserts["g_insert"],
                        "ct_insert": inserts["ct_insert"], "client": client1}

    # "/sfs/gpfs/tardis/project/zanglab_project/hz9fq/RAG_project/evaluation/output/inspect/assessment/41271638/evidence_based_questions.csv"
    metrics = evaluate_summary(article_summary, evidence, prompt_paramters, output_dir, tracker,
                               e_question_file="/project/zanglab_project/hz9fq/RAG_project/evaluation/output/inspect/41271638/assessment/evidence_based_questions.csv")

    print(metrics)
    tracker.print_report()


if __name__ == "__main__":
    main()
