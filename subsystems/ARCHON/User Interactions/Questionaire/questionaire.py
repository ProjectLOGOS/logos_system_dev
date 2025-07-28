# ARCHON1/User_Interactions/questionnaire.py
import json
import os

QUESTIONNAIRE_PATH = "questionnaire.json"
OUTPUT_DIR = "ARCHON1/User_Interactions"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "answers.json")


def load_questionnaire(path):
    with open(path, 'r') as f:
        return json.load(f)


def ask_instruction_and_consent(instructions):
    print("\n--- INSTRUCTIONS ---\n")
    print(instructions)
    print("\nDo you agree to continue? (yes/no)")
    consent = input("").strip().lower()
    if consent not in ["yes", "y"]:
        print("Consent not given. Exiting.")
        exit(0)


def strict_ask_and_log(questions):
    responses = []
    for q in questions:
        print(f"\n{q['text']}")
        answer = input("Your answer: ").strip()
        responses.append({"id": q["id"], "question": q["text"], "answer": answer})
    return responses


def ask_definition_rating(concept_block):
    results = []
    for block in concept_block:
        print(f"\nConcept: {block['concept']}")
        for i, definition in enumerate(block['definitions'], 1):
            print(f"  {i}. {definition}")
        print("\nRank the above definitions from 1 (least like your view) to 7 (most like your view). If none fit, provide your own definition below.")
        user_rank = input("Rank (1-7, or 'own'): ").strip()
        own_def = input("Own definition (if any, else leave blank): ").strip()
        example = input("Short example of how you express or relate to this concept: ").strip()
        freq = input("On a scale of 1-7, how often do you express this trait? ").strip()
        results.append({
            "concept": block['concept'],
            "rank": user_rank,
            "own_definition": own_def,
            "example": example,
            "frequency": freq
        })
    return results


def main():
    # Enforce directory exists
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    data = load_questionnaire(QUESTIONNAIRE_PATH)

    # Print and consent
    ask_instruction_and_consent(data['instructions'])

    all_answers = {}

    # Section 1
    sec1 = [s for s in data['sections'] if s['section_title'] == 'Section 1: Worldview Foundations'][0]
    all_answers['Section 1: Worldview Foundations'] = strict_ask_and_log(sec1['questions'])

    # Section 2
    sec2 = [s for s in data['sections'] if s['section_title'] == 'Section 2: Background Questions'][0]
    all_answers['Section 2: Background Questions'] = strict_ask_and_log(sec2['questions'])

    # Definition Ratings
    all_answers['Definition Ratings'] = ask_definition_rating(data['definition_ratings'])

    # Output to JSON
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_answers, f, indent=2)
    print(f"\nAll responses logged to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
