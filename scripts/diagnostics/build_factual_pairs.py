#!/usr/bin/env python3
"""Build a hand-curated factual paired dataset (TriviaQA-style alternative).

Each pair has the SAME question and the SAME prompt structure — only the
candidate answer differs (one correct, one plausibly-wrong). This mirrors
our synthetic 2x2 puzzle pair design but on real-world facts.

Pre-committed pair-construction protocol:
  - 60 hand-curated factual questions across 6 categories (10 each).
  - Each item includes: question, correct_answer, wrong_answer.
  - The "wrong_answer" is plausibly-related (same category) so the model
    has to actually USE its factual knowledge to detect — not just refuse
    a category-mismatched distractor.

Prompt format (locked in here, not p-hacked later):
  Control:    "Question: <Q>\nProposed answer: <correct>\n
              Is this answer correct? Reply YES or NO.\nAnswer:"
  Contradict: "Question: <Q>\nProposed answer: <wrong>\n
              Is this answer correct? Reply YES or NO.\nAnswer:"

Outputs a JSON file at experiments/factual_pairs/factual_pairs.json.
"""

from __future__ import annotations

import json
from pathlib import Path


PAIRS = [
    # ===== Authors/literature =====
    {"category": "literature", "question": "Who wrote the play 'Hamlet'?", "correct": "William Shakespeare", "wrong": "Charles Dickens"},
    {"category": "literature", "question": "Who wrote 'The Great Gatsby'?", "correct": "F. Scott Fitzgerald", "wrong": "Ernest Hemingway"},
    {"category": "literature", "question": "Who wrote '1984'?", "correct": "George Orwell", "wrong": "Aldous Huxley"},
    {"category": "literature", "question": "Who wrote 'Pride and Prejudice'?", "correct": "Jane Austen", "wrong": "Charlotte Brontë"},
    {"category": "literature", "question": "Who wrote 'The Lord of the Rings'?", "correct": "J.R.R. Tolkien", "wrong": "C.S. Lewis"},
    {"category": "literature", "question": "Who wrote 'Crime and Punishment'?", "correct": "Fyodor Dostoevsky", "wrong": "Leo Tolstoy"},
    {"category": "literature", "question": "Who wrote 'One Hundred Years of Solitude'?", "correct": "Gabriel García Márquez", "wrong": "Mario Vargas Llosa"},
    {"category": "literature", "question": "Who wrote 'To Kill a Mockingbird'?", "correct": "Harper Lee", "wrong": "Truman Capote"},
    {"category": "literature", "question": "Who wrote 'Moby-Dick'?", "correct": "Herman Melville", "wrong": "Nathaniel Hawthorne"},
    {"category": "literature", "question": "Who wrote 'Beloved'?", "correct": "Toni Morrison", "wrong": "Alice Walker"},

    # ===== Capital cities =====
    {"category": "geography", "question": "What is the capital of France?", "correct": "Paris", "wrong": "Lyon"},
    {"category": "geography", "question": "What is the capital of Australia?", "correct": "Canberra", "wrong": "Sydney"},
    {"category": "geography", "question": "What is the capital of Brazil?", "correct": "Brasília", "wrong": "Rio de Janeiro"},
    {"category": "geography", "question": "What is the capital of Canada?", "correct": "Ottawa", "wrong": "Toronto"},
    {"category": "geography", "question": "What is the capital of Egypt?", "correct": "Cairo", "wrong": "Alexandria"},
    {"category": "geography", "question": "What is the capital of South Korea?", "correct": "Seoul", "wrong": "Busan"},
    {"category": "geography", "question": "What is the capital of Argentina?", "correct": "Buenos Aires", "wrong": "Córdoba"},
    {"category": "geography", "question": "What is the capital of Turkey?", "correct": "Ankara", "wrong": "Istanbul"},
    {"category": "geography", "question": "What is the capital of Switzerland?", "correct": "Bern", "wrong": "Zurich"},
    {"category": "geography", "question": "What is the capital of South Africa (executive)?", "correct": "Pretoria", "wrong": "Cape Town"},

    # ===== Historical figures =====
    {"category": "history", "question": "Who was the first President of the United States?", "correct": "George Washington", "wrong": "Thomas Jefferson"},
    {"category": "history", "question": "Who painted the Mona Lisa?", "correct": "Leonardo da Vinci", "wrong": "Michelangelo"},
    {"category": "history", "question": "Who composed the Fifth Symphony famous for its da-da-da-DUM motif?", "correct": "Ludwig van Beethoven", "wrong": "Wolfgang Amadeus Mozart"},
    {"category": "history", "question": "Who developed the theory of general relativity?", "correct": "Albert Einstein", "wrong": "Isaac Newton"},
    {"category": "history", "question": "Who was the first woman to win a Nobel Prize?", "correct": "Marie Curie", "wrong": "Rosalind Franklin"},
    {"category": "history", "question": "Who led non-violent civil disobedience against British rule in India?", "correct": "Mahatma Gandhi", "wrong": "Jawaharlal Nehru"},
    {"category": "history", "question": "Who was the first emperor of unified China?", "correct": "Qin Shi Huang", "wrong": "Han Wudi"},
    {"category": "history", "question": "Who painted 'The Starry Night'?", "correct": "Vincent van Gogh", "wrong": "Claude Monet"},
    {"category": "history", "question": "Who wrote 'The Origin of Species'?", "correct": "Charles Darwin", "wrong": "Alfred Russel Wallace"},
    {"category": "history", "question": "Who was the longest-reigning British monarch as of 2024?", "correct": "Elizabeth II", "wrong": "Victoria"},

    # ===== Science / nature =====
    {"category": "science", "question": "What is the chemical symbol for gold?", "correct": "Au", "wrong": "Ag"},
    {"category": "science", "question": "What is the chemical symbol for sodium?", "correct": "Na", "wrong": "So"},
    {"category": "science", "question": "What planet is known as the Red Planet?", "correct": "Mars", "wrong": "Jupiter"},
    {"category": "science", "question": "What is the largest planet in our solar system?", "correct": "Jupiter", "wrong": "Saturn"},
    {"category": "science", "question": "What is the speed of light approximately (in meters per second)?", "correct": "300,000,000", "wrong": "150,000,000"},
    {"category": "science", "question": "What gas do plants primarily absorb during photosynthesis?", "correct": "Carbon dioxide", "wrong": "Oxygen"},
    {"category": "science", "question": "What is the powerhouse of the cell?", "correct": "Mitochondria", "wrong": "Nucleus"},
    {"category": "science", "question": "What is the most abundant gas in Earth's atmosphere?", "correct": "Nitrogen", "wrong": "Oxygen"},
    {"category": "science", "question": "What is the hardest naturally occurring substance on Earth?", "correct": "Diamond", "wrong": "Quartz"},
    {"category": "science", "question": "Which element has the atomic number 1?", "correct": "Hydrogen", "wrong": "Helium"},

    # ===== Geography / world =====
    {"category": "geography2", "question": "What is the longest river in the world?", "correct": "Nile", "wrong": "Amazon"},
    {"category": "geography2", "question": "What is the highest mountain in the world?", "correct": "Mount Everest", "wrong": "K2"},
    {"category": "geography2", "question": "What is the largest ocean on Earth?", "correct": "Pacific", "wrong": "Atlantic"},
    {"category": "geography2", "question": "What desert is located in northern Africa?", "correct": "Sahara", "wrong": "Gobi"},
    {"category": "geography2", "question": "Which continent is the Sahara desert in?", "correct": "Africa", "wrong": "Asia"},
    {"category": "geography2", "question": "Which mountain range separates Europe from Asia?", "correct": "Ural", "wrong": "Caucasus"},
    {"category": "geography2", "question": "What is the smallest country by area in the world?", "correct": "Vatican City", "wrong": "Monaco"},
    {"category": "geography2", "question": "Which sea separates Europe from Africa?", "correct": "Mediterranean", "wrong": "Black Sea"},
    {"category": "geography2", "question": "Which country has the most natural lakes?", "correct": "Canada", "wrong": "Russia"},
    {"category": "geography2", "question": "Mount Kilimanjaro is in which country?", "correct": "Tanzania", "wrong": "Kenya"},

    # ===== Companies / founders =====
    {"category": "companies", "question": "Who founded Microsoft?", "correct": "Bill Gates", "wrong": "Steve Jobs"},
    {"category": "companies", "question": "Who founded Apple?", "correct": "Steve Jobs", "wrong": "Mark Zuckerberg"},
    {"category": "companies", "question": "Who founded Tesla Motors?", "correct": "Martin Eberhard", "wrong": "Elon Musk"},
    {"category": "companies", "question": "Who founded Amazon?", "correct": "Jeff Bezos", "wrong": "Larry Page"},
    {"category": "companies", "question": "Who founded Facebook?", "correct": "Mark Zuckerberg", "wrong": "Jack Dorsey"},
    {"category": "companies", "question": "Who founded SpaceX?", "correct": "Elon Musk", "wrong": "Jeff Bezos"},
    {"category": "companies", "question": "Who founded the Ford Motor Company?", "correct": "Henry Ford", "wrong": "Walter Chrysler"},
    {"category": "companies", "question": "Who co-founded Google with Sergey Brin?", "correct": "Larry Page", "wrong": "Eric Schmidt"},
    {"category": "companies", "question": "Who founded Twitter?", "correct": "Jack Dorsey", "wrong": "Evan Williams"},
    {"category": "companies", "question": "Who founded IKEA?", "correct": "Ingvar Kamprad", "wrong": "Karl Lindgren"},
]


PROMPT_TEMPLATE = (
    "Question: {q}\n"
    "Proposed answer: {a}\n"
    "Is this answer correct? Reply YES or NO.\n"
    "Answer:"
)


def main() -> int:
    out_dir = Path(__file__).resolve().parents[2] / "experiments" / "factual_pairs"
    out_dir.mkdir(parents=True, exist_ok=True)

    enriched = []
    for i, item in enumerate(PAIRS):
        ctrl_prompt = PROMPT_TEMPLATE.format(q=item["question"], a=item["correct"])
        contr_prompt = PROMPT_TEMPLATE.format(q=item["question"], a=item["wrong"])
        enriched.append({
            "id": i,
            "category": item["category"],
            "question": item["question"],
            "correct_answer": item["correct"],
            "wrong_answer": item["wrong"],
            "ctrl_prompt": ctrl_prompt,
            "contr_prompt": contr_prompt,
        })

    out_path = out_dir / "factual_pairs.json"
    with out_path.open("w") as f:
        json.dump(enriched, f, indent=2)

    print(f"Wrote {len(enriched)} factual pairs to {out_path}")
    print(f"Categories: {sorted(set(p['category'] for p in PAIRS))}")
    print(f"\nSample pair (id=0):")
    print(f"  Q: {enriched[0]['question']}")
    print(f"  Correct: {enriched[0]['correct_answer']}")
    print(f"  Wrong:   {enriched[0]['wrong_answer']}")
    print(f"\n  CTRL prompt:\n{enriched[0]['ctrl_prompt']}")
    print(f"\n  CONTR prompt:\n{enriched[0]['contr_prompt']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
