# End-to-end text QA (src/qa.py)

World: 60 people, 20 cities, 6 countries, 10 companies, 6 relations, ~290 facts, memory D=8192 with ~32 facts/shard. Relation and question-type classifiers were trained on **3 phrasings each** and evaluated on **phrasings never seen in training**. 5 random worlds, 25 questions per question type per world.

## Held-out phrasings (the real test)

| Stage | Accuracy |
|---|---|
| Statement -> relation (reading) | 73.6% |
| Question -> question type (understanding) | 47.6% |
| End-to-end, 1-step questions | 25.8% |
| End-to-end, 2-step questions | 9.9% |
| Memory alone with perfect reading (1-step / 2-step) | 99.5% / 97.3% |
| Memory size | 8,968 bytes for 270 facts |

Example questions (held-out phrasing; answer / correct / detected question type):

```
Which city does Yezufika stay in?                       -> Tizufida (correct: Tizufida) ('lives_in',)
What city is Kotira living in?                          -> Dalodaba (correct: Potiam) ('located_in',)
Which city does Pofira come from?                       -> Koyeka (correct: Baamti) ('lives_in',)
Where is Danemi originally from?                        -> Fitisusu (correct: Daamfi) ('best_friend',)
What is the birthplace of Pogeka?                       -> Yezuko (correct: Daamfi) ('best_friend',)
Where is Zunekopo originally from?                      -> Kazuam (correct: Tizufida) ('best_friend',)
Where is Raneamge originally from?                      -> Fikarapo (correct: Koyeka) ('best_friend',)
What is the employer of Tiamti?                         -> Gemilo (correct: Ramifige) ('best_friend',)
What is the employer of Korane?                         -> Kazuam (correct: Ramifige) ('best_friend',)
What is the employer of Gemiti?                         -> Gefifiba (correct: Datiye) ('best_friend',)
Who is Bakapofi employed by?                            -> Kababara (correct: Sukakane) ('best_friend',)
Which company does Amyeti work for?                     -> Datiye (correct: Datiye) ('works_at',)
What is the employer of Gemisu?                         -> Geamlolo (correct: Ramifige) ('best_friend',)
What did Bakapofi start?                                -> Sukakane (correct: Nebasu) ('founded',)
```

## Training phrasings (sanity check)

| Stage | Accuracy |
|---|---|
| Statement -> relation (reading) | 100.0% |
| Question -> question type (understanding) | 100.0% |
| End-to-end, 1-step questions | 99.8% |
| End-to-end, 2-step questions | 96.5% |
| Memory alone with perfect reading (1-step / 2-step) | 99.8% / 96.5% |
| Memory size | 8,968 bytes for 270 facts |

## How many example phrasings are needed? (fixed test phrasing, 10 worlds)

Each relation / question type starts with 3 training phrasings; `+k` adds k randomly chosen held-out phrasings to training. Every row is tested on the SAME single never-trained phrasing per type (the last held-out one), so rows are comparable; wording variety in the test is small, so treat as indicative.

| Training phrasings per type | Reading (statement -> relation) | Question type | End-to-end 1-step | End-to-end 2-step |
|---|---|---|---|---|
| 3 + 0 | 88.9% | 26.8% | 0.2% | 0.5% |
| 3 + 1 | 69.6% | 49.5% | 27.3% | 18.1% |
| 3 + 2 | 79.3% | 75.6% | 46.1% | 13.9% |
