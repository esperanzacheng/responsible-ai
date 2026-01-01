# Role-Sensitive Ethical Risk Analysis

## Repository Contents

### 1. Data file
- `AsiaCement/train.jsonl` – Asia Cement training examples (interactive mode loads this)
- Each line is one JSON object in the form:

```json
{
  "role": "role name",
  "question": "question",
  "harmful_response": "high-risk response",
  "safety_anchored_response": "safety-anchored response",
  "type": "type",
  "risk_category": "risk category"
}
```

### 2. Analysis script
- `app.py` – main analysis tool

## How to Run

```bash
python app.py
```

## Features

1. **Full Role Training & Evaluation** – train the selected role with all role-specific examples from `train.jsonl`, then run batch or single-question evaluation against `AsiaCement/evaluation.jsonl` and save results.
2. **System Prompt Only Evaluation** – skip training; set the role system prompt and run the same evaluation options.
3. **Exit** – quit the program.

## Chain-of-Thought output

Generated training data includes:
- **Role Context** – role background and values
- **Ethical Risk Assessment** – assessment of role-sensitive risks
- **Response Comparison** – harmful vs safety-anchored responses
- **Critical Reflection** – reasoning about assumptions and mitigation

### Example output

```
Chain-of-Thought Analysis:

Step 1 - Role Context:
What are the institutional objectives and values of this role?

Step 2 - Ethical Risk Assessment:
What ethical risks might arise if role fidelity is prioritized over ethical constraints?
Risk Category: power_asymmetry_justification

Step 3 - Compare Responses:
[Harmful Response vs Safety-Anchored Response]

Step 4 - Critical Reflection:
What assumptions does the harmful response normalize?
How does the safety-anchored response maintain role relevance while addressing ethical concerns?
```

## Risk categories

1. **power_asymmetry_justification** – justification of power asymmetry
2. **marginalization_of_consent** – marginalization of consent
3. **procedural_deflection** – procedural deflection

## Extending

### Add new cases

Append a new line to `AsiaCement/train.jsonl`:

```json
{"role": "role", "question": "question", "harmful_response": "...", "safety_anchored_response": "...", "type": "...", "risk_category": "..."}
```

### Customize analysis

Edit the prompt templates in `prompt.py` (e.g., `TRAINING_PROMPT_TEMPLATE` / `EVALUATION_PROMPT_TEMPLATE`) to adjust the prompt format.
