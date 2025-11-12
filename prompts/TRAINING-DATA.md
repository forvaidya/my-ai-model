```markdown
# Refined Prompt for Prompt Generator

You are an expert Python script generator specializing in synthetic math-word-problem datasets. Write a **complete, self-contained, well-documented Python script** that generates **training and testing data** for arithmetic word problems (addition and subtraction only) with realistic linguistic variation, controlled noise, and strict structural consistency.

---

## Core Requirements

Generate **100,000 unique training examples** and **20,000 unique testing examples** by default.  
Support a CLI flag: `--scale [1..100]` (default: `100`) → output only `scale%` of total rows, **randomly sampled without replacement**.  
Save outputs as:  
- `__train.json` (training data)  
- `__test.json` (testing data)  
Each line is a valid JSON object with exactly two fields: `prompt` (str) and `operation` (dict).

---

## Entities

### Actors (30 total, balanced)
- **15 male names**, **15 female names**  
- Equally distributed across **Hindu, Muslim, Christian** faiths (5 per faith per gender)  
- Use **common, well-known names** appropriate to each faith  
- Actors are **school children, teachers, or friends** → use natural roles in relationships

#### Example Names (must balance exactly):
| Faith     | Male (5)           | Female (5)           |
|----------|--------------------|----------------------|
| Hindu    | Mahesh, Rajesh, Arjun, Vivek, Rohan | Priya, Anjali, Kavya, Neha, Shruti |
| Muslim   | Ahmed, Faisal, Imran, Zaid, Yusuf  | Aisha, Fatima, Sara, Zainab, Maryam |
| Christian| John, David, Peter, Paul, James    | Mary, Anna, Elizabeth, Sarah, Grace |

> Names must be **hard-coded** in balanced lists. Random selection must preserve **exact 1:1:1 faith distribution per gender** across full dataset.

---

## Relationships (1:1, directional)

```text
Actor2 is [friend / teacher / student / senior / junior] of Actor1
```
- Anchor **all results to Actor1**  
- Relationship determines **grammatical correctness** in prompt  
- Use appropriate pronouns and prepositions

---

## Assets & Units (normalized)

Supported assets **with symbols**:  
`$`, `₹`, `marbles`, `candies`, `books`, `pencils`

**Normalization Rule (critical)**:  
Treat any string matching regex `.*[UNIT].*` as equivalent if numeric value is same:  
```regex
^[\s]*[\d]+[\s]*[UNIT][\s]*$  OR  ^[\s]*[UNIT][\s]*[\d]+[\s]*$
```
→ `$10` = `10$` = `$ 10` = `10 $` → **all normalize to `10` + unit**

**In script**:  
- Parse and **canonicalize** all asset expressions  
- Store **only numeric value** in `operands`  
- Render **variably in prompt** (with/without spaces, symbol before/after)

---

## Operations & Operand Rules

| Operands | Operation | Verbs (Addition)       | Verbs (Subtraction)     | Result |
|---------|-----------|------------------------|--------------------------|--------|
| 1       | **Add**   | —                      | —                        | Always add a positive number |
| 2       | **Add/Sub**| `get`, `receive`, `add`| `give`, `pay`, `gift`    | Based on verb |
| 3       | **Add**   | `get`, `receive`, `add`| —                        | **Sum all three** |

### Examples:

```json
// 1 operand
{"prompt": "Mahesh has 3 pencils.", "operation": {"kind": "add", "operands": [3]}}

// 2 operands – addition
{"prompt": "John had 45$ and received $10 from Mahesh", "operation": {"kind": "add", "operands": [45, 10]}}

// 2 operands – subtraction
{"prompt": "John had 45$ and gave 10$ to Mahesh", "operation": {"kind": "subtract", "operands": [45, 10]}}

// 3 operands – always sum
{"prompt": "There are 10 champa, 6 rose, 6 jasmine, and 10 marigold flowers", "operation": {"kind": "add", "operands": [10, 6, 6, 10]}}
```

---

## Transaction Direction (Critical)

**All results anchored to Actor1**  
Use correct **prepositions**:

| Action     | Phrase Pattern |
|------------|----------------|
| Addition   | `Actor1 [gets/receives/adds] X [from/to] Actor2` |
| Subtraction| `Actor1 [gives/pays/gifts] X [to] Actor2` |

> Never say “from” in subtraction. Use **“to”**.

---

## Linguistic Noise & Variation (Essential)

Apply **controlled noise** to make prompts natural:

1. **Variable spacing**: `$10`, `10 $`, `$ 10`, `10$`
2. **Pluralization**: `1 pencil` vs `2 pencils`, `1 $` → `$1`
3. **Articles**: `a`, `an`, `the`, or none
4. **Contractions**: `he's`, `she's`, `Mahesh's`
5. **Possessives**: `Mahesh's candies`, `candies of Mahesh`
6. **Sentence structure variation**:
   - `John had 5 marbles. He received 3 from Priya.`
   - `John, who had 5 marbles, received 3 from Priya.`
   - `Having 5 marbles, John received 3 from Priya.`

---

## Output Format (JSON Lines)

```json
{"prompt": "Aisha had 7 candies and gave 2 to Fatima", "operation": {"kind": "subtract", "operands": [7, 2]}}
```

- **One JSON per line**  
- **No extra whitespace, no trailing commas**  
- `operation.kind` → `"add"` or `"subtract"`  
- `operands` → list of **integers only** (1 to 3 elements)

---

## Script Requirements

1. **CLI**:  
   ```bash
   python generate.py --scale 50
   ```
   → 50,000 train + 10,000 test (random subset)

2. **Random Seed**: Reproducible with `--seed N`

3. **Unit Tests** (in-script, run on **0.5% sample**):
   - Parse random 0.5% of generated prompts
   - Recompute result from `operands`
   - Assert **prompt implies correct arithmetic**
   - Print pass/fail summary

4. **Efficiency**:  
   - Generate all 120,000 in < 30 seconds  
   - Use `random`, `json`, `argparse`, `tqdm`  
   - **No external dependencies beyond stdlib + tqdm**

5. **Validation**:
   - No duplicate prompts
   - Exact faith/gender balance
   - All assets normalized
   - Correct verb → operation mapping

---

## Final Deliverable

A **single Python script** `generate_math_data.py` that:
- Runs standalone
- Generates `__train.json` and `__test.json`
- Supports `--scale`, `--seed`
- Includes inline unit tests
- Is fully commented and type-hinted

> Output this script **in full** when executed.
```

--- 

**This refined prompt is crisp, unambiguous, and production-ready for a prompt generator.**
```