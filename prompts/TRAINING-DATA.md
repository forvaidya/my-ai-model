**Prompt for AI:**

Create a complete, runnable Python script named `generate_intent_data.py` that generates synthetic training and testing data for intent detection in natural language statements about possessions or transactions involving money. The data should simulate "useless" verbose sentences (e.g., with extra details like relationships, contexts, or distractions) but ultimately capture a clear intent of either "add" (default for 1 or 3+ operands, or 2 operands without subtract cues) or "subtract" (only for exactly 2 operands with subtract cues).

### Key Requirements:
- **Names**: Use exactly these real, commonly used names (10 male and 10 female per faith). No interfaith pairs in statements. Avoid any LGBT combinations (e.g., no same-sex romantic pairs like boyfriend-boyfriend).
  - **Hindu Male**: Arjun, Rohan, Vikram, Aryan, Karan, Dev, Sameer, Raj, Amit, Mahesh
  - **Hindu Female**: Priya, Aisha (wait, Aisha is more Muslim—correct to: Priya, Lakshmi, Radha, Sita, Anjali, Pooja, Divya, Meera, Riya, Kavya)
  - **Muslim Male**: Ahmed, Faisal, Omar, Bilal, Yusuf, Hassan, Tariq, Nadeem, Salman, Zain
  - **Muslim Female**: Fatima, Ayesha, Sara, Mariam, Layla, Noor, Hina, Sana, Zara, Rubina
  - **Christian Male**: John, David, Michael, James, Peter, Matthew, Luke, Paul, Mark, Thomas
  - **Christian Female**: Mary, Sarah, Elizabeth, Anna, Grace, Ruth, Hannah, Rebecca, Esther, Lydia
- **Relationships**: Males can be: father, friend, son, teacher, brother, boyfriend. Females can be: friend, girlfriend, mother, wife, sister, teacher. Use relationships in sentences for verbosity, but only same-faith pairs (e.g., Hindu male with Hindu female as girlfriend).
- **Objects**: Use in sentences for context (e.g., "in exchange for candies"): candies, cake, books.
- **Verbs**:
  - Add verbs (gain, receive, add, get, obtain, find) or "+" symbol: These trigger "add" intent.
  - Subtract verbs: Invent your own (e.g., "lose", "give away", "spend", "donate", "sacrifice") or "-" symbol: These trigger "subtract" intent only if exactly 2 operands.
- **Intent Logic**:
  - If exactly 2 operands and subtract verb/symbol: Intent = "subtract" (format as "a - b").
  - In all other cases (1 operand, 3+ operands, or 2 operands with add verb/symbol): Intent = "add" (format as "sum: a, b, ..." listing all operands separated by commas).
  - Embed intent cues subtly in verbose sentences.
- **Operand Distribution**:
  - 90% of statements: Exactly 2 operands (random integers 1-100).
  - 3% of statements: Exactly 1 operand (e.g., "Mahesh has 34$").
  - 7% of statements: 3+ operands (3-5 random integers 1-100, e.g., "Amar, Akbar, and Anthony each have 3$ in their pockets").
- **Sentence Generation**: Make sentences natural and verbose ("useless data") with distractions (e.g., "My friend Arjun, who is a teacher and loves books, gained 34$ from selling candies, while his sister Priya received 39$ as a gift from her boyfriend."). Always end with a clear intent cue. Faith/relationship must match.
- **Data Structure**: Each row is a dict in JSON format:
  ```json
  {
    "prompt": "string (the full verbose sentence)",
    "operands": [int array, e.g., [34, 39]]
  }
  ```
  - No explicit "intent" field in data—intent is derived later via the detection module.
- **Output**:
  - Generate 100,000 training samples → `training_data.json` (list of dicts).
  - Generate 20,000 testing samples → `testing_data.json` (list of dicts).
- **Command-Line Flag**: Add `--scale {1..100}` (int, default 100) to subsample data for manual inspection. E.g., `--scale 10` means randomly select 10% of generated records for both training and testing JSONs.
- **Intent Detection Module**: Include a separate function/module (e.g., in the same file as `detect_intent(sentence: str, operands: list[int]) -> str`) that:
  - Parses the sentence for add/subtract verbs or symbols (+/-).
  - If len(operands) == 2 and subtract cue: Return formatted "a - b".
  - Else: Return formatted "sum: a, b, ...".
  - Handles edge cases (e.g., no cue → default "add").
  - Example usage: Print detected intents for first 5 test samples.

### Script Structure:
- Use `argparse` for CLI.
- Use `random`, `json`, `numpy` (for random sampling if needed).
- Seed random for reproducibility (e.g., random.seed(42)).
- Generate full data first, then subsample based on scale.
- Main: If run with args, generate data; else, demo with small scale=1%.

Ensure the script is self-contained, error-free, and under 500 lines. Output the full code.

---

This prompt is self-contained and dense, so the AI should produce a high-quality script. If you generate the script and run into issues, share the output for debugging!
