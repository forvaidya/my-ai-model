# Prompt for Generating Python Script: Intent Detection Data Generator and Classifier Module

## Overview
You are an expert Python developer. Create a comprehensive Python script that generates synthetic training and testing datasets for intent detection in natural language statements involving people (actors) and objects (money or items). The core task is to classify the **intent** of each statement as either **"add"** (for addition/summation) or **"subtract"** (for subtraction/difference), based on the number of operands (people mentioned) and the context.

### Key Rules for Intent Determination
- **Operands**: These are the people (actors) mentioned in the statement. Represented as an integer array `[id1, id2, ...]` where each `id` is a unique integer assigned to a person (e.g., 0 for first male, 1 for first female, etc.). Use sequential integers starting from 0 for all people combined.
- **Intent Logic**:
  - If **exactly 2 operands** (people) and the context implies subtraction (e.g., using subtract verbs like "lose", "give away", "spend", "remove", "deduct", or symbols like "-"), then intent = **"subtract"**. Format the operands as `a - b` in the prompt string for clarity.
  - In **all other cases** (1 operand, 3+ operands, or 2 operands with add verbs/symbols), intent = **"add"** (summation, e.g., total money or items). Default to addition unless explicitly subtractive with exactly 2 operands.
- **Verbs and Symbols**:
  - **Add verbs/symbols**: gain, receive, add, get, obtain, find, +. Invent neutral/contextual words if needed (e.g., "collect", "earn").
  - **Subtract verbs**: Invent your own words like "lose", "give away", "spend", "remove", "deduct", "-". Use these sparingly, only for 2-operand subtract cases.
- **Objects**: Use money (e.g., "$5", "3 dollars") or items (e.g., "2 candies", "1 cake", "4 books"). Vary them randomly.
- **People (Actors)**:
  - **Religions/Faiths**: Hindu, Muslim, Christian. Use **real, commonly used names** (10 male and 10 female per faith). Examples:
    - **Hindu Male**: Rahul, Amit, Vikram, Sanjay, Deepak, Arjun, Rohan, Karan, Sameer, Pranav.
    - **Hindu Female**: Priya, Anjali, Pooja, Ritu, Neha, Kavita, Shalini, Meera, Divya, Lakshmi.
    - **Muslim Male**: Ahmed, Faisal, Imran, Tariq, Asif, Nadeem, Salman, Rafiq, Zubair, Irfan.
    - **Muslim Female**: Aisha, Fatima, Sara, Mariam, Lubna, Sana, Hina, Rukhsana, Nazia, Bushra.
    - **Christian Male**: John, David, Michael, James, Peter, Thomas, Mark, Luke, Paul, Andrew.
    - **Christian Female**: Mary, Sarah, Elizabeth, Anna, Grace, Ruth, Hannah, Rebecca, Esther, Lydia.
  - **Relationships** (to make statements natural):
    - **Male**: Can be father, friend, son, teacher, brother, boyfriend.
    - **Female**: Can be friend, girlfriend, mother, wife, sister, teacher.
  - **Pairing**: Prefer **same-faith pairs** (90% of cases) to avoid interfaith pairs. Randomly mix faiths only 10% of the time.
- **Operand Distribution** (for variety and default addition):
  - **90% of statements**: Exactly **2 operands** (e.g., "Rahul and Amit have...").
  - **3% of statements**: Exactly **1 operand** (e.g., "Priya has 3$"). Always "add" intent (self-summation or neutral).
  - **7% of statements**: **3+ operands** (e.g., "Amar, Akbar, and Anthony have 3$ each"). Always "add" intent (group total).
- **Useless Data**: Make statements verbose and noisy with **lots of useless/irrelevant details** to simulate real-world messiness (e.g., "In the bustling market on a sunny Tuesday, Rahul, who is a great friend and teacher, suddenly gained 5 candies from his brother Sanjay, while ignoring the rain clouds overhead."). Include fillers like weather, locations (market, school, home), emotions (happy, surprised), or tangents (e.g., "after eating lunch"). This should make ~70-80% of the text irrelevant to the core intent, forcing the model to capture the **final intent** from key verbs, operand count, and structure.

### Output Structure
- Generate **100,000 training rows** and **20,000 testing rows**.
- Each row is a dictionary or CSV row with **exactly these fields**:
  - `prompt`: String – The full noisy natural language statement (e.g., "Amit's girlfriend Priya received 7 books from her sister, but lost 2 to the wind, adding up to...").
  - `operands`: List of integers `[34, 39]` (e.g., IDs of the two people; empty list `[]` if no people, but always at least 1).
  - `intent`: String – Either "add" or "subtract".
- For subtract cases (rare, ~5-10% overall, only in 2-operand): Ensure the prompt subtly implies difference (e.g., "A minus B").
- Save data as:
  - `train_data.csv` (100K rows).
  - `test_data.csv` (20K rows).
- Use `pandas` for generation and saving. Seed randomness with `random.seed(42)` for reproducibility.

### Additional Module: Intent Detector
- Create a separate Python **module** (`intent_detector.py`) with a class `IntentDetector`.
  - It takes a `prompt` string as input.
  - Uses simple rule-based logic (NLP lite with regex/nltk if needed, but keep it basic: count people names from a predefined list, detect verbs/symbols, count operands).
  - Outputs: `{"operands": [int_list], "intent": "add" or "subtract"}`.
  - Train/eval: In the main script, after generating data, demonstrate accuracy on test set (e.g., print overall accuracy; expect high due to rules).
- Edge Cases: Handle names correctly (case-insensitive), ignore possessives (e.g., "Rahul's"), support "+" / "-" symbols inline (e.g., "5 + 3$").

### Script Requirements
- Use Python 3.10+.
- Libraries: `random`, `pandas`, `re` (for basic parsing in detector). No external installs.
- Main script: `data_generator.py` – Run it to generate CSVs and test the detector.
- Make code modular, commented, and efficient (generate in batches if needed for 100K rows).
- Ensure 50/50 male/female mix overall; vary amounts (1-10 for objects).

Write the complete, runnable Python code for `data_generator.py` and `intent_detector.py`. Output the code in a code block for easy copy-paste.
