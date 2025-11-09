You are an expert Python developer specializing in data generation for natural language processing (NLP) intent classification. Create a complete, self-contained Python script file named `intent_data_generator.py` that performs the following tasks:

#### Core Requirements:
- **Data Generation Logic**:
  - Generate 100,000 training examples and 20,000 testing examples. Each example is a pair: a natural-sounding English sentence (input) and a corresponding intent label (output).
  - Sentences must simulate real-world scenarios involving two actors (e.g., "A gives B a book") and an object (e.g., "a book"). The intent is derived from the interaction:
    - **Addition Intent (+ or 'add')**: Triggered if the verb implies gaining/receiving (use these verbs: gain, receive, add, get, obtain, find). In this case, the label is 'add' or '+'. The sentence structure should imply summation, e.g., "Alice receives 5 candies from Bob" → label: 'add'.
    - **Subtraction Intent (- or 'subtract')**: For all other cases (e.g., verbs implying loss, transfer away, or neutral/default). Invent 8-10 subtraction-like verbs such as: lose, give, take, remove, spend, donate, forfeit, surrender, yield, relinquish. The label is 'subtract' or '-'. Example: "Bob gives 3 books to Alice" → label: 'subtract'.
    - If the sentence explicitly uses symbols like '+' or '-' in a mathematical context (e.g., "a + b" or "x - y"), map directly to 'add' or 'subtract'.
    - Always involve exactly two operands/actors and one object. If exactly two actors are present, treat as subtraction unless an add-verb is used.
    - Intents can be labeled as either the word ('add'/'subtract') or the symbol ('+'/' - '), chosen randomly with 50% probability for variety.
  - **Incorporate Useless Data (Noise for Robustness)**:
    - 70% of sentences should be "useful" (clearly matching the above logic).
    - 30% should be "useless" noise: Random, irrelevant, or garbled sentences that don't fit the pattern (e.g., "The sky is blue and birds fly high", "Random gibberish xyz 123", weather descriptions, shopping lists, or off-topic chit-chat). Label these as 'neutral' or 'none' to train the model to ignore them.
    - Mix in typos, slang, or incomplete sentences in 20% of all data for realism (e.g., "bob giv 2 candys to alice" instead of proper grammar).

- **Names and Roles**:
  - Use real, commonly used names categorized by faith and gender. Provide lists in the code:
    - **Hindu Males (10)**: Aryan, Rohan, Arjun, Vikram, Karan, Dev, Raj, Sameer, Amit, Rahul.
    - **Hindu Females (10)**: Priya, Aishwarya, Lakshmi, Radha, Sita, Anjali, Meera, Pooja, Divya, Kavya.
    - **Muslim Males (10)**: Ahmed, Faisal, Omar, Bilal, Yusuf, Hassan, Tariq, Khalid, Imran, Zain.
    - **Muslim Females (10)**: Ayesha, Fatima, Zainab, Noor, Sana, Hina, Mariam, Layla, Rania, Sara.
    - **Christian Males (10)**: John, David, Michael, James, Daniel, Matthew, Luke, Peter, Andrew, Thomas.
    - **Christian Females (10)**: Mary, Sarah, Elizabeth, Anna, Grace, Hannah, Rebecca, Ruth, Esther, Lydia.
  - Randomly select names from these lists, ensuring diversity (e.g., 33% from each faith).
  - Assign roles to actors for natural phrasing:
    - **Male Roles**: father, friend, son, teacher, brother, boyfriend.
    - **Female Roles**: friend, girlfriend, mother, wife, sister, teacher.
    - Example: "Rohan's father gives Fatima's sister 4 cakes." (Here, roles add context but don't change intent logic.)

- **Objects and Verbs**:
  - **Objects (randomly choose 1 per sentence)**: candies, cake, books. Vary quantity (1-10) or make plural/singular.
  - **Verbs**: As defined above for add/subtract. Randomly pick and conjugate properly (e.g., "gives", "receives").

- **Sentence Structure**:
  - Base template: "[Actor1] [role1] [verb] [quantity] [object(s)] [preposition] [Actor2] [role2]."
  - Add variety: Prepositions like "from/to/with", adjectives (e.g., "sweet candies"), or casual phrases (e.g., "Hey, John got some books from his buddy Mike").
  - For symbol cases: Occasionally embed math-like: "Calculate Aryan + Priya's candies" → 'add'.

- **Output Format**:
  - Save training data to `train_data.csv`: Columns: 'sentence' (str), 'intent' (str).
  - Save testing data to `test_data.csv`: Same format.
  - Use pandas for generation and saving. Ensure balanced classes: ~50% add, ~50% subtract (excluding neutral noise).
  - Print summary stats: Total examples, class distribution.

#### Intent Detection Module:
- Create a separate module/class in the same file called `IntentDetector`.
  - Use a simple rule-based approach first: Keyword matching on verbs/symbols, actor counting (must have exactly 2 for valid intent), and ignore noise if <2 actors or no verb.
  - For advanced: Integrate a lightweight ML model using scikit-learn (e.g., TF-IDF vectorizer + Logistic Regression). Train it on the generated training data within the script.
  - Methods:
    - `detect_intent(sentence: str) -> str`: Returns 'add', 'subtract', '+', '-', or 'none'.
    - `train_model()`: Fits the ML model on training data.
    - `evaluate_on_test()`: Computes accuracy on test data and prints it.
  - Handle edge cases: Ambiguous sentences default to 'subtract'; noise to 'none'.

#### Script Structure:
- Use `random` and `pandas` libraries (import necessary modules).
- Seed random for reproducibility (seed=42).
- Main function: `generate_data()` to create CSVs, then `detector = IntentDetector(); detector.train_model(); detector.evaluate_on_test()`.
- Make the script runnable via `if __name__ == "__main__":`.
- Add comments for clarity. Ensure it's efficient (under 5-10 seconds to run).

Output the full Python code for `intent_data_generator.py` directly in your response, without additional explanations.
