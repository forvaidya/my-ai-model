```markdown
You are an expert Python developer and data generation specialist. Write a complete, well-structured Python script (do NOT execute or show output — only generate the code) that generates synthetic training and testing datasets for arithmetic intent detection (addition/subtraction) based on the following detailed specifications.

---

### CORE REQUIREMENTS

1. **Dataset Size**:
   - Generate **100,000 training samples** and **20,000 test samples** by default.
   - Add a command-line flag `--scale {1..100}` (default: 100). When used, randomly sample `scale%` of the generated data for both train and test JSON files.

2. **Data Structure per Sample**:
   Each JSON line must be a dictionary with:
   ```python
   {
     "prompt": "<natural language sentence>",
     "operands": [int, int, ...],  # list of all numeric values in order of appearance
     "intent": "add" | "subtract",  # inferred from rules below
     "result": <final value relative to first subject>
   }
   ```

3. **Subjects (Names)**:
   Use **real, commonly used Indian names** from Hindu, Muslim, and Christian communities.  
   - **10 Male Hindu**: Amar, Arjun, Rohan, Vikram, Krishna, Rahul, Siddharth, Dev, Aryan, Vivek  
   - **10 Male Muslim**: Akbar, Ahmed, Faisal, Imran, Kareem, Mustafa, Omar, Salman, Yusuf, Zaid  
   - **10 Male Christian**: Anthony, David, George, Jacob, Joseph, Matthew, Paul, Peter, Samuel, Thomas  
   - **10 Female Hindu**: Radha, Sita, Priya, Anjali, Lakshmi, Meera, Kavya, Neha, Pooja, Riya  
   - **10 Female Muslim**: Ayesha, Fatima, Hafsa, Khadija, Maryam, Noor, Sana, Zainab, Amira, Layla  
   - **10 Female Christian**: Anna, Elizabeth, Grace, Hannah, Maria, Rachel, Rebecca, Sarah, Sophia, Teresa  

   **Relationships**:
   - Male → father, friend, son, teacher, brother, boyfriend
   - Female → mother, wife, sister, teacher, friend, girlfriend  
   **Never pair interfaith names** (e.g., Amar + Ayesha not allowed).

4. **Objects**:
   - candies, cake, books, marbles, rupees, dollars (`$`, `₹`)

5. **Verbs**:
   - **Add intent verbs/synonyms**: gain, receive, add, get, obtain, find, collect, earn, secure, acquire
   - **Subtract intent synonyms (invent creative ones)**: lose, give away, donate, spend, transfer, lend, sacrifice, forfeit, surrender, relinquish

6. **Operand Distribution**:
   - **90%**: Exactly 2 operands → use intent detection logic
   - **3%**: 1 operand → intent = `"add"` (e.g., "Mahesh has 5 candies" → result = 5)
   - **7%**: 3+ operands (e.g., "Amar, Akbar, and Anthony have 3, 4, and 5 books each" → sum all)

7. **Intent Logic**:
   - If **exactly 2 operands** AND a **subtract verb** is used → `intent = "subtract"`, compute `a - b`
   - In **all other cases** → `intent = "add"`, compute sum of all operands
   - **Result is always relative to the first subject mentioned**

8. **Sentence Patterns**:
   - Always start with **first subject + initial quantity**
   - Use varied natural phrasing
   - Examples:
     - Add: "Amar had 3 marbles, then Akbar and Anthony gave him 4 and 5 marbles respectively."
     - Subtract: "Radha had 50 rupees but gave 15 rupees to her sister Priya."
     - Multi: "Arjun, Rohan, and Vikram found 7, 8, and 9 candies each."
   - Final answer is relative to first subject, in this case Amar, Radha and Arjun respectively

9. **Output Files**:
   - `train.jsonl` (one JSON per line)
   - `test.jsonl`
   - Use UTF-8 encoding

10. **Intent Detection Module**:
    Include a **separate function** `detect_intent(prompt: str, operands: list) -> str` that:
    - Takes prompt and operands
    - Returns `"add"` or `"subtract"` using the **exact same logic** as data generation
    - Must be robust and rule-based (no ML)

11. **Additional Requirements**:
    - Use `argparse` for `--scale`
    - Seed random with fixed value for reproducibility
    - Use `tqdm` for progress bars
    - Include detailed docstrings and comments
    - Handle currency symbols correctly (`$`, `₹`)
    - Ensure **no interfaith name pairing**
    - Randomly shuffle final data before writing

---

### DELIVERABLE
Output **only the complete Python script** as a code block. Do **not run it**. Do **not explain**. Include:
- Imports
- Name lists
- Verb lists
- Sentence templating logic
- Operand count control
- Intent detection function
- Data generation loop
- JSONL writing
- CLI with `--scale`

Use best practices: type hints, constants, modular functions.
```
```