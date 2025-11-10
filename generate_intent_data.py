#!/usr/bin/env python3
"""
Generate synthetic training and testing data for intent detection.
Produces verbose sentences about money transactions with add/subtract intents.
"""

import argparse
import json
import random
from typing import List, Dict, Tuple


# Names by faith (no interfaith pairs)
NAMES = {
    'hindu': {
        'male': ['Arjun', 'Rohan', 'Vikram', 'Aryan', 'Karan', 'Dev', 'Sameer', 'Raj', 'Amit', 'Mahesh'],
        'female': ['Priya', 'Lakshmi', 'Radha', 'Sita', 'Anjali', 'Pooja', 'Divya', 'Meera', 'Riya', 'Kavya']
    },
    'muslim': {
        'male': ['Ahmed', 'Faisal', 'Omar', 'Bilal', 'Yusuf', 'Hassan', 'Tariq', 'Nadeem', 'Salman', 'Zain'],
        'female': ['Fatima', 'Ayesha', 'Sara', 'Mariam', 'Layla', 'Noor', 'Hina', 'Sana', 'Zara', 'Rubina']
    },
    'christian': {
        'male': ['John', 'David', 'Michael', 'James', 'Peter', 'Matthew', 'Luke', 'Paul', 'Mark', 'Thomas'],
        'female': ['Mary', 'Sarah', 'Elizabeth', 'Anna', 'Grace', 'Ruth', 'Hannah', 'Rebecca', 'Esther', 'Lydia']
    }
}

# Relationships (same-faith only, no LGBT)
RELATIONSHIPS = {
    'male': ['father', 'friend', 'son', 'teacher', 'brother', 'boyfriend'],
    'female': ['friend', 'girlfriend', 'mother', 'wife', 'sister', 'teacher']
}

# Objects for context
OBJECTS = ['candies', 'cake', 'books']

# Verbs
ADD_VERBS = ['gained', 'received', 'added', 'got', 'obtained', 'found']
SUBTRACT_VERBS = ['lost', 'gave away', 'spent', 'donated', 'sacrificed']

# Sentence templates
VERBOSE_TEMPLATES = [
    "My {rel1} {name1}, who is a {occupation} and loves {obj}, {verb} {amt1}$ from {context}, while {poss} {rel2} {name2} {verb2} {amt2}$ as a gift.",
    "{name1}, a {occupation} from the neighborhood, {verb} {amt1}$ by selling {obj}, and {poss} {rel2} {name2} {verb2} {amt2}$ unexpectedly.",
    "During the festival, {name1} {verb} {amt1}$ while {poss} {rel2} {name2}, who teaches at the local school, {verb2} {amt2}$.",
    "Last week, {name1} {verb} {amt1}$ in exchange for {obj}, and {poss} friend {name2} {verb2} {amt2}$ from a lottery.",
    "{name1} has been saving for months and finally {verb} {amt1}$, meanwhile {poss} {rel2} {name2} {verb2} {amt2}$ on {obj}.",
]

SINGLE_OPERAND_TEMPLATES = [
    "{name1} has {amt1}$ in {poss} wallet.",
    "My {rel1} {name1} currently possesses {amt1}$ after the transaction.",
    "{name1}, who works as a {occupation}, owns {amt1}$ in savings.",
    "After all expenses, {name1} is left with {amt1}$.",
]

MULTI_OPERAND_TEMPLATES = [
    "{names_list} each have {amounts}$ respectively in their pockets after sharing {obj}.",
    "The group including {names_list} gained {amounts}$ respectively from the fundraiser for {obj}.",
    "{names_list} received {amounts}$ as their shares from selling {obj}.",
    "After the event, {names_list} found themselves with {amounts}$ respectively.",
]


def detect_intent(sentence: str, operands: List[int]) -> str:
    """
    Detect intent from sentence and operands.
    
    Rules:
    - If exactly 2 operands and subtract cue: return "a - b"
    - Otherwise: return "sum: a, b, ..."
    """
    sentence_lower = sentence.lower()
    
    # Check for subtract cues
    has_subtract = any(verb in sentence_lower for verb in ['lost', 'gave away', 'spent', 'donated', 'sacrificed']) or ' - ' in sentence
    
    # Check for add cues (overrides subtract if present)
    has_add = any(verb in sentence_lower for verb in ['gained', 'received', 'added', 'got', 'obtained', 'found']) or ' + ' in sentence
    
    # Intent logic
    if len(operands) == 2 and has_subtract and not has_add:
        return f"{operands[0]} - {operands[1]}"
    else:
        return "sum: " + ", ".join(map(str, operands))


def get_same_faith_pair() -> Tuple[str, Dict, Dict]:
    """Get a random same-faith name pair with genders and relationships."""
    faith = random.choice(['hindu', 'muslim', 'christian'])
    
    # Randomly pick genders for two people
    gender1 = random.choice(['male', 'female'])
    gender2 = random.choice(['male', 'female'])
    
    name1 = random.choice(NAMES[faith][gender1])
    name2 = random.choice(NAMES[faith][gender2])
    
    # Ensure no same-sex romantic pairs (no LGBT)
    rel1 = random.choice(RELATIONSHIPS[gender1])
    rel2 = random.choice(RELATIONSHIPS[gender2])
    
    # Avoid boyfriend-boyfriend, girlfriend-girlfriend
    if gender1 == gender2:
        if rel1 in ['boyfriend', 'girlfriend']:
            rel1 = random.choice(['friend', 'teacher', 'brother' if gender1 == 'male' else 'sister'])
        if rel2 in ['boyfriend', 'girlfriend']:
            rel2 = random.choice(['friend', 'teacher', 'brother' if gender2 == 'male' else 'sister'])
    
    person1 = {'name': name1, 'gender': gender1, 'rel': rel1}
    person2 = {'name': name2, 'gender': gender2, 'rel': rel2}
    
    return faith, person1, person2


def generate_two_operand_sentence() -> Dict[str, any]:
    """Generate a sentence with exactly 2 operands (90% of data)."""
    operands = [random.randint(1, 100), random.randint(1, 100)]
    
    # Decide intent (50/50 add vs subtract for variety)
    is_subtract = random.random() < 0.5
    
    faith, person1, person2 = get_same_faith_pair()
    
    if is_subtract:
        verb1 = random.choice(SUBTRACT_VERBS)
        verb2 = random.choice(SUBTRACT_VERBS)
    else:
        verb1 = random.choice(ADD_VERBS)
        verb2 = random.choice(ADD_VERBS)
    
    template = random.choice(VERBOSE_TEMPLATES)
    
    # Fill template
    sentence = template.format(
        name1=person1['name'],
        name2=person2['name'],
        rel1=person1['rel'],
        rel2=person2['rel'],
        verb=verb1,
        verb2=verb2,
        amt1=operands[0],
        amt2=operands[1],
        obj=random.choice(OBJECTS),
        occupation=random.choice(['teacher', 'doctor', 'engineer', 'artist', 'farmer']),
        poss='his' if person2['gender'] == 'male' else 'her',
        context=random.choice(['selling goods', 'work bonus', 'inheritance', 'winning a bet'])
    )
    
    return {'prompt': sentence, 'operands': operands}


def generate_single_operand_sentence() -> Dict[str, any]:
    """Generate a sentence with exactly 1 operand (3% of data)."""
    operands = [random.randint(1, 100)]
    
    faith = random.choice(['hindu', 'muslim', 'christian'])
    gender = random.choice(['male', 'female'])
    name = random.choice(NAMES[faith][gender])
    rel = random.choice(RELATIONSHIPS[gender])
    
    template = random.choice(SINGLE_OPERAND_TEMPLATES)
    
    sentence = template.format(
        name1=name,
        rel1=rel,
        amt1=operands[0],
        poss='his' if gender == 'male' else 'her',
        occupation=random.choice(['teacher', 'doctor', 'engineer', 'artist', 'farmer'])
    )
    
    return {'prompt': sentence, 'operands': operands}


def generate_multi_operand_sentence() -> Dict[str, any]:
    """Generate a sentence with 3-5 operands (7% of data)."""
    num_operands = random.randint(3, 5)
    operands = [random.randint(1, 100) for _ in range(num_operands)]
    
    faith = random.choice(['hindu', 'muslim', 'christian'])
    
    # Generate multiple names from same faith
    names = []
    for _ in range(num_operands):
        gender = random.choice(['male', 'female'])
        names.append(random.choice(NAMES[faith][gender]))
    
    names_list = ', '.join(names[:-1]) + f', and {names[-1]}'
    amounts = ', '.join(map(str, operands))
    
    template = random.choice(MULTI_OPERAND_TEMPLATES)
    
    sentence = template.format(
        names_list=names_list,
        amounts=amounts,
        obj=random.choice(OBJECTS)
    )
    
    return {'prompt': sentence, 'operands': operands}


def generate_dataset(num_samples: int) -> List[Dict]:
    """Generate dataset with specified distribution of operand counts."""
    dataset = []
    
    # Calculate counts based on distribution
    two_operand_count = int(num_samples * 0.90)
    single_operand_count = int(num_samples * 0.03)
    multi_operand_count = num_samples - two_operand_count - single_operand_count
    
    # Generate samples
    for _ in range(two_operand_count):
        dataset.append(generate_two_operand_sentence())
    
    for _ in range(single_operand_count):
        dataset.append(generate_single_operand_sentence())
    
    for _ in range(multi_operand_count):
        dataset.append(generate_multi_operand_sentence())
    
    # Shuffle to mix operand types
    random.shuffle(dataset)
    
    return dataset


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic intent detection training data')
    parser.add_argument('--scale', type=int, default=100, choices=range(1, 101),
                        help='Percentage of data to generate (1-100, default: 100)')
    args = parser.parse_args()
    
    # Set seed for reproducibility
    random.seed(42)
    
    # Calculate actual counts based on scale
    training_count = int(100000 * args.scale / 100)
    testing_count = int(20000 * args.scale / 100)
    
    print(f"Generating {training_count} training samples...")
    training_data = generate_dataset(training_count)
    
    print(f"Generating {testing_count} testing samples...")
    testing_data = generate_dataset(testing_count)
    
    # Save to JSON files
    with open('training_data.json', 'w') as f:
        json.dump(training_data, f, indent=2)
    print(f"Saved training_data.json ({len(training_data)} samples)")
    
    with open('testing_data.json', 'w') as f:
        json.dump(testing_data, f, indent=2)
    print(f"Saved testing_data.json ({len(testing_data)} samples)")
    
    # Demo: Show intent detection for first 5 test samples
    print("\n=== Intent Detection Demo (First 5 Test Samples) ===")
    for i, sample in enumerate(testing_data[:5], 1):
        intent = detect_intent(sample['prompt'], sample['operands'])
        print(f"\nSample {i}:")
        print(f"  Prompt: {sample['prompt']}")
        print(f"  Operands: {sample['operands']}")
        print(f"  Detected Intent: {intent}")


if __name__ == '__main__':
    main()
