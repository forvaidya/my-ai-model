#!/usr/bin/env python3
"""
Math Word Problem Dataset Generator

Generates synthetic training and testing data for arithmetic word problems
(addition and subtraction) with realistic linguistic variation and controlled noise.

Usage:
    python generate_math_data.py --scale 50 --seed 42
    
Output:
    __train.json: Training data (100,000 examples by default, scaled by --scale)
    __test.json: Testing data (20,000 examples by default, scaled by --scale)
"""

import json
import random
import argparse
import sys
from typing import List, Dict, Tuple, Any
from collections import defaultdict
from tqdm import tqdm


# ============================================================================
# ACTOR NAMES (Balanced across faiths and genders)
# ============================================================================

ACTORS = {
    "male": {
        "hindu": ["Mahesh", "Rajesh", "Arjun", "Vivek", "Rohan"],
        "muslim": ["Ahmed", "Faisal", "Imran", "Zaid", "Yusuf"],
        "christian": ["John", "David", "Peter", "Paul", "James"],
    },
    "female": {
        "hindu": ["Priya", "Anjali", "Kavya", "Neha", "Shruti"],
        "muslim": ["Aisha", "Fatima", "Sara", "Zainab", "Maryam"],
        "christian": ["Mary", "Anna", "Elizabeth", "Sarah", "Grace"],
    },
}

FAITHS = ["hindu", "muslim", "christian"]
GENDERS = ["male", "female"]

# ============================================================================
# RELATIONSHIPS
# ============================================================================

RELATIONSHIPS = ["friend", "teacher", "student", "senior", "junior"]

# ============================================================================
# ASSETS & UNITS
# ============================================================================

ASSETS = ["$", "₹", "marbles", "candies", "books", "pencils"]

# ============================================================================
# VERBS
# ============================================================================

ADDITION_VERBS = ["get", "receive", "add"]
SUBTRACTION_VERBS = ["give", "pay", "gift"]

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def get_all_actors() -> List[str]:
    """Return all actor names as a flat list."""
    actors = []
    for gender in GENDERS:
        for faith in FAITHS:
            actors.extend(ACTORS[gender][faith])
    return actors


def get_actor_faith(actor: str) -> str:
    """Return the faith of an actor."""
    for gender in GENDERS:
        for faith in FAITHS:
            if actor in ACTORS[gender][faith]:
                return faith
    return None


def get_actor_gender(actor: str) -> str:
    """Return the gender of an actor."""
    for gender in GENDERS:
        for faith in FAITHS:
            if actor in ACTORS[gender][faith]:
                return gender
    return None


def get_pronoun(actor: str) -> str:
    """Return appropriate pronoun for actor."""
    gender = get_actor_gender(actor)
    return "he" if gender == "male" else "she"


def get_possessive(actor: str) -> str:
    """Return possessive form of actor name."""
    return f"{actor}'s"


def format_asset_value(value: int, asset: str) -> str:
    """Format asset value with unit, with random variation."""
    if asset in ["$", "₹"]:
        # Variation: symbol before/after, with/without space
        variations = [
            f"{asset}{value}",
            f"{value}{asset}",
            f"{asset} {value}",
            f"{value} {asset}",
        ]
        return random.choice(variations)
    else:
        # For non-currency assets
        return f"{value} {asset}"


def pluralize(word: str, count: int) -> str:
    """Simple pluralization."""
    if count == 1:
        return word
    if word.endswith("y"):
        return word[:-1] + "ies"
    return word + "s"


def generate_operands(num_operands: int) -> List[int]:
    """Generate random operand values (1-100)."""
    return [random.randint(1, 100) for _ in range(num_operands)]


def select_actors() -> Tuple[str, str]:
    """Select two different actors."""
    all_actors = get_all_actors()
    actor1, actor2 = random.sample(all_actors, 2)
    return actor1, actor2


def generate_prompt_one_operand(actor1: str, asset: str, value: int) -> str:
    """Generate prompt for single operand (always addition)."""
    asset_str = pluralize(asset, value)
    formatted_value = format_asset_value(value, asset)
    
    variations = [
        f"{actor1} has {formatted_value} {asset_str}.",
        f"{actor1} owns {formatted_value} {asset_str}.",
        f"{actor1} has {formatted_value} {asset_str}.",
    ]
    return random.choice(variations)


def generate_prompt_two_operands(
    actor1: str, actor2: str, asset: str, value1: int, value2: int, is_addition: bool
) -> str:
    """Generate prompt for two operands (addition or subtraction)."""
    pronoun = get_pronoun(actor1)
    asset_str1 = pluralize(asset, value1)
    asset_str2 = pluralize(asset, value2)
    formatted_value1 = format_asset_value(value1, asset)
    formatted_value2 = format_asset_value(value2, asset)
    
    if is_addition:
        verb = random.choice(ADDITION_VERBS)
        preposition = random.choice(["from", "off"])
        variations = [
            f"{actor1} had {formatted_value1} {asset_str1} and {pronoun} {verb} {formatted_value2} {asset_str2} {preposition} {actor2}.",
            f"{actor1}, who had {formatted_value1} {asset_str1}, {verb} {formatted_value2} {asset_str2} {preposition} {actor2}.",
            f"Having {formatted_value1} {asset_str1}, {actor1} {verb} {formatted_value2} {asset_str2} {preposition} {actor2}.",
            f"{actor1} had {formatted_value1} {asset_str1}. {pronoun.capitalize()} {verb} {formatted_value2} {asset_str2} {preposition} {actor2}.",
        ]
    else:
        verb = random.choice(SUBTRACTION_VERBS)
        variations = [
            f"{actor1} had {formatted_value1} {asset_str1} and {pronoun} {verb} {formatted_value2} {asset_str2} to {actor2}.",
            f"{actor1}, who had {formatted_value1} {asset_str1}, {verb} {formatted_value2} {asset_str2} to {actor2}.",
            f"Having {formatted_value1} {asset_str1}, {actor1} {verb} {formatted_value2} {asset_str2} to {actor2}.",
            f"{actor1} had {formatted_value1} {asset_str1}. {pronoun.capitalize()} {verb} {formatted_value2} {asset_str2} to {actor2}.",
        ]
    
    return random.choice(variations)


def generate_prompt_three_operands(
    actor1: str, asset: str, values: List[int]
) -> str:
    """Generate prompt for three operands (always addition/sum)."""
    asset_strs = [pluralize(asset, v) for v in values]
    formatted_values = [format_asset_value(v, asset) for v in values]
    
    # Build list of items
    items = []
    for i, (fv, asset_str) in enumerate(zip(formatted_values, asset_strs)):
        if i == len(formatted_values) - 1:
            items.append(f"and {fv} {asset_str}")
        else:
            items.append(f"{fv} {asset_str}")
    
    items_str = ", ".join(items)
    
    variations = [
        f"There are {items_str}.",
        f"{actor1} has {items_str}.",
        f"In total, there are {items_str}.",
    ]
    return random.choice(variations)


def generate_example() -> Dict[str, Any]:
    """Generate a single training example."""
    num_operands = random.choices([1, 2, 3], weights=[20, 60, 20])[0]
    asset = random.choice(ASSETS)
    
    if num_operands == 1:
        value = random.randint(1, 100)
        actor1, _ = select_actors()
        prompt = generate_prompt_one_operand(actor1, asset, value)
        operation = {"kind": "add", "operands": [value]}
    
    elif num_operands == 2:
        is_addition = random.choice([True, False])
        value1, value2 = generate_operands(2)
        actor1, actor2 = select_actors()
        prompt = generate_prompt_two_operands(
            actor1, actor2, asset, value1, value2, is_addition
        )
        operation = {
            "kind": "add" if is_addition else "subtract",
            "operands": [value1, value2],
        }
    
    else:  # num_operands == 3
        values = generate_operands(3)
        actor1, _ = select_actors()
        prompt = generate_prompt_three_operands(actor1, asset, values)
        operation = {"kind": "add", "operands": values}
    
    return {"prompt": prompt, "operation": operation}


def compute_result(operation: Dict[str, Any]) -> int:
    """Compute the expected result from an operation."""
    kind = operation["kind"]
    operands = operation["operands"]
    
    if kind == "add":
        return sum(operands)
    elif kind == "subtract":
        return operands[0] - operands[1]
    else:
        raise ValueError(f"Unknown operation kind: {kind}")


def validate_example(example: Dict[str, Any]) -> bool:
    """Validate that an example is well-formed."""
    if "prompt" not in example or "operation" not in example:
        return False
    
    operation = example["operation"]
    if "kind" not in operation or "operands" not in operation:
        return False
    
    if operation["kind"] not in ["add", "subtract"]:
        return False
    
    operands = operation["operands"]
    if not isinstance(operands, list) or len(operands) < 1 or len(operands) > 3:
        return False
    
    if not all(isinstance(op, int) and op > 0 for op in operands):
        return False
    
    return True


def check_faith_balance(examples: List[Dict[str, Any]]) -> Dict[str, int]:
    """Check faith distribution in generated examples."""
    faith_counts = defaultdict(int)
    
    for example in examples:
        prompt = example["prompt"]
        all_actors = get_all_actors()
        
        for actor in all_actors:
            if actor in prompt:
                faith = get_actor_faith(actor)
                faith_counts[faith] += 1
                break
    
    return dict(faith_counts)


def run_unit_tests(examples: List[Dict[str, Any]], sample_size: float = 0.005) -> bool:
    """Run unit tests on a sample of examples."""
    sample_count = max(1, int(len(examples) * sample_size))
    sample = random.sample(examples, sample_count)
    
    passed = 0
    failed = 0
    
    for example in sample:
        if not validate_example(example):
            failed += 1
            continue
        
        operation = example["operation"]
        result = compute_result(operation)
        
        # Basic sanity check: result should be positive
        if result <= 0:
            failed += 1
        else:
            passed += 1
    
    print(f"\n[Unit Tests] Tested {sample_count} examples: {passed} passed, {failed} failed")
    return failed == 0


def generate_dataset(
    num_train: int = 100000,
    num_test: int = 20000,
    seed: int = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generate training and testing datasets."""
    if seed is not None:
        random.seed(seed)
    
    print(f"Generating {num_train} training examples...")
    train_data = []
    seen_prompts = set()
    
    pbar = tqdm(total=num_train, desc="Training")
    while len(train_data) < num_train:
        example = generate_example()
        prompt = example["prompt"]
        
        # Avoid duplicates
        if prompt not in seen_prompts:
            train_data.append(example)
            seen_prompts.add(prompt)
            pbar.update(1)
    pbar.close()
    
    print(f"Generating {num_test} testing examples...")
    test_data = []
    
    pbar = tqdm(total=num_test, desc="Testing")
    while len(test_data) < num_test:
        example = generate_example()
        prompt = example["prompt"]
        
        # Avoid duplicates and overlap with training
        if prompt not in seen_prompts:
            test_data.append(example)
            seen_prompts.add(prompt)
            pbar.update(1)
    pbar.close()
    
    return train_data, test_data


def save_dataset(data: List[Dict[str, Any]], filename: str) -> None:
    """Save dataset to JSON Lines format."""
    with open(filename, "w") as f:
        for example in data:
            f.write(json.dumps(example) + "\n")
    print(f"Saved {len(data)} examples to {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic math word problem datasets"
    )
    parser.add_argument(
        "--scale",
        type=int,
        default=100,
        help="Scale factor (1-100): output only scale%% of total rows (default: 100)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--no-tests",
        action="store_true",
        help="Skip unit tests",
    )
    
    args = parser.parse_args()
    
    # Validate scale
    if not 1 <= args.scale <= 100:
        print("Error: --scale must be between 1 and 100")
        sys.exit(1)
    
    # Calculate dataset sizes based on scale
    base_train = 100000
    base_test = 20000
    num_train = max(1, int(base_train * args.scale / 100))
    num_test = max(1, int(base_test * args.scale / 100))
    
    print(f"Configuration:")
    print(f"  Scale: {args.scale}%")
    print(f"  Training examples: {num_train}")
    print(f"  Testing examples: {num_test}")
    print(f"  Seed: {args.seed if args.seed else 'random'}")
    print()
    
    # Generate datasets
    train_data, test_data = generate_dataset(num_train, num_test, args.seed)
    
    # Run unit tests
    if not args.no_tests:
        print()
        run_unit_tests(train_data + test_data)
    
    # Check faith balance (sample)
    print()
    faith_balance = check_faith_balance(train_data[:1000])
    print(f"[Faith Balance Sample] {faith_balance}")
    
    # Save datasets
    print()
    save_dataset(train_data, "__train.json")
    save_dataset(test_data, "__test.json")
    
    print()
    print("✓ Dataset generation complete!")


if __name__ == "__main__":
    main()
