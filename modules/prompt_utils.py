import json
import os

ENTITY_DIR = "semantic/entities/"

def generate_entity_tokens(class_name):
    """
    Load a list of semantic entity tokens from a pre-saved JSON file
    corresponding to the given class name.

    Args:
        class_name (str): e.g., "goldfish"

    Returns:
        List[str]: List of semantic entity tokens for the class
    """
    class_file = os.path.join(ENTITY_DIR, f"{class_name.lower()}.json")

    if not os.path.exists(class_file):
        print(f"[Warning] Entity file not found: {class_file}")
        return [class_name.lower()]  # fallback: use the class name itself

    with open(class_file, "r") as f:
        data = json.load(f)

    return data.get("entities", [class_name.lower()])