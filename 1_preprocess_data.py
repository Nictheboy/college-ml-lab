import json
import os
from pathlib import Path

# Define file paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_INPUT_FILE = DATA_DIR / "train_data.json"
TEST_INPUT_FILE = DATA_DIR / "test_data.json"
TRAIN_OUTPUT_FILE = DATA_DIR / "train_processed.jsonl"
TEST_OUTPUT_FILE = DATA_DIR / "test_processed.jsonl"

def process_plan_recursively(node, is_train_set, actual_rows_capture_list):
    """
    Recursively processes a node (dict or list) in the explain plan.
    - Removes keys starting with "Actual".
    - If is_train_set is True and a key "Actual Rows" is encountered for the
      first time in this item's plan, its value is stored in actual_rows_capture_list[0].
    Modifies the node in-place.
    """
    if isinstance(node, dict):
        keys_to_delete = []
        # Iterate over a copy of items for safe modification
        for key, value in list(node.items()):
            if key.startswith("Actual"):
                if is_train_set and key == "Actual Rows" and actual_rows_capture_list[0] is None:
                    actual_rows_capture_list[0] = value
                keys_to_delete.append(key)
            else:
                # Recurse for nested structures
                process_plan_recursively(value, is_train_set, actual_rows_capture_list)
        
        for key in keys_to_delete:
            del node[key]
            # For test set, if we needed to set to null instead of deleting:
            # if not is_train_set:
            #     node[key] = None # This part is not requested based on "直接去掉"

    elif isinstance(node, list):
        for item in node:
            process_plan_recursively(item, is_train_set, actual_rows_capture_list)

def preprocess_file(input_path: Path, output_path: Path, is_train_set: bool):
    """
    Processes a single JSON input file and writes to JSONL output.
    """
    print(f"Processing {'train' if is_train_set else 'test'} data: {input_path} -> {output_path}")
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return

    processed_count = 0
    with open(input_path, 'r', encoding='utf-8') as infile, \
         open(output_path, 'w', encoding='utf-8') as outfile:
        
        original_data = json.load(infile)
        
        for item in original_data:
            processed_item = {
                "query": item["query"],
                "query_id": item["query_id"]
            }
            
            # 1. Parse the stringified JSON in "explain_result"
            try:
                explain_plan_data = json.loads(item["explain_result"])
            except json.JSONDecodeError as e:
                print(f"Warning: Could not parse explain_result for query_id {item['query_id']}: {e}")
                processed_item["explain_result"] = None # Or handle as an error
                # If explain_result is crucial and cannot be parsed, you might want to skip this item
                # or fill with a specific error structure. For now, setting to None.
                if is_train_set:
                    processed_item["target_actual_rows"] = None
                outfile.write(json.dumps(processed_item) + '\n')
                processed_count += 1
                continue


            # This list is used to capture the first "Actual Rows" value from the train set.
            # It's a list so it can be modified by the recursive function (pass-by-reference-like).
            actual_rows_capture = [None] 

            # 2. Recursively process the plan: remove "Actual*" fields
            #    and extract "Actual Rows" for training set.
            #    The function modifies explain_plan_data in-place.
            process_plan_recursively(explain_plan_data, is_train_set, actual_rows_capture)
            
            processed_item["explain_result"] = explain_plan_data
            
            if is_train_set:
                processed_item["target_actual_rows"] = actual_rows_capture[0]
            
            # 3. Save as JSONL
            outfile.write(json.dumps(processed_item) + '\n')
            processed_count += 1
            
    print(f"Successfully processed {processed_count} items into {output_path}")

def main():
    # Ensure data directory exists (though it should if input files are there)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Process training data
    preprocess_file(TRAIN_INPUT_FILE, TRAIN_OUTPUT_FILE, is_train_set=True)
    
    # Process test data
    preprocess_file(TEST_INPUT_FILE, TEST_OUTPUT_FILE, is_train_set=False)

if __name__ == "__main__":
    main()
