import os
import json

def combine_json_files(output_file='combined.json'):
    # Get the current directory
    current_directory = os.getcwd()
    # List all files in the current directory
    all_files = os.listdir(current_directory)

    # Filter files to get only JSON files
    json_files = [file for file in all_files if file.endswith('.json')]

    # List to store all objects from JSON files
    combined_objects = []

    # Iterate through each JSON file
    for json_file in json_files:
        file_path = os.path.join(current_directory, json_file)

        # Open and read the JSON file
        with open(file_path, 'r') as file:
            try:
                # Load JSON data
                json_data = json.load(file)

                # If the JSON file contains an array of objects, add them to the combined list
                if isinstance(json_data, list):
                    combined_objects.extend(json_data)
                # If the JSON file contains a single object, add it to the combined list
                elif isinstance(json_data, dict):
                    combined_objects.append(json_data)

            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in file {json_file}: {e}")

    # Write the combined objects to a new JSON file
    with open(output_file, 'w') as output_file:
        json.dump(combined_objects, output_file, indent=2)

    print(f"Combined objects written to {output_file.name}")

if __name__ == "__main__":
    combine_json_files()
