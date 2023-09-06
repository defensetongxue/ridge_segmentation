import json
import yaml

def yaml_to_json(yaml_file_path, json_file_path):
    # Read YAML file
    with open(yaml_file_path, 'r') as yaml_file:
        yaml_data = yaml.safe_load(yaml_file)

    # Convert to JSON format
    json_data = json.dumps(yaml_data, indent=4)

    # Write to JSON file
    with open(json_file_path, 'w') as json_file:
        json_file.write(json_data)

# Example usage
yaml_file_path = './YAML/default.yaml'
json_file_path = './config_file/default.json.'
yaml_to_json(yaml_file_path, json_file_path)



