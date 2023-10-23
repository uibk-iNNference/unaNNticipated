from glob import glob
import json

files = glob('*.json')

predictions = {}
for file in files:
    parts = file.split('-')
    new = '.' in parts[0]

    if new:
        type_parts = parts[0].split('.')
    else:
        type_parts = parts[0].split('-')
    identifier = '-'.join(type_parts)

    identifier += parts[-2]
    identifier += parts[-1]

    with open(file, 'r') as result_file:
        result = json.load(result_file)
        prediction = result['prediction']['bytes']

    if identifier not in predictions:
        predictions[identifier] = prediction

    if not predictions[identifier] == prediction:
        print(f"ERROR: {identifier} contains a different prediction than previous version")
        break

    if not new:
        print(file)
