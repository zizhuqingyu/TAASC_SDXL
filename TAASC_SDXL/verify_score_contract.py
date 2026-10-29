"""Check malformed VLM responses cannot be mistaken for valid experiment scores."""
from internvl_server import parse_scores, validate_annotation


def main():
    annotation = validate_annotation({'entities': ['bench', 'car'],
                                      'attributes': [['bench', 'green'], ['car', 'red']],
                                      'relations': [['bench', 'left of', 'car']]})
    result = parse_scores('```json\n{"entities":[1,0],"attributes":[0.9,0.1],"relations":[0.2]}\n```', annotation)
    assert result == {'entities': [1.0, 0.0], 'attributes': [0.9, 0.1], 'relations': [0.2]}
    invalid = [
        '{"entities":[1,0],"attributes":[[1,1],[0,0]],"relations":[0.2]}',
        '{"entities":[true,0],"attributes":[0.9,0.1],"relations":[0.2]}',
        '{"entities":[1,0],"attributes":[0.9,0.1],"relations":[]}',
        '{"entities":[1,0],"attributes":[0.9,0.1],"relations":[NaN]}',
        '{"entities":[1,0],"attributes":[0.9,0.1],"relations":[1.1]}',
        'No JSON output',
    ]
    for raw in invalid:
        try:
            parse_scores(raw, annotation)
        except ValueError:
            continue
        raise AssertionError(f'Accepted malformed response: {raw}')
    print('Score contract checks passed: valid JSON accepted; nested, boolean, length, NaN, range, missing rejected.')


if __name__ == '__main__':
    main()
