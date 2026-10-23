"""Post-review factorial generation conditions; primary detector remains unchanged."""
import copy
from src.revision.grounding import SYSTEM_PROMPT, expected_output


def output_schema(value):
    """Constrain structure/types, never reveal the correct numerical answers."""
    if isinstance(value, dict):
        return {'type': 'object', 'properties': {k: output_schema(v) for k,v in value.items()},
                'required': list(value), 'additionalProperties': False}
    if isinstance(value, list):
        if not value:
            raise ValueError('Cannot infer schema from empty array')
        schemas = [output_schema(v) for v in value]
        if any(s != schemas[0] for s in schemas):
            raise ValueError('Heterogeneous array requires an explicit schema')
        return {'type': 'array', 'items': schemas[0], 'minItems': len(value), 'maxItems': len(value)}
    if type(value) is int:
        return {'type': 'integer'}
    if type(value) is float:
        return {'type': 'number'}
    if isinstance(value, str):
        return {'type': 'string'}
    raise TypeError(type(value).__name__)


def messages_for(case, examples, few_shot=False, supplied_direction=False):
    prompt = SYSTEM_PROMPT
    if supplied_direction:
        prompt += '\nThe trusted input also provides precomputed_direction. Copy it into direction instead of calculating direction. Do not emit precomputed_direction as an output key.'
    def user(c):
        content = {'values': copy.deepcopy(c['values'])}
        if supplied_direction:
            content['precomputed_direction'] = expected_output(c)['direction']
        return {'role': 'user', 'content': __import__('json').dumps(content)}
    messages = [{'role': 'system', 'content': prompt}]
    if few_shot:
        for c in examples:
            messages += [user(c), {'role': 'assistant', 'content': __import__('json').dumps(expected_output(c))}]
    return messages + [user(case)]


def contract_schema(case):
    schema = output_schema(expected_output(case))
    schema['properties']['direction'] = {'type': 'string', 'enum': ['rising','falling','stable']}
    schema['properties']['study_tier'] = {'type': 'integer', 'minimum': 1, 'maximum': 5}
    return schema
