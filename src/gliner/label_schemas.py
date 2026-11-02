"""Shared label identities and ordered GLiNER prompts for both experiment schemas."""
SOURCE_LABEL_MAP = {
    'DIAGNOSIS': 'diagnosis', 'FACILITY': 'facility',
    'GENDER_IDENTITY': 'gender identity', 'LAB_RESULT': 'lab result',
    'LOCATION_ADDRESS': 'location address', 'MEDICATION': 'medication',
    'PERSON': 'person', 'PROCEDURE': 'procedure', 'PROVIDER': 'provider',
    'SEX': 'sex', 'UNIQUE_ID': 'unique identifier',
}
ORIGINAL_LABELS = tuple(SOURCE_LABEL_MAP.values())
RECOMMENDED_LABEL_MAPPING = {
    'provider': 'name medical professional',
    'facility': 'organization medical facility',
    'person': 'name',
    'location address': 'location address',
    'diagnosis': 'condition',
    'unique identifier': 'personal identifier number',
    'sex': 'sex assigned at birth',
    'gender identity': 'gender identity',
    'procedure': 'medical procedure or test performed',
    'medication': 'medication, drug, or prescription',
    'lab result': 'lab result or test finding',
}
SCHEMAS = ('original', 'recommended')


def get_labels(schema='original'):
    if schema not in SCHEMAS:
        raise ValueError(f'Unknown label_schema {schema!r}; choose {SCHEMAS}')
    return ORIGINAL_LABELS if schema == 'original' else tuple(RECOMMENDED_LABEL_MAPPING[x] for x in ORIGINAL_LABELS)


def source_label_map(schema='original'):
    names = dict(zip(ORIGINAL_LABELS, get_labels(schema)))
    return {key: names[label] for key, label in SOURCE_LABEL_MAP.items()}


def provider_label(schema='original'):
    return source_label_map(schema)['PROVIDER']


def schema_metadata(schema='original'):
    labels = get_labels(schema)
    return {'label_schema': schema, 'labels': list(labels),
            'label_mapping': dict(zip(ORIGINAL_LABELS, labels)),
            'label_to_original': dict(zip(labels, ORIGINAL_LABELS))}


def check_model_schema(model, schema):
    """Reject an explicitly recorded model/prompt mismatch; legacy exports lack metadata."""
    get_labels(schema)
    recorded = getattr(getattr(model, 'config', None), 'label_schema', None)
    if recorded is not None and recorded != schema:
        raise ValueError(f'Model label_schema={recorded!r} differs from requested {schema!r}')
