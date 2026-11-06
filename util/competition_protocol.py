"""Public competition constraints, separate from inference and ground truth."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Protocol:
    name: str
    minimum: int
    maximum: int
    input_mode: str


PROTOCOLS = {
    'task1_n1': Protocol('task1_n1', 2, 3, 'single'),
    'task1_n2': Protocol('task1_n2', 3, 4, 'single'),
    'task2': Protocol('task2', 0, 5, 'multi'),
}


def protocol_for_item(item):
    if item in ('1-1', '1-3'): return PROTOCOLS['task1_n1']
    if item in ('1-2', '1-4'): return PROTOCOLS['task1_n2']
    if item in ('2-1', '2-2'): return PROTOCOLS['task2']
    raise ValueError(f'Invalid competition test item: {item!r}')


def resolve_protocol(profile='unconstrained', input_mode=None, test_item=None):
    if profile == 'unconstrained':
        if test_item is not None:
            raise ValueError('test_item requires an explicit competition_profile')
        return None
    if profile not in PROTOCOLS:
        raise ValueError(f'Unknown competition_profile: {profile}')
    result = PROTOCOLS[profile]
    if input_mode is not None and result.input_mode != input_mode:
        raise ValueError(f'{profile} requires unknown_input_mode={result.input_mode}')
    if test_item is not None and protocol_for_item(test_item).name != result.name:
        raise ValueError('test_item and competition_profile disagree')
    return result


def validate_counts(counts):
    """Validate each supplied complete test item, then available N1/N2 relations."""
    errors = []
    for item, count in counts.items():
        p = protocol_for_item(item)
        if not p.minimum <= count <= p.maximum:
            errors.append(f'{item}: unknown class count {count} outside [{p.minimum},{p.maximum}]')
    first = {counts[x] for x in ('1-1', '1-3') if x in counts}
    second = {counts[x] for x in ('1-2', '1-4') if x in counts}
    if len(first) > 1: errors.append('1-1 and 1-3 must share N1')
    if len(second) > 1: errors.append('1-2 and 1-4 must share N2')
    if first and second and not max(first) < min(second):
        errors.append('Task 1 requires N1 < N2')
    return errors


def choose_joint_counts(candidate_scores):
    """Choose N1,N2 from label-free per-item scores; never align class identities."""
    if not candidate_scores or any(x not in ('1-1','1-2','1-3','1-4') for x in candidate_scores):
        raise ValueError('Need task-one per-item candidate scores')
    choices = []
    for n1, n2 in ((2,3),(2,4),(3,4)):
        selected = {item: n1 if item in ('1-1','1-3') else n2 for item in candidate_scores}
        if all(k in candidate_scores[item] for item, k in selected.items()):
            score = sum(candidate_scores[item][k] for item,k in selected.items())
            choices.append((score, -n1, -n2, selected))
    if not choices: raise ValueError('No supported joint N1/N2 candidate')
    return max(choices, key=lambda x:x[:3])[3]
