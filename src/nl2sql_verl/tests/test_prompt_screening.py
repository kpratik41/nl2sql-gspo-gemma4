from src.nl2sql_verl.screen_training_prompts import select_indices


def test_selection_keeps_mixed_groups_and_only_tenth_of_perfect_groups():
    hits = [0, 1, 15, 0] + [16] * 27
    chosen, mixed, easy = select_indices(hits, seed=37)
    assert mixed == [1, 2]
    assert len(easy) == 2  # floor(27 * 10%)
    assert set(chosen) == {1, 2} | set(easy)
    assert all(hits[i] == 16 for i in easy)
    assert not {0, 3} & set(chosen)
    assert select_indices(hits, seed=37) == (chosen, mixed, easy)
    assert select_indices(hits, seed=38)[2] != easy


def test_small_perfect_pool_rounds_down_and_no_all_wrong_prompt_is_kept():
    assert select_indices([0, 0, 16, 1]) == ([3], [3], [])


def test_complete_export_preserves_shuffled_source_rows_and_rejects_missing_rollouts(tmp_path):
    import json
    from types import SimpleNamespace
    import pyarrow.parquet as pq
    import pyarrow as pa
    import pytest
    from src.nl2sql_verl.screen_training_prompts import select, prompt_hash, file_hash

    # Same dimensions as production, deliberately shuffled reference IDs.
    rows = [{'prompt': [{'role': 'user', 'content': f'question {i}'}],
             'extra_info': {'reference_index': 6600-i}} for i in range(6601)]
    table = pa.Table.from_pylist(rows)
    data = tmp_path / 'input.parquet'
    pq.write_table(table, data)
    digest = file_hash(data)
    hits = [1, 15] + [16] * 20 + [0] * 6579
    for shard in range(12):
        directory = tmp_path / f'shard-{shard:02d}'
        directory.mkdir()
        (directory / 'heartbeat.json').write_text(json.dumps({'state': 'completed'}))
        (directory / 'provenance.json').write_text(json.dumps({'contract': {
            'data_sha256': digest, 'shard': shard}}))
        with (directory / 'results.jsonl').open('w') as stream:
            for i in range(shard, 6601, 12):
                fingerprint = prompt_hash(rows[i]['prompt'])
                for j in range(16):
                    stream.write(json.dumps({'data_index': i, 'rollout_index': j,
                        'reference_index': 6600-i, 'source_prompt_sha256': fingerprint,
                        'seed': 20261029+i*100000+j*1000,
                        'scores': {'result': int(j < hits[i])}})+'\n')
    arguments = SimpleNamespace(output_dir=str(tmp_path), data=str(data), selection_seed=37)
    select(arguments)
    selection = json.loads((tmp_path / 'selection.json').read_text())
    expected, mixed, easy = select_indices(hits, 37)
    assert selection['selected_prompts'] == 4
    assert selection['heterogeneous'] == 2 and selection['sampled_all_correct'] == 2
    assert selection['all_incorrect'] == 6579
    assert selection['pass_at_16_count'] == 22
    assert pq.read_table(tmp_path/'selected_train.parquet').equals(table.take(pa.array(expected)))
    path = tmp_path / 'shard-00/results.jsonl'
    with path.open() as stream:
        lines = stream.readlines()
    path.write_text(''.join(lines[:-1]))
    with pytest.raises(AssertionError):
        select(arguments)
