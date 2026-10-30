"""Same seven reward channels, plus observed tool metrics for future runs."""
from src.nl2sql_verl.task import compute_score as task_score
from src.nl2sql_verl.native_agent_loop import METRICS
from src.nl2sql_verl.consensus.sql_utils import extract_final_answer_sql


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    extra = extra_info or {}
    missing = [name for name in METRICS if name not in extra]
    if missing:
        raise ValueError(f'Native agent telemetry missing: {missing}')
    score_extra = dict(extra)
    if not extract_final_answer_sql(extra.get('assistant_text','')):
        # The shared legacy extractor accepts fenced SQL for non-agent callers.
        # Native tool trajectories must not earn semantic rewards for drafts
        # when they run out of turns/tokens before a completed final answer.
        score_extra['assistant_text'] = ''
    result = task_score(data_source, solution_str, ground_truth, score_extra, **kwargs)
    result.update({name: float(extra[name]) for name in METRICS})
    return result
