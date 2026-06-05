from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, List, Optional


def _load_skill_headers() -> List[str]:
    try:
        from prompts_suffix_idea import SKILL_HEADERS

        return list(SKILL_HEADERS)
    except Exception:
        return [""]


def skill_name_from_header(header: str) -> str:
    stripped = (header or "").strip()
    if not stripped:
        return "default"
    first_line = stripped.splitlines()[0].strip()
    return first_line.replace("SKILL:", "").strip().lower().replace(" ", "-") or "skill"


def skill_header_map() -> Dict[str, tuple[Optional[int], str]]:
    headers = _load_skill_headers()
    mapping: Dict[str, tuple[Optional[int], str]] = {"default": (None, "")}
    for index, header in enumerate(headers):
        name = skill_name_from_header(header)
        mapping[name] = (index, header)
        mapping[name.replace("_", "-")] = (index, header)
    return mapping


@dataclass(frozen=True)
class SampleSpec:
    sample_id: int
    sample_plan_id: int
    skill_name: str
    skill_id: Optional[int]
    skill_header: str
    temperature: float
    top_p: float

    @property
    def replica_label(self) -> str:
        return f"{self.skill_name}@t{self.temperature:g}/p{self.top_p:g}#{self.sample_plan_id}"


_PLAN_RE = re.compile(
    r"^(?P<skill>[a-zA-Z0-9_-]+):(?P<count>[0-9]+)@(?P<temperature>[0-9]+(?:\.[0-9]+)?)(?:/(?P<top_p>[0-9]+(?:\.[0-9]+)?))?$"
)


def expand_sample_plan(
    sample_plan: str,
    *,
    num_generations: int,
    temperature: float,
    top_p: float,
) -> List[SampleSpec]:
    mapping = skill_header_map()
    plan = (sample_plan or "").strip()
    if not plan:
        if num_generations <= 0:
            raise ValueError("--num_generations must be >= 1 when --sample_plan is omitted")
        return [
            SampleSpec(
                sample_id=sample_id,
                sample_plan_id=0,
                skill_name="default",
                skill_id=None,
                skill_header="",
                temperature=float(temperature),
                top_p=float(top_p),
            )
            for sample_id in range(num_generations)
        ]

    specs: List[SampleSpec] = []
    sample_id = 0
    for plan_id, token in enumerate(part.strip() for part in plan.split(",")):
        if not token:
            raise ValueError(f"Invalid empty sample-plan entry in {sample_plan!r}")
        match = _PLAN_RE.match(token)
        if not match:
            raise ValueError(
                "Invalid --sample_plan entry. Expected skill:count@temperature or "
                f"skill:count@temperature/top_p, got: {token!r}"
            )
        skill_name = match.group("skill").lower().replace("_", "-")
        if skill_name not in mapping:
            allowed = ", ".join(sorted(mapping))
            raise ValueError(f"Unknown sample-plan skill {skill_name!r}. Allowed skills: {allowed}")
        count = int(match.group("count"))
        entry_temperature = float(match.group("temperature"))
        entry_top_p = float(match.group("top_p") or top_p)
        if count <= 0:
            raise ValueError(f"Sample-plan count must be >= 1, got {count} in {token!r}")
        if entry_temperature < 0:
            raise ValueError(f"Sample-plan temperature must be >= 0, got {entry_temperature} in {token!r}")
        if not (0 < entry_top_p <= 1):
            raise ValueError(f"Sample-plan top_p must satisfy 0 < top_p <= 1, got {entry_top_p} in {token!r}")
        skill_id, header = mapping[skill_name]
        for _ in range(count):
            specs.append(
                SampleSpec(
                    sample_id=sample_id,
                    sample_plan_id=plan_id,
                    skill_name=skill_name,
                    skill_id=skill_id,
                    skill_header=header,
                    temperature=entry_temperature,
                    top_p=entry_top_p,
                )
            )
            sample_id += 1
    if not specs:
        raise ValueError("--sample_plan produced no samples")
    return specs
