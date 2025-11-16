from dataclasses import dataclass
from typing import List,Dict,Literal


@dataclass
class ChartModel:
    chart_type: Literal["bar", "line", "scatter"]
    title: str
    data: List[Dict[str, float]]
    xKey: str
    yKey: str
    xLabel: str
    yLabel: str