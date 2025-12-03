from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any


class AgentState(BaseModel):
    prompt: str
    classification: List[str] = Field(default_factory=list)
    route_plan: List[str] = Field(default_factory=list)
    route_taken: List[str] = Field(default_factory=list)
    lookup_result: Optional[Dict[str, Any]] = None
    news_result: Optional[Dict[str, Any]] = None
    sector_result: Optional[SectorState] = None


class SectorState(BaseModel):
    source: Optional[str] = None
    prompt: Optional[str] = None
    sectors: Optional[List[str]] = None
    raw_rows: Optional[List[Dict[str, Any]]] = None
    structured_view: Optional[Dict[str, Any]] = None
    interpreted_results: Optional[str] = None
    error: Optional[str] = None
