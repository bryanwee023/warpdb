
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

class UpsertRequest(BaseModel):
    vector: List[float]
    metadata: Optional[Dict[str, Any]] = None