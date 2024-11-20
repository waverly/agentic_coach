from typing import List, Optional, Dict
from pydantic import BaseModel
from datetime import date


class Employee(BaseModel):
    employee_id: str
    first_name: str
    last_name: str
    email: str
    phone_number: str
    job_title: str
    team_name: str
    manager: Optional[str]  # Can be employee_id or manager's name
    start_date: date
    employment_type: str
    status: str
    location: str
    skills: List[str]
    projects: List[str]
