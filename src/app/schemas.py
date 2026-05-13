from typing import Literal

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    model_config = {"extra": "forbid"}  # rejects any unexpected fields

    Age: int = Field(..., ge=18, le=75, example=28)
    Gender: Literal["Male", "Female", "Other"]
    self_employed: Literal["Yes", "No"]
    family_history: Literal["Yes", "No"]
    work_interfere: Literal["Often", "Rarely", "Sometimes", "Never"]
    no_employees: Literal["1-5", "6-25", "26-100", "100-500", "500-1000", "More than 1000"]
    remote_work: Literal["Yes", "No"]
    tech_company: Literal["Yes", "No"]
    benefits: Literal["Yes", "No", "Don't know"]
    care_options: Literal["Yes", "No", "Not sure"]
    wellness_program: Literal["Yes", "No", "Don't know"]
    seek_help: Literal["Yes", "No", "Don't know"]
    anonymity: Literal["Yes", "No", "Don't know"]
    leave: Literal["Very easy", "Somewhat easy", "Somewhat difficult", "Very difficult", "Don't know"]
    mental_health_consequence: Literal["Yes", "No", "Maybe"]
    phys_health_consequence: Literal["Yes", "No", "Maybe"]
    coworkers: Literal["Yes", "No", "Some of them"]
    supervisor: Literal["Yes", "No", "Some of them"]
    mental_health_interview: Literal["Yes", "No", "Maybe"]
    phys_health_interview: Literal["Yes", "No", "Maybe"]
    mental_vs_physical: Literal["Yes", "No", "Don't know"]
    obs_consequence: Literal["Yes", "No"]

    @field_validator("*", mode="before")
    @classmethod
    def strip_and_normalize(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v