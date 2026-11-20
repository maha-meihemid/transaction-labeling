from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator


class Transaction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    amount_value: float = Field(allow_inf_nan=False)
    created_at: AwareDatetime
    fees: float = Field(default=0, ge=0, allow_inf_nan=False)
    card_name: str | None = None
    digitalization_type: str | None = None
    merchant_pan_entry_method: str | None = None

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value):
        value = value.strip()
        if not value:
            raise ValueError("title ne peut pas être vide.")
        return value
