from pydantic import BaseModel, Extra

class CarFeatures(BaseModel):
    # Allow any extra fields dynamically
    class Config:
        extra = Extra.allow
