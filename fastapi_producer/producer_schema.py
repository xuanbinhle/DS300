from pydantic import BaseModel, Field

class ProducerMessage(BaseModel):
    message : str = Field(min_length=1, max_length=500, description="The content of the message to be produced.")
    