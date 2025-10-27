from pydantic import BaseModel, EmailStr, Field

class UserSignup(BaseModel): #validation schema when registering new user
    username: str = Field(..., min_length=3)
    email: EmailStr
    password: str = Field(..., min_length=8)
    first_name: str = Field(..., min_length=1)
    last_name: str = Field(..., min_length=1)

class UserLogin(BaseModel): #validation schema for logging in
    username: str
    password: str