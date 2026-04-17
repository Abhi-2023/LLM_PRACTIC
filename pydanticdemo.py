from pydantic import BaseModel
from typing import Optional
class Student(BaseModel):
    
    name: str
    age:Optional[int] = None
    
    
new_Student = {'name':'Abhishek Gupta'}
student = Student(**new_Student)

