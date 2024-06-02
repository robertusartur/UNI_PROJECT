from fastapi import FastAPI, Depends, HTTPException, status, Header, Query
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, EmailStr, validator
from jose import JWTError, jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware

# Constants
SECRET_KEY = "52367badbf4e42f3a94d9ce456e1f01cbfee36a604da5c9589fa84f0bb9e661b"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize application and database
app = FastAPI()
# Настройка CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:8800', 'http://127.0.0.1:8800'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SQLALCHEMY_DATABASE_URL = "postgresql://postgres:1234@localhost/postgres7"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Database model
class UserORM(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    phone = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    password = Column(String)

Base.metadata.create_all(bind=engine)

# Pydantic models
class Token(BaseModel):
    access_token: str
    token_type: str

class LoginRequest(BaseModel):
    phone: str
    password: str

class NewUser(BaseModel):
    phone: str
    password: str
    confirm_password: str
    email: EmailStr = None

    @validator('confirm_password')
    def passwords_match(cls, v, values, **kwargs):
        if 'password' in values and v != values['password']:
            raise ValueError('passwords do not match')
        return v

class NewUserResponse(BaseModel):
    message: str

class UserUpdateRequest(BaseModel):
    phone: str = None
    email: EmailStr = None
    name: str = None
    password: str = None

class UserProfileResponse(BaseModel):
    id: int
    phone: str
    email: EmailStr = None
    name: str = None

# Utility to create a token
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# Function to get a database session
def get_session_local():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Dependency to get the current user
def get_current_user(token: str = Query(...), db: Session = Depends(get_session_local)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(UserORM).filter(UserORM.id == user_id).first()
    if user is None:
        raise credentials_exception
    return user

# Endpoint to check token validity
@app.get("/api/check_token")
def check_token(authorization: str = Header(...)):
    token = authorization.split("Bearer ")[-1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return {"message": "Token is valid"}
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is invalid or expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Endpoint for user login
@app.post("/api/login", response_model=Token)
def login(request: LoginRequest, db: Session = Depends(get_session_local)):
    user = db.query(UserORM).filter(UserORM.phone == request.phone).first()
    if not user or not pwd_context.verify(request.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid phone number or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": str(user.id)})
    return {"access_token": access_token, "token_type": "bearer"}

# Endpoint to get current user profile
@app.get("/api/users/me", response_model=UserProfileResponse)
def read_users_me(current_user: UserORM = Depends(get_current_user)):
    return UserProfileResponse(
        id=current_user.id,
        phone=current_user.phone,
        email=current_user.email,
        name=current_user.name
    )

# Endpoint to register a new user
@app.post("/api/register", response_model=NewUserResponse)
def register_user(new_user: NewUser, db: Session = Depends(get_session_local)):
    if db.query(UserORM).filter(UserORM.phone == new_user.phone).first():
        raise HTTPException(status_code=400, detail="Phone number already registered")
    if new_user.email and db.query(UserORM).filter(UserORM.email == new_user.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_password = pwd_context.hash(new_user.password)
    user = UserORM(
        phone=new_user.phone,
        password=hashed_password,
        email=new_user.email,
        name=""  # Initialize name to an empty string
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return NewUserResponse(message="User registered successfully!")

# Endpoint to update user information
@app.post("/api/users/{user_id}/update", response_model=UserProfileResponse)
def update_user(user_id: int, update_request: UserUpdateRequest, current_user: UserORM = Depends(get_current_user), db: Session = Depends(get_session_local)):
    user = db.query(UserORM).filter(UserORM.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    if update_request.phone is not None:
        user.phone = update_request.phone
    if update_request.name is not None:
        user.name = update_request.name
    if update_request.email is not None:
        user.email = update_request.email
    if update_request.password is not None:
        user.password = pwd_context.hash(update_request.password)
    db.commit()
    db.refresh(user)
    return user

# Endpoint to delete a user
@app.delete("/api/users/{user_id}")
def delete_user(user_id: int, current_user: UserORM = Depends(get_current_user), db: Session = Depends(get_session_local)):
    user = db.query(UserORM).filter(UserORM.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    db.delete(user)
    db.commit()
    return {"message": "User deleted successfully"}

# Endpoint to get a user by ID
@app.get("/api/users/{user_id}", response_model=UserProfileResponse)
def get_user(user_id: int, current_user: UserORM = Depends(get_current_user), db: Session = Depends(get_session_local)):
    user = db.query(UserORM).filter(UserORM.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user




