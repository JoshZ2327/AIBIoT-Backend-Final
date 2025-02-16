import os
import time
import logging
import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import sessionmaker, declarative_base, Session
from passlib.context import CryptContext
import jwt
import openai

# ----- Logging Setup -----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----- Configuration -----
DATABASE_URL = "sqlite:///./aibiot.db"
SECRET_KEY = "your-secret-key"  # Replace with a secure random value in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Set the OpenAI API key from an environment variable
openai.api_key = os.environ.get("OPENAI_API_KEY")

# ----- Database Setup -----
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class UserModel(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

# Create the database tables if they do not exist
Base.metadata.create_all(bind=engine)

# ----- Pydantic Schemas -----
class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: int
    created_at: datetime.datetime

    class Config:
        orm_mode = True

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    response: str

# ----- Security Utilities -----
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: Optional[datetime.timedelta] = None) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.datetime.utcnow() + expires_delta
    else:
        expire = datetime.datetime.utcnow() + datetime.timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# ----- Database Dependency -----
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user_by_email(db: Session, email: str) -> Optional[UserModel]:
    return db.query(UserModel).filter(UserModel.email == email).first()

def authenticate_user(db: Session, email: str, password: str) -> Optional[UserModel]:
    user = get_user_by_email(db, email)
    if not user or not verify_password(password, user.hashed_password):
        return None
    return user

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> UserModel:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
        token_data = TokenData(email=email)
    except jwt.PyJWTError:
        raise credentials_exception
    user = get_user_by_email(db, email=token_data.email)
    if user is None:
        raise credentials_exception
    return user

# ----- FastAPI Application Setup -----
app = FastAPI(title="AIBIoT Robust AI Engine Backend")

# Enable CORS (adjust origins for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----- Endpoints -----

@app.post("/signup", response_model=UserResponse)
def signup(user: UserCreate, db: Session = Depends(get_db)):
    if get_user_by_email(db, user.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    hashed_pw = get_password_hash(user.password)
    new_user = UserModel(email=user.email, full_name=user.full_name, hashed_password=hashed_pw)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    logger.info(f"New user registered: {new_user.email}")
    return new_user

@app.post("/login", response_model=Token)
def login(user: UserCreate, db: Session = Depends(get_db)):
    db_user = authenticate_user(db, user.email, user.password)
    if not db_user:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    access_token = create_access_token(data={"sub": db_user.email})
    logger.info(f"User logged in: {db_user.email}")
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/query", response_model=QueryResponse)
def query_data(query_request: QueryRequest, current_user: UserModel = Depends(get_current_user)):
    logger.info(f"Received query from {current_user.email}: {query_request.query}")
    try:
        completion = openai.Completion.create(
            engine="text-davinci-003",
            prompt=query_request.query,
            max_tokens=150,
            temperature=0.7,
        )
        response_text = completion.choices[0].text.strip()
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise HTTPException(status_code=500, detail="Error generating AI response")
    return QueryResponse(response=response_text)

@app.post("/train")
def train_model(background_tasks: BackgroundTasks, current_user: UserModel = Depends(get_current_user)):
    def simulate_training():
        logger.info(f"Training started for {current_user.email}")
        time.sleep(5)  # Simulate time-consuming training
        logger.info(f"Training completed for {current_user.email}")
    background_tasks.add_task(simulate_training)
    return {"message": "Model training started"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# ----- Main Entry Point -----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("index:app", host="0.0.0.0", port=8000, reload=True)
