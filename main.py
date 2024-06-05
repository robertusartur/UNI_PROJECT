from ml.model import load_staff
import logging
from fastapi import FastAPI, Depends, HTTPException, status, Query, Body, Request, Path, Header
from fastapi.responses import JSONResponse
from sqlalchemy import create_engine, Column, Integer, String, Text, Float, DateTime, ForeignKey, Boolean, ARRAY, JSON, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker, Session, relationship
from pydantic import BaseModel, EmailStr, validator, HttpUrl
from jose import JWTError, jwt
from datetime import datetime, timedelta
from passlib.context import CryptContext
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import json

# Constants
SECRET_KEY = "52367badbf4e42f3a94d9ce456e1f01cbfee36a604da5c9589fa84f0bb9e661b"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Initialize app and database
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

SQLALCHEMY_DATABASE_URL = "postgresql://test_x95y_user:vJBljgKdl4oJujHfKMJ0VCu6GcyBzCqT@dpg-cpe3jp7109ks73ep4jng-a/test_x95y"
engine = create_engine(SQLALCHEMY_DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database models
class UserORM(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, index=True)
    phone = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    name = Column(String)
    password = Column(String)
    ratings = relationship("UserRatingORM", back_populates="user", cascade="all, delete-orphan")
    houses = relationship("HouseORM", back_populates="user", cascade="all, delete-orphan")
    favorites = relationship("FavoriteORM", back_populates="user", cascade="all, delete-orphan")
    notifications = relationship("NotificationORM", back_populates="user", cascade="all, delete-orphan")
    sent_messages = relationship("MessageORM", foreign_keys="[MessageORM.sender_id]", back_populates="sender", cascade="all, delete-orphan")
    received_messages = relationship("MessageORM", foreign_keys="[MessageORM.recipient_id]", back_populates="recipient", cascade="all, delete-orphan")
    cart = relationship("CartORM", back_populates="user", cascade="all, delete-orphan")

class HouseORM(Base):
    __tablename__ = 'houses'
    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)
    views = Column(Integer, default=0)
    description = Column(String)
    price = Column(Float)
    recommended_price = Column(Float)
    agent_image = Column(String)
    agent_name = Column(String)
    photos = Column(JSON)  # Используем JSON для хранения списка строк
    available_dates = Column(JSON)  # Используем JSON для хранения списка строк
    user_id = Column(Integer, ForeignKey('users.id', ondelete="CASCADE"))
    user = relationship("UserORM", back_populates="houses")

class FavoriteORM(Base):
    __tablename__ = 'favorites'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete="CASCADE"))
    house_id = Column(Integer, ForeignKey('houses.id', ondelete="CASCADE"))
    user = relationship("UserORM", back_populates="favorites")
    house = relationship("HouseORM")

class CartORM(Base):
    __tablename__ = 'cart'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete="CASCADE"))
    house_id = Column(Integer, ForeignKey('houses.id', ondelete="CASCADE"))
    user = relationship("UserORM", back_populates="cart")

class UserRatingORM(Base):
    __tablename__ = 'user_ratings'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete="CASCADE"))
    rating = Column(Float)
    user = relationship("UserORM", back_populates="ratings")

class NotificationORM(Base):
    __tablename__ = 'notifications'
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete="CASCADE"))
    title = Column(String)
    text = Column(Text)
    read = Column(Boolean, default=False)
    user = relationship("UserORM", back_populates="notifications")


class MessageORM(Base):
    __tablename__ = 'messages'
    id = Column(Integer, primary_key=True, index=True)
    sender_id = Column(Integer, ForeignKey('users.id', ondelete="CASCADE"))
    recipient_id = Column(Integer, ForeignKey('users.id', ondelete="CASCADE"))
    content = Column(Text)
    timestamp = Column(DateTime, default=datetime.utcnow)
    dialogue_name = Column(String)
    sender = relationship("UserORM", foreign_keys=[sender_id], back_populates="sent_messages")
    recipient = relationship("UserORM", foreign_keys=[recipient_id], back_populates="received_messages")


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
            raise ValueError('Пароли не совпадают')
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
    name: Optional[str] = None


class HouseCreateRequest(BaseModel):
    title: str
    description: str
    price: float
    recommended_price: float
    agent_image: str
    agent_name: str
    photos: List[str]
    available_dates: List[str]

class HouseResponse(BaseModel):
    id: int
    title: str
    description: str
    price: float
    recommended_price: float
    agent_image: str
    agent_name: str
    photos: List[str]
    available_dates: List[str]

class MessageCreateRequest(BaseModel):
    sender_id: int
    recipient_id: int
    content: str
    dialogue_name: str

class UserAdsResponse(BaseModel):
    id: int
    title: str
    description: str
    price: float
    photos: List[str]
    available_dates: List[str]

class RatingRequest(BaseModel):
    rating: float

    @validator('rating')
    def rating_must_be_between_0_and_5(cls, v):
        if v < 0 or v > 5:
            raise ValueError('Rating must be between 0 and 5')
        return v
class FavouritesResponse(BaseModel):
    id: int
    title: str
    description: str
    price: float
    photos: List[str]
    available_dates: List[str]

class NotificationResponse(BaseModel):
    id: int
    content: str
    read: bool

class MessageResponse(BaseModel):
    id: int
    sender_id: int
    recipient_id: int
    content: str
    timestamp: datetime
    dialogue_name: str

class SuccessMessageResponse(BaseModel):
    message: str

class AdResponse(BaseModel):
    id: int
    title: str
    description: str
    price: float

class UserAdsResponse(BaseModel):
    user_id: int
    ads: List[AdResponse]

class AdIdRequest(BaseModel):
    ad_id: int

class NotificationCreateRequest(BaseModel):
    title: str
    text: str

class NotificationResponse(BaseModel):
    id: int
    title: str
    text: str
    read: bool

class PredictParams(BaseModel):
    datetime: str
    geo_lat: float
    geo_lon: float
    building_type: int
    level: int
    levels: int
    rooms: int
    area: float
    kitchen_area: float
    object_type: int
    studio_apartment: int
    region: str

    class Config:
        schema_extra = {
            "example": {
                "datetime": "2021-08-01",
                "geo_lat": 59.805808,
                "geo_lon": 30.376141,
                "building_type": 1,
                "level": 1,
                "levels": 8,
                "rooms": 3,
                "area": 82.6,
                "kitchen_area": 10.8,
                "object_type": 1,
                "studio_apartment": 0,
                "region": "Санкт-Петербург"
            }
        }
# Utility functions
def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_session_local():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_current_user(token: str = Header(...), db: Session = Depends(get_session_local)):
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


@app.middleware("http")
async def log_requests(request: Request, call_next):
    logger.info(f"Начало запроса path={request.url.path} method={request.method}")
    response = JSONResponse(content={"detail": "Внутренняя ошибка сервера"}, status_code=500)
    try:
        response = await call_next(request)
    except Exception as exc:
        logger.error(f"Произошла ошибка: {exc}")
    finally:
        logger.info(f"Ответ готов status_code={response.status_code}")
        return response

# Routes
@app.post("/api/register", response_model=NewUserResponse)
def register_user(user: NewUser, db: Session = Depends(get_session_local)):
    hashed_password = pwd_context.hash(user.password)
    db_user = UserORM(phone=user.phone, email=user.email, password=hashed_password)
    db.add(db_user)
    db.commit()
    db.refresh(db_user)
    return NewUserResponse(message="Пользователь успешно зарегистрирован")

@app.post("/api/login", response_model=Token)
def login_for_access_token(form_data: LoginRequest, db: Session = Depends(get_session_local)):
    user = db.query(UserORM).filter(UserORM.phone == form_data.phone).first()
    if user is None or not pwd_context.verify(form_data.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный номер телефона или пароль",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(data={"sub": str(user.id)})
    return Token(access_token=access_token, token_type="bearer")

@app.get("/api/user/me", response_model=UserProfileResponse)
def read_users_me(current_user: UserORM = Depends(get_current_user)):
    return UserProfileResponse(
        id=current_user.id,
        phone=current_user.phone,
        email=current_user.email,
        name=current_user.name
    )


@app.put("/users/{user_id}/update", response_model=UserProfileResponse)
def update_user_profile(
        user_id: int,
        user_update: UserUpdateRequest,
        db: Session = Depends(get_session_local),
        current_user: UserORM = Depends(get_current_user)
):
    # Check if the user exists
    db_user = db.query(UserORM).filter(UserORM.id == user_id).first()
    if not db_user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Пользователь не найден")

    # Update user details
    if user_update.phone:
        db_user.phone = user_update.phone
    if user_update.email:
        db_user.email = user_update.email
    if user_update.name:
        db_user.name = user_update.name
    if user_update.password:
        db_user.password = pwd_context.hash(user_update.password)

    # Commit the changes to the database
    db.add(db_user)
    db.commit()
    db.refresh(db_user)

    # Return the updated user profile
    return UserProfileResponse(
        id=db_user.id,
        phone=db_user.phone,
        email=db_user.email,
        name=db_user.name
    )

@app.post("/user/{user_id}/ads", response_model=HouseResponse, status_code=status.HTTP_201_CREATED)
def create_house(user_id: int, house: HouseCreateRequest, db: Session = Depends(get_session_local)):
    user = db.query(UserORM).filter(UserORM.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Пользователь не найден")

    db_house = HouseORM(
        title=house.title,
        description=house.description,
        price=house.price,
        recommended_price=house.recommended_price,
        agent_image=house.agent_image,
        agent_name=house.agent_name,
        photos=house.photos,  # Сохраняем список строк
        available_dates=house.available_dates,  # Сохраняем список строк
        user_id=user_id
    )

    db.add(db_house)
    db.commit()
    db.refresh(db_house)

    return HouseResponse(
        id=db_house.id,
        title=db_house.title,
        views=db_house.views,
        description=db_house.description,
        price=db_house.price,
        recommended_price=db_house.recommended_price,
        agent_image=db_house.agent_image,
        agent_name=db_house.agent_name,
        photos=db_house.photos,  # Возвращаем список строк
        available_dates=db_house.available_dates  # Возвращаем список строк
    )

@app.get("/users/me/houses", response_model=List[UserAdsResponse])
def read_user_houses(db: Session = Depends(get_session_local), current_user: UserORM = Depends(get_current_user)):
    try:
        houses = db.query(HouseORM).filter(HouseORM.user_id == current_user.id).all()
        return [
            UserAdsResponse(
                id=house.id,
                title=house.title,
                description=house.description,
                price=house.price,
                photos=house.photos,
                available_dates=house.available_dates
            ) for house in houses
        ]
    except Exception as e:
        logger.error(f"Error reading user houses: {e}")
        raise HTTPException(status_code=500, detail="Error reading user houses")

@app.get("/houses/{house_id}", response_model=HouseResponse)
def get_house(house_id: int, db: Session = Depends(get_session_local)):
    # Fetch the house from the database using the house_id
    house = db.query(HouseORM).filter(HouseORM.id == house_id).first()
    if not house:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Объявление не найдено")

    # Return the house details
    return HouseResponse(
        id=house.id,
        title=house.title,
        views=house.views,
        description=house.description,
        price=house.price,
        recommended_price=house.recommended_price,
        agent_image=house.agent_image,
        agent_name=house.agent_name,
        photos=house.photos,
        available_dates=house.available_dates
    )

@app.delete("/clear_all")
def clear_all_data(db: Session = Depends(get_session_local)):
    try:
        db.execute(text("TRUNCATE TABLE messages, notifications, user_ratings, cart, favorites, houses, users CASCADE"))
        db.commit()
        return {"message": "Все данные успешно удалены"}
    except Exception as e:
        logger.error(f"Ошибка при удалении данных: {e}")
        raise HTTPException(status_code=500, detail="Ошибка при удалении данных")

@app.post("/houses/{house_id}/favorite")
def add_house_to_favorites(
    house_id: int,
    db: Session = Depends(get_session_local),
    current_user: UserORM = Depends(get_current_user)
):
    # Check if the house exists
    house = db.query(HouseORM).filter(HouseORM.id == house_id).first()
    if not house:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Объявление не найдено")

    # Check if the house is already in the user's favorites
    favorite = db.query(FavoriteORM).filter(FavoriteORM.user_id == current_user.id, FavoriteORM.house_id == house_id).first()
    if favorite:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Объявление уже добавлено в избранное")

    # Add the house to the user's favorites
    new_favorite = FavoriteORM(user_id=current_user.id, house_id=house_id)
    db.add(new_favorite)
    db.commit()
    db.refresh(new_favorite)

    return {"message": "Объявление добавлено в избранное"}

@app.post("/houses/{house_id}/cart")
def add_house_to_cart(
    house_id: int,
    db: Session = Depends(get_session_local),
    current_user: UserORM = Depends(get_current_user)
):
    # Check if the house exists
    house = db.query(HouseORM).filter(HouseORM.id == house_id).first()
    if not house:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Объявление не найдено")

    # Check if the house is already in the user's cart
    cart_item = db.query(CartORM).filter(CartORM.user_id == current_user.id, CartORM.house_id == house_id).first()
    if cart_item:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Объявление уже добавлено в корзину")

    # Add the house to the user's cart
    new_cart_item = CartORM(user_id=current_user.id, house_id=house_id)
    db.add(new_cart_item)
    db.commit()
    db.refresh(new_cart_item)

    return {"message": "Объявление добавлено в корзину"}

@app.get("/api/houses", response_model=List[HouseResponse])
def get_all_houses(db: Session = Depends(get_session_local)):
    try:
        # Fetch all houses from the database
        houses = db.query(HouseORM).all()

        # Return the list of houses
        return [
            HouseResponse(
                id=house.id,
                title=house.title,
                views=house.views,
                description=house.description,
                price=house.price,
                recommended_price=house.recommended_price,
                agent_image=house.agent_image,
                agent_name=house.agent_name,
                photos=house.photos,
                available_dates=house.available_dates
            ) for house in houses
        ]
    except Exception as e:
        logger.error(f"Error fetching houses: {e}")
        raise HTTPException(status_code=500, detail="Error fetching houses")

@app.get("/api/houses/search", response_model=List[HouseResponse])
def search_houses(
    facade_type: Optional[str] = Query(None),
    floor: Optional[int] = Query(None),
    total_floors: Optional[int] = Query(None),
    rooms: Optional[int] = Query(None),
    area: Optional[float] = Query(None),
    kitchen_area: Optional[float] = Query(None),
    price_min: Optional[float] = Query(None),
    price_max: Optional[float] = Query(None),
    region: Optional[str] = Query(None),
    db: Session = Depends(get_session_local)
):
    try:
        # Create the base query
        query = db.query(HouseORM)

        # Apply filters
        if facade_type is not None:
            query = query.filter(HouseORM.facade_type == facade_type)
        if floor is not None:
            query = query.filter(HouseORM.floor == floor)
        if total_floors is not None:
            query = query.filter(HouseORM.total_floors == total_floors)
        if rooms is not None:
            query = query.filter(HouseORM.rooms == rooms)
        if area is not None:
            query = query.filter(HouseORM.area == area)
        if kitchen_area is not None:
            query = query.filter(HouseORM.kitchen_area == kitchen_area)
        if price_min is not None:
            query = query.filter(HouseORM.price >= price_min)
        if price_max is not None:
            query = query.filter(HouseORM.price <= price_max)
        if region is not None:
            query = query.filter(HouseORM.region == region)

        # Fetch the filtered houses from the database
        houses = query.all()

        # Return the list of houses
        return [
            HouseResponse(
                id=house.id,
                title=house.title,
                views=house.views,
                description=house.description,
                price=house.price,
                recommended_price=house.recommended_price,
                agent_image=house.agent_image,
                agent_name=house.agent_name,
                photos=house.photos,
                available_dates=house.available_dates
            ) for house in houses
        ]
    except Exception as e:
        logger.error(f"Error searching houses: {e}")
        raise HTTPException(status_code=500, detail="Error searching houses")

@app.get("/api/users", response_model=List[UserProfileResponse])
def get_all_users(db: Session = Depends(get_session_local)):
    try:
        # Fetch all users from the database
        users = db.query(UserORM).all()

        # Return the list of users
        return [
            UserProfileResponse(
                id=user.id,
                phone=user.phone,
                email=user.email,
                name=user.name
            ) for user in users
        ]
    except Exception as e:
        logger.error(f"Error fetching users: {e}")
        raise HTTPException(status_code=500, detail="Error fetching users")

@app.get("/api/messages/{user_id}", response_model=List[MessageResponse])
def get_user_messages(user_id: int, db: Session = Depends(get_session_local)):
    try:
        # Fetch messages sent to or received by the specified user
        messages = db.query(MessageORM).filter(
            (MessageORM.sender_id == user_id) | (MessageORM.recipient_id == user_id)
        ).all()

        # Return the list of messages
        return [
            MessageResponse(
                id=message.id,
                sender_id=message.sender_id,
                recipient_id=message.recipient_id,
                content=message.content,
                timestamp=message.timestamp,
                dialogue_name=message.dialogue_name
            ) for message in messages
        ]
    except Exception as e:
        logger.error(f"Error fetching messages for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching messages")

@app.post("/api/messages/", response_model=MessageResponse)
def send_message(message: MessageCreateRequest, db: Session = Depends(get_session_local)):
    try:
        # Check if the sender and recipient exist
        sender = db.query(UserORM).filter(UserORM.id == message.sender_id).first()
        recipient = db.query(UserORM).filter(UserORM.id == message.recipient_id).first()
        if not sender or not recipient:
            raise HTTPException(status_code=404, detail="Sender or recipient not found")

        # Create a new message
        new_message = MessageORM(
            sender_id=message.sender_id,
            recipient_id=message.recipient_id,
            content=message.content,
            dialogue_name=message.dialogue_name
        )

        # Add the new message to the database
        db.add(new_message)
        db.commit()
        db.refresh(new_message)

        # Return the created message
        return MessageResponse(
            id=new_message.id,
            sender_id=new_message.sender_id,
            recipient_id=new_message.recipient_id,
            content=new_message.content,
            timestamp=new_message.timestamp,
            dialogue_name=new_message.dialogue_name
        )
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail="Error sending message")

@app.get("/user/{user_id}", response_model=UserProfileResponse)
def get_user(user_id: int = Path(..., title="The ID of the user to get"), db: Session = Depends(get_session_local)):
    try:
        # Fetch the user from the database
        user = db.query(UserORM).filter(UserORM.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Return the user information
        return UserProfileResponse(
            id=user.id,
            phone=user.phone,
            email=user.email,
            name=user.name
        )
    except Exception as e:
        logger.error(f"Error fetching user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching user information")


@app.get("/user/{user_id}/ads", response_model=List[UserAdsResponse])
def get_user_ads(user_id: int = Path(..., title="The ID of the user to get ads for"),
                 db: Session = Depends(get_session_local)):
    try:
        # Fetch the user from the database
        user = db.query(UserORM).filter(UserORM.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Fetch the user's ads from the database
        houses = db.query(HouseORM).filter(HouseORM.user_id == user_id).all()
        return [
            UserAdsResponse(
                id=house.id,
                title=house.title,
                description=house.description,
                price=house.price,
                photos=house.photos,
                available_dates=house.available_dates
            ) for house in houses
        ]
    except Exception as e:
        logger.error(f"Error fetching ads for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching user ads")

@app.get("/users/{user_id}", response_model=UserProfileResponse)
def get_user(user_id: int = Path(..., title="The ID of the user to get"), db: Session = Depends(get_session_local)):
    try:
        # Fetch the user from the database
        user = db.query(UserORM).filter(UserORM.id == user_id).first()
        if not user:
            raise HTTPException(status_code=404, detail="User not found")

        # Return the user information
        return UserProfileResponse(
            id=user.id,
            phone=user.phone,
            email=user.email,
            name=user.name
        )
    except Exception as e:
        logger.error(f"Error fetching user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error fetching user information")

@app.delete("/users/{user_id}", response_model=SuccessMessageResponse)
def delete_user(
    user_id: int = Path(..., title="The ID of the user to delete"),
    db: Session = Depends(get_session_local)
):
    logger.info(f"Начало запроса path=/users/{user_id} method=DELETE")
    try:
        # Поиск пользователя в базе данных
        user = db.query(UserORM).filter(UserORM.id == user_id).first()
        if not user:
            logger.error(f"User with id {user_id} not found")
            raise HTTPException(status_code=404, detail="User not found")

        # Удаление пользователя и связанных записей
        db.delete(user)
        db.commit()

        # Возврат сообщения об успешном удалении
        response = SuccessMessageResponse(message="User successfully deleted")
        logger.info(f"Ответ готов status_code=200")
        return response

    except HTTPException as e:
        # Обработка HTTP исключений
        logger.error(f"Error deleting user {user_id}: {e.detail}")
        raise e
    except IntegrityError as e:
        logger.error(f"ForeignKey violation error deleting user {user_id}: {e}")
        raise HTTPException(status_code=409, detail="Cannot delete user as it is referenced in another table")
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error deleting user")

@app.get("/users/{user_id}/ads", response_model=UserAdsResponse)
def get_user_ads(
    user_id: int = Path(..., title="The ID of the user to get ads for"),
    db: Session = Depends(get_session_local)
):
    logger.info(f"Начало запроса path=/users/{user_id}/ads method=GET")
    try:
        # Поиск пользователя в базе данных
        user = db.query(UserORM).filter(UserORM.id == user_id).first()
        if not user:
            logger.error(f"User with id {user_id} not found")
            raise HTTPException(status_code=404, detail="User not found")

        # Извлечение объявлений пользователя
        ads = db.query(HouseORM).filter(HouseORM.user_id == user_id).all()
        ad_responses = [AdResponse(id=ad.id, title=ad.title, description=ad.description, price=ad.price) for ad in ads]

        # Формирование ответа
        response = UserAdsResponse(user_id=user_id, ads=ad_responses)
        logger.info(f"Ответ готов status_code=200")
        return response

    except HTTPException as e:
        logger.error(f"Error getting ads for user {user_id}: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error getting ads for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail="Error getting user ads")

@app.post("/users/{user_id}/rate", response_model=SuccessMessageResponse, status_code=status.HTTP_200_OK)
async def rate_user(
    user_id: int = Path(..., description="The ID of the user to rate"),
    rating_request: RatingRequest = Body(..., description="The user rating data"),
    current_user: UserORM = Depends(get_current_user),
    db: Session = Depends(get_session_local)
):
    user_to_rate = db.query(UserORM).filter(UserORM.id == user_id).first()
    if not user_to_rate:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    # Create a new UserRatingORM object
    new_rating = UserRatingORM(
        user_id=user_id,
        rating=rating_request.rating
    )
    db.add(new_rating)
    db.commit()
    db.refresh(new_rating)

    return SuccessMessageResponse(message="User rated successfully")

@app.get("/user/{user_id}/favourites", response_model=List[FavouritesResponse], status_code=status.HTTP_200_OK)
async def get_user_favourites(
    user_id: int = Path(..., description="The ID of the user to get favourites for"),
    current_user: UserORM = Depends(get_current_user),
    db: Session = Depends(get_session_local)
):
    user = db.query(UserORM).filter(UserORM.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    favourites = db.query(FavoriteORM).filter(FavoriteORM.user_id == user_id).all()
    response = []
    for favourite in favourites:
        house = favourite.house
        response.append(FavouritesResponse(
            id=house.id,
            title=house.title,
            description=house.description,
            price=house.price,
            photos=house.photos,
            available_dates=house.available_dates
        ))

    return response


@app.post("/user/{user_id}/favourites/add")
async def add_favourite(
        user_id: int,
        ad_id_request: AdIdRequest,
        db: Session = Depends(get_session_local),
        current_user: UserORM = Depends(get_current_user)
):
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to perform this action"
        )

    new_favorite = FavoriteORM(user_id=user_id, house_id=ad_id_request.ad_id)
    db.add(new_favorite)
    db.commit()
    db.refresh(new_favorite)
    return {"message": "Favorite added successfully"}


@app.post("/user/{user_id}/favourites/remove")
async def remove_favourite(
        user_id: int,
        ad_id_request: AdIdRequest,
        db: Session = Depends(get_session_local),
        current_user: UserORM = Depends(get_current_user)
):
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to perform this action"
        )

    favorite = db.query(FavoriteORM).filter_by(user_id=user_id, house_id=ad_id_request.ad_id).first()
    if not favorite:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Favorite not found"
        )

    db.delete(favorite)
    db.commit()
    return {"message": "Favorite removed successfully"}

@app.get("/user/{user_id}/notifications", response_model=List[NotificationResponse])
async def get_notifications(
    user_id: int,
    db: Session = Depends(get_session_local),
    current_user: UserORM = Depends(get_current_user)
):
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access these notifications"
        )

    user = db.query(UserORM).filter(UserORM.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    notifications = db.query(NotificationORM).filter(NotificationORM.user_id == user_id).all()
    return notifications


@app.post("/user/{user_id}/notifications/mark-as-read")
async def mark_notifications_as_read(
        user_id: int,
        db: Session = Depends(get_session_local),
        current_user: UserORM = Depends(get_current_user)
):
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to perform this action"
        )

    user = db.query(UserORM).filter(UserORM.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    notifications = db.query(NotificationORM).filter(NotificationORM.user_id == user_id).all()
    for notification in notifications:
        notification.read = True
    db.commit()

    return {"message": "All notifications marked as read"}


@app.post("/user/{user_id}/notifications/{notification_id}/mark-as-read")
async def mark_notification_as_read(
        user_id: int,
        notification_id: int,
        db: Session = Depends(get_session_local),
        current_user: UserORM = Depends(get_current_user)
):
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to perform this action"
        )

    user = db.query(UserORM).filter(UserORM.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    notification = db.query(NotificationORM).filter(NotificationORM.id == notification_id,
                                                    NotificationORM.user_id == user_id).first()
    if not notification:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Notification not found"
        )

    notification.read = True
    db.commit()

    return {"message": "Notification marked as read"}


@app.get("/user/{user_id}/dialogues", response_model=List[MessageResponse])
async def get_user_dialogues(
        user_id: int,
        db: Session = Depends(get_session_local),
        current_user: UserORM = Depends(get_current_user)
):
    if current_user.id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access these dialogues"
        )

    user = db.query(UserORM).filter(UserORM.id == user_id).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )

    dialogues = db.query(MessageORM).filter(
        (MessageORM.sender_id == user_id) | (MessageORM.recipient_id == user_id)
    ).all()

    # Group messages by dialogue name
    dialogues_grouped = {}
    for dialogue in dialogues:
        if dialogue.dialogue_name not in dialogues_grouped:
            dialogues_grouped[dialogue.dialogue_name] = []
        dialogues_grouped[dialogue.dialogue_name].append(dialogue)

    return [MessageResponse(
        id=dialogue.id,
        sender_id=dialogue.sender_id,
        recipient_id=dialogue.recipient_id,
        content=dialogue.content,
        timestamp=dialogue.timestamp,
        dialogue_name=dialogue.dialogue_name
    ) for dialogue_name, messages in dialogues_grouped.items() for dialogue in messages]

@app.get("/dialogue/{dialogue_id}/messages", response_model=List[MessageResponse])
async def get_dialogue_messages(
    dialogue_id: str,
    db: Session = Depends(get_session_local),
    current_user: UserORM = Depends(get_current_user)
):
    dialogue_messages = db.query(MessageORM).filter(MessageORM.dialogue_name == dialogue_id).all()

    if not dialogue_messages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dialogue not found"
        )

    # Ensure the user is part of the dialogue
    if not any(msg.sender_id == current_user.id or msg.recipient_id == current_user.id for msg in dialogue_messages):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access these messages"
        )

    return [MessageResponse(
        id=msg.id,
        sender_id=msg.sender_id,
        recipient_id=msg.recipient_id,
        content=msg.content,
        timestamp=msg.timestamp,
        dialogue_name=msg.dialogue_name
    ) for msg in dialogue_messages]

@app.post("/messages", response_model=MessageResponse)
async def send_message(
    message_data: MessageCreateRequest,
    db: Session = Depends(get_session_local),
    current_user: UserORM = Depends(get_current_user)
):
    # Ensure the dialogue exists and the user is a part of it
    dialogue_messages = db.query(MessageORM).filter(MessageORM.dialogue_name == message_data.dialogue_name).all()
    if not dialogue_messages:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Dialogue not found"
        )

    if not any(msg.sender_id == current_user.id or msg.recipient_id == current_user.id for msg in dialogue_messages):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to send messages in this dialogue"
        )

    # Create the new message
    new_message = MessageORM(
        sender_id=message_data.user_id,
        recipient_id=dialogue_messages[0].recipient_id if dialogue_messages[0].sender_id == message_data.user_id else dialogue_messages[0].sender_id,
        content=message_data.text,
        dialogue_name=message_data.dialogue_name,
    )
    db.add(new_message)
    db.commit()
    db.refresh(new_message)

    return MessageResponse(
        id=new_message.id,
        sender_id=new_message.sender_id,
        recipient_id=new_message.recipient_id,
        content=new_message.content,
        timestamp=new_message.timestamp,
        dialogue_name=new_message.dialogue_name
    )

@app.post("/user/{user_id}/notifications", response_model=NotificationResponse)
async def create_notification(
    user_id: int,
    notification_data: NotificationCreateRequest,
    db: Session = Depends(get_session_local),
    current_user: UserORM = Depends(get_current_user)
):
    # Проверка наличия пользователя
    user = db.query(UserORM).filter(UserORM.id == user_id).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Пользователь не найден")

    # Создание нового уведомления
    new_notification = NotificationORM(
        user_id=user_id,
        title=notification_data.title,
        text=notification_data.text,
        read=False
    )

    db.add(new_notification)
    db.commit()
    db.refresh(new_notification)

    return NotificationResponse(
        id=new_notification.id,
        title=new_notification.title,
        text=new_notification.text,
        read=new_notification.read
    )

@app.on_event("startup")
def startup_event():
  global model
  model = load_staff()

@app.post("/predict")
def predict_cost(params: PredictParams):
  model_prediction = model(params.dict())

  response = model_prediction

  return response
# Testing endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the FastAPI app"}

