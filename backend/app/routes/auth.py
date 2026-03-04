from fastapi import APIRouter, HTTPException, Depends
from bson import ObjectId
from app.schemas.auth_schema import RegisterSchema, LoginSchema, TokenResponse
from app.database.mongodb import get_database
from app.models.user import User
from app.core.security import hash_password, verify_password, create_access_token
from app.dependencies.auth_dependency import get_current_user
from fastapi.security import OAuth2PasswordRequestForm

router = APIRouter(prefix="/auth", tags=["Auth"])

@router.post("/register")
async def register_user(data: RegisterSchema):
    db = get_database()

    existing = await db.users.find_one(
        {"email": data.email}
    )

    if existing:
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    
    user_dict = User(
        email=data.email,
        hashed_password=hash_password(data.password)
    )

    user = user_dict.model_dump(
        by_alias=True,
        exclude={"id"}
    )

    result = await db.users.insert_one(user)

    return {"message": "User registered"}

@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends()
):
    db = get_database()

    user = await db.users.find_one(
        {"email": form_data.username}
    )

    if not user:
        raise HTTPException(401, "Invalid credentials")
    
    if not verify_password(
        form_data.password,
        user["hashed_password"]
    ):
        raise HTTPException(401, "Invalid credentials")
    
    token = create_access_token(str(user["_id"]))

    return {
        "access_token": token
    }

@router.get("/me")
async def get_me(user=Depends(get_current_user)):
    return {
        "email": user["email"],
        "id": str(user["_id"])
    }