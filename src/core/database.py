"""데이터베이스 세션 설정 (MariaDB용)"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# .env 파일에서 DB 설정 로드
# (server.py가 아닌 여기서도 load_dotenv를 호출해야 DB 초기화 시 값을 읽을 수 있습니다)
load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", 3306)
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

if not all([DB_USER, DB_PASSWORD, DB_NAME]):
    raise ValueError("DB_USER, DB_PASSWORD, DB_NAME 환경변수를 .env 파일에 설정해주세요.")

# MariaDB/MySQL connection string (mysql+pymysql 드라이버 사용)
SQLALCHEMY_DATABASE_URL = (
    f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"
)

engine = create_engine(
    SQLALCHEMY_DATABASE_URL
    # MariaDB/MySQL은 connect_args={"check_same_thread": False}가 필요 없습니다.
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def init_db():
    """DB 테이블 생성"""
    # models.py에서 Base를 상속받은 모든 모델(테이블)을 생성
    import src.core.models
    Base.metadata.create_all(bind=engine)