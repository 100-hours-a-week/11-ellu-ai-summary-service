from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_max_id():
    # 데이터베이스 URL 직접 설정
    DATABASE_URL = ""
    
    engine = create_engine(DATABASE_URL)
    
    try:
        with engine.begin() as connection:
            query = text("SELECT MAX(id) FROM user_io")
            result = connection.execute(query)
            max_id = result.fetchone()[0]
            
            if max_id is not None:
                logger.info(f"user_io 테이블의 최대 ID: {max_id}")
                return {"max_id": max_id}
            else:
                logger.info("user_io 테이블이 비어있습니다")
                return {"max_id": 0}
                
    except SQLAlchemyError as e:
        logger.error(f"user_io 테이블 조회 실패: {str(e)}")
        return {"error": str(e)}

if __name__ == "__main__":
    result = check_max_id()
    print(f"결과: {result}")