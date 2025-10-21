"""데이터베이스 모델 (테이블) 정의"""

from sqlalchemy import Column, Integer, String, DateTime, Text
from sqlalchemy.sql import func
from src.core.database import Base


class MeetingRecord(Base):
    """회의록 저장 모델"""
    __tablename__ = "meeting_records"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # STT 결과
    transcript = Column(Text)
    
    # AI 분석 결과 (기존 5개)
    summary = Column(Text, nullable=True)
    meeting_notes = Column(Text, nullable=True)
    action_items = Column(Text, nullable=True)
    sentiment = Column(Text, nullable=True)
    follow_up_questions = Column(Text, nullable=True)
    
    # --- [충돌 병합된 신규 AI 분석 결과] ---
    keywords = Column(Text, nullable=True)
    speaker_summary = Column(Text, nullable=True) # ( summarize_by_speaker 결과 )
    
    # (Git 'e319...' 버전에서 추가된 컬럼들)
    topics = Column(Text, nullable=True)
    speaker_analysis = Column(Text, nullable=True) # ( analyze_by_speaker 결과 )
    meeting_type = Column(Text, nullable=True)
    # -------------------------------------

    def to_dict(self):
        """모델 객체를 딕셔너리로 변환 (API 응답용)"""
        return {c.name: getattr(self, c.name) for c in self.__table__.columns}