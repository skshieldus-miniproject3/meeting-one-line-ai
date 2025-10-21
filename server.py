"""CLOVA Speech REST API 서버"""

import os
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from fastapi import (
    FastAPI, UploadFile, File, HTTPException, BackgroundTasks,
    Depends, Form
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

# --- [DB 연동 추가] ---
from sqlalchemy.orm import Session
from src.core import database, models
# -----------------------

# Add src to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.core.stt_client import ClovaSpeechClient, ClovaSpeechError
from src.core.formatter import (
    format_segments_to_transcript,
    format_segments_to_detailed_transcript,
    extract_speaker_statistics,
    generate_meeting_summary_header
)
from src.core.ai_analyzer import ReportGenerator, ReportGeneratorError

# .env 로드
load_dotenv()

# FastAPI 앱 생성
app = FastAPI(
    title="CLOVA Speech STT API",
    description="NAVER Cloud Platform CLOVA Speech API 서버 (DB, AI 통합)",
    version="1.2.0" # 버전 업데이트
)

# 글로벌 설정
INVOKE_URL = os.getenv('CLOVA_SPEECH_INVOKE_URL')
SECRET_KEY = os.getenv('CLOVA_SPEECH_SECRET_KEY')

if not INVOKE_URL or not SECRET_KEY:
    raise ValueError("CLOVA_SPEECH_INVOKE_URL과 CLOVA_SPEECH_SECRET_KEY를 .env에 설정해주세요")

# 클라이언트 인스턴스
client = ClovaSpeechClient(INVOKE_URL, SECRET_KEY)

# AI 보고서 생성기 (OpenAI 키가 있는 경우에만)
try:
    report_generator = ReportGenerator()
except Exception:
    report_generator = None

# 작업 상태 저장소 (실제 환경에서는 Redis 등 사용)
job_store: Dict[str, Dict[str, Any]] = {}


# --- [DB 연동 추가] ---
def get_db():
    """DB 세션 의존성"""
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()
# -----------------------


class URLRequest(BaseModel):
    """URL 방식 요청 모델"""
    url: HttpUrl
    language: str = "ko-KR"
    completion: str = "sync"
    word_alignment: bool = True
    full_text: bool = True
    callback: Optional[str] = None
    userdata: Optional[str] = None
    enable_diarization: bool = True
    enable_noise_filtering: bool = True
    enable_sed: bool = False
    speaker_count_min: int = 2
    speaker_count_max: int = 10


class MeetingRequest(BaseModel):
    """회의 전사 요청 모델"""
    url: HttpUrl
    language: str = "ko-KR"
    include_ai_summary: bool = True
    meeting_title: Optional[str] = None
    speaker_count_min: int = 2
    speaker_count_max: int = 10


class TranscriptFormatRequest(BaseModel):
    """대화록 포맷팅 요청 모델"""
    segments: list
    format_type: str = "basic"  # basic, detailed, with_speakers
    include_timestamps: bool = False
    include_confidence: bool = False
    speaker_names: Optional[Dict[str, str]] = None


class SummaryRequest(BaseModel):
    """요약 요청 모델"""
    transcript: str
    # [수정] Git 'e319...' 버전의 신규 타입들 추가
    summary_type: str = "summary"  # summary, meeting_notes, action_items, sentiment, follow_up, keywords, topics, by_speaker, meeting_type, speaker_summary


class JobResponse(BaseModel):
    """작업 응답 모델"""
    job_id: str
    status: str
    created_at: str
    message: Optional[str] = None


@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "CLOVA Speech STT API",
        "version": "1.2.0", # 버전 업데이트
        "endpoints": [
            "/stt/url",
            "/stt/file",
            "/stt/status/{job_id}",
            "/meeting/transcribe",
            "/transcript/format",
            "/transcript/summarize",
            "/transcript/statistics",
            "/upload_and_analyze", # [신규]
            "/history"            # [신규]
        ]
    }


@app.get("/health")
async def health():
    """헬스체크"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# --- [기존 엔드포인트 (변경 없음)] ---

@app.post("/stt/url")
async def transcribe_url(request: URLRequest, background_tasks: BackgroundTasks):
    """URL 방식 음성 인식"""
    try:
        # 옵션 구성
        options = {
            'language': request.language,
            'completion': request.completion,
            'wordAlignment': request.word_alignment,
            'fullText': request.full_text,
            'enable_diarization': request.enable_diarization,
            'enable_noise_filtering': request.enable_noise_filtering,
            'enable_sed': request.enable_sed
        }

        if request.enable_diarization:
            options['diarization'] = {
                'enable': True,
                'speakerCountMin': request.speaker_count_min,
                'speakerCountMax': request.speaker_count_max
            }

        if request.callback:
            options['callback'] = request.callback
        if request.userdata:
            options['userdata'] = request.userdata

        # 요청 실행
        # [참고] stt_client.py가 'enko'를 기본으로 사용하도록 변경되었음
        result = client.request_by_url(str(request.url), **options)

        if request.completion == 'sync':
            # 동기 모드: 바로 결과 반환
            return JSONResponse(content=result)
        else:
            # 비동기 모드: 작업 ID 반환
            job_id = result if isinstance(result, str) else str(uuid.uuid4())

            # 작업 정보 저장
            job_store[job_id] = {
                'status': 'processing',
                'created_at': datetime.now().isoformat(),
                'type': 'url',
                'url': str(request.url),
                'options': options,
                'result': None
            }

            # 백그라운드에서 결과 폴링
            background_tasks.add_task(poll_result, job_id)

            return JobResponse(
                job_id=job_id,
                status='processing',
                created_at=job_store[job_id]['created_at']
            )

    except ClovaSpeechError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {e}")


@app.post("/stt/file")
async def transcribe_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = Form("enko"), # [수정] 기본값을 'enko'로 변경 (Git 'e319...' 버전 반영)
    completion: str = Form("sync"),
    word_alignment: bool = Form(True),
    full_text: bool = Form(True),
    callback: Optional[str] = Form(None),
    userdata: Optional[str] = Form(None),
    enable_diarization: bool = Form(True),
    enable_noise_filtering: bool = Form(True),
    enable_sed: bool = Form(False),
    speaker_count_min: int = Form(2),
    speaker_count_max: int = Form(10)
):
    """파일 업로드 방식 음성 인식 (기존 기능)"""
    try:
        # 임시 파일 저장
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)

        temp_file = temp_dir / f"{uuid.uuid4()}_{file.filename}"

        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)

        # 옵션 구성
        options = {
            'language': language,
            'completion': completion,
            'wordAlignment': word_alignment,
            'fullText': full_text,
            'enable_diarization': enable_diarization,
            'enable_noise_filtering': enable_noise_filtering,
            'enable_sed': enable_sed
        }

        if enable_diarization:
            options['diarization'] = {
                'enable': True,
                'speakerCountMin': speaker_count_min,
                'speakerCountMax': speaker_count_max
            }

        if callback:
            options['callback'] = callback
        if userdata:
            options['userdata'] = userdata

        # 요청 실행
        # [참고] stt_client.py가 'enko'를 기본으로 사용하도록 변경되었음
        result = client.request_by_file(temp_file, **options)

        if completion == 'sync':
            # 동기 모드: 임시 파일 삭제 후 결과 반환
            temp_file.unlink(missing_ok=True)
            return JSONResponse(content=result)
        else:
            # 비동기 모드: 작업 ID 반환
            job_id = result if isinstance(result, str) else str(uuid.uuid4())

            # 작업 정보 저장
            job_store[job_id] = {
                'status': 'processing',
                'created_at': datetime.now().isoformat(),
                'type': 'file',
                'filename': file.filename,
                'temp_file': str(temp_file),
                'options': options,
                'result': None
            }

            # 백그라운드에서 결과 폴링
            background_tasks.add_task(poll_result, job_id)

            return JobResponse(
                job_id=job_id,
                status='processing',
                created_at=job_store[job_id]['created_at']
            )

    except ClovaSpeechError as e:
        # 임시 파일 정리
        if 'temp_file' in locals():
            temp_file.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # 임시 파일 정리
        if 'temp_file' in locals():
            temp_file.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {e}")


@app.get("/stt/status/{job_id}")
async def get_job_status(job_id: str):
    """작업 상태 조회"""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")

    job_info = job_store[job_id]

    if job_info['status'] == 'completed' and job_info['result']:
        return JSONResponse(content=job_info['result'])
    elif job_info['status'] == 'failed':
        return JobResponse(
            job_id=job_id,
            status='failed',
            created_at=job_info['created_at'],
            message=job_info.get('error', '알 수 없는 오류')
        )
    else:
        return JobResponse(
            job_id=job_id,
            status=job_info['status'],
            created_at=job_info['created_at']
        )


async def poll_result(job_id: str):
    """백그라운드에서 비동기 작업 결과 폴링"""
    try:
        # 작업 완료까지 대기
        result = client.wait_for_completion(job_id, poll_interval=2, max_wait=300)

        # 결과 저장
        job_store[job_id]['status'] = 'completed'
        job_store[job_id]['result'] = result

        # 임시 파일 정리 (파일 업로드 방식인 경우)
        if job_store[job_id]['type'] == 'file':
            temp_file_path = job_store[job_id].get('temp_file')
            if temp_file_path:
                Path(temp_file_path).unlink(missing_ok=True)

    except Exception as e:
        # 오류 저장
        job_store[job_id]['status'] = 'failed'
        job_store[job_id]['error'] = str(e)

        # 임시 파일 정리 (파일 업로드 방식인 경우)
        if job_store[job_id]['type'] == 'file':
            temp_file_path = job_store[job_id].get('temp_file')
            if temp_file_path:
                Path(temp_file_path).unlink(missing_ok=True)


@app.on_event("startup")
async def startup():
    """서버 시작 시 실행"""
    # [수정] DB 초기화
    try:
        database.init_db()
        print("DB 테이블이 초기화/확인되었습니다.")
    except Exception as e:
        print(f"DB 초기화 실패: {e}")
        print("DB 접속 정보를 .env 파일에 올바르게 설정했는지, DB가 실행 중인지 확인하세요.")

    print("CLOVA Speech STT API server started")
    print(f"Invoke URL: {INVOKE_URL}")


@app.on_event("shutdown")
async def shutdown():
    """서버 종료 시 정리"""
    print("Server shutting down...")

    # 임시 파일 정리
    temp_dir = Path("temp")
    if temp_dir.exists():
        for temp_file in temp_dir.glob("*"):
            temp_file.unlink(missing_ok=True)
        try:
            temp_dir.rmdir()
        except OSError:
            pass # 디렉토리가 비어있지 않으면 무시

    print("Cleanup completed")


@app.post("/meeting/transcribe")
async def transcribe_meeting(request: MeetingRequest, background_tasks: BackgroundTasks):
    """회의 오디오 전사 및 AI 요약 (URL 방식, 기존 기능)"""
    try:
        # STT 옵션 구성 (회의에 최적화)
        options = {
            'language': request.language,
            'completion': 'sync',
            'wordAlignment': True,
            'fullText': True,
            'enable_diarization': True,
            'enable_noise_filtering': True,
            'diarization': {
                'enable': True,
                'speakerCountMin': request.speaker_count_min,
                'speakerCountMax': request.speaker_count_max
            }
        }

        # STT 실행
        stt_result = client.request_by_url(str(request.url), **options)

        if 'segments' not in stt_result:
            raise HTTPException(status_code=400, detail="STT 결과에 segments가 없습니다")

        segments = stt_result['segments']

        # 대화록 포맷팅
        transcript = format_segments_to_transcript(segments)
        detailed_transcript = format_segments_to_detailed_transcript(
            segments, include_timestamps=True, include_confidence=True
        )

        # 화자 통계
        speaker_stats = extract_speaker_statistics(segments)

        # 응답 구성
        response_data = {
            'stt_result': stt_result,
            'transcript': transcript,
            'detailed_transcript': detailed_transcript,
            'speaker_statistics': speaker_stats,
            'meeting_header': generate_meeting_summary_header(segments, request.meeting_title)
        }

        # AI 요약 추가 (옵션)
        if request.include_ai_summary and report_generator:
            try:
                # [수정] Git 'e319...' 버전의 신규 기능들 호출
                ai_reports = {
                    'summary': report_generator.summarize(transcript),
                    'meeting_notes': report_generator.generate_meeting_notes(transcript),
                    'action_items': report_generator.generate_action_items(transcript),
                    'sentiment': report_generator.analyze_sentiment(transcript),
                    'follow_up_questions': report_generator.generate_follow_up_questions(transcript),
                    
                    'keywords': report_generator.extract_keywords(transcript),
                    'topics': report_generator.classify_topics(transcript),
                    'by_speaker': report_generator.analyze_by_speaker(transcript),
                    'meeting_type': report_generator.classify_meeting_type(transcript),
                    'speaker_summary': report_generator.summarize_by_speaker(transcript)
                }
                response_data['ai_reports'] = ai_reports
            except Exception as e:
                response_data['ai_reports_error'] = f"AI 요약 생성 실패: {str(e)}"

        return JSONResponse(content=response_data)

    except ClovaSpeechError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {e}")


@app.post("/transcript/format")
async def format_transcript(request: TranscriptFormatRequest):
    """STT segments를 다양한 형태의 대화록으로 포맷팅 (기존 기능)"""
    try:
        segments = request.segments

        if request.format_type == "basic":
            result = format_segments_to_transcript(segments)
        elif request.format_type == "detailed":
            result = format_segments_to_detailed_transcript(
                segments,
                include_timestamps=request.include_timestamps,
                include_confidence=request.include_confidence
            )
        elif request.format_type == "with_speakers" and request.speaker_names:
            # src.core.formatter에서 임포트 (경로 수정)
            from src.core.formatter import format_transcript_with_speakers
            result = format_transcript_with_speakers(segments, request.speaker_names)
        else:
            raise HTTPException(status_code=400, detail="지원하지 않는 format_type입니다")

        return JSONResponse(content={"formatted_transcript": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"포맷팅 오류: {e}")


@app.post("/transcript/summarize")
async def summarize_transcript(request: SummaryRequest):
    """대화록 AI 요약 및 분석 (기존 기능 + 신규 분석 타입)"""
    if not report_generator:
        raise HTTPException(
            status_code=503,
            detail="AI 요약 기능을 사용할 수 없습니다. OPENAI_API_KEY를 설정해주세요."
        )

    try:
        transcript = request.transcript
        summary_type = request.summary_type

        if summary_type == "summary":
            result = report_generator.summarize(transcript)
        elif summary_type == "meeting_notes":
            result = report_generator.generate_meeting_notes(transcript)
        elif summary_type == "action_items":
            result = report_generator.generate_action_items(transcript)
        elif summary_type == "sentiment":
            result = report_generator.analyze_sentiment(transcript)
        elif summary_type == "follow_up":
            result = report_generator.generate_follow_up_questions(transcript)
        
        # --- [신규 분석 타입 추가 (Git 'e319...' + Local Merge)] ---
        elif summary_type == "keywords":
            result = report_generator.extract_keywords(transcript)
        elif summary_type == "topics":
            result = report_generator.classify_topics(transcript)
        elif summary_type == "by_speaker":
            result = report_generator.analyze_by_speaker(transcript)
        elif summary_type == "meeting_type":
            result = report_generator.classify_meeting_type(transcript)
        elif summary_type == "speaker_summary":
            result = report_generator.summarize_by_speaker(transcript)
        # ----------------------------------------------------
        
        else:
            raise HTTPException(status_code=400, detail="지원하지 않는 summary_type입니다")

        return JSONResponse(content={
            "summary_type": summary_type,
            "result": result
        })

    except ReportGeneratorError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"요약 생성 오류: {e}")


@app.post("/transcript/statistics")
async def get_transcript_statistics(segments: list):
    """대화록 통계 정보 추출 (기존 기능)"""
    try:
        stats = extract_speaker_statistics(segments)
        header = generate_meeting_summary_header(segments)

        return JSONResponse(content={
            "speaker_statistics": stats,
            "meeting_header": header
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 추출 오류: {e}")


# --- [신규 기능 엔드포인트] ---

@app.post("/upload_and_analyze")
async def upload_and_analyze(
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    language: str = Form("enko"), # [수정] 기본값을 'enko'로 변경 (Git 'e319...' 버전 반영)
    speaker_count_min: int = Form(2),
    speaker_count_max: int = Form(10)
):
    """
    [신규] 파일 업로드 + STT + 전체 AI 분석 + DB 저장 통합 엔드포인트
    
    프론트엔드에서 이 엔드포인트 하나만 호출하면 됩니다.
    """
    if not report_generator:
        raise HTTPException(
            status_code=503,
            detail="AI 요약 기능을 사용할 수 없습니다. OPENAI_API_KEY를 설정해주세요."
        )

    # 1. 파일 임시 저장
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    temp_file = temp_dir / f"{uuid.uuid4()}_{file.filename}"

    try:
        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)

        # 2. STT 옵션 구성 (회의용 최적화, 동기 모드 고정)
        options = {
            'language': language,
            'completion': 'sync', # 동기(sync) 모드로 고정
            'wordAlignment': True,
            'fullText': True,
            'enable_diarization': True,
            'enable_noise_filtering': True,
            'diarization': {
                'enable': True,
                'speakerCountMin': speaker_count_min,
                'speakerCountMax': speaker_count_max
            }
        }

        # 3. STT 실행
        stt_result = client.request_by_file(temp_file, **options)

    except ClovaSpeechError as e:
        raise HTTPException(status_code=400, detail=f"STT 오류: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 처리 또는 STT 오류: {e}")
    finally:
        # 임시 파일 즉시 삭제
        temp_file.unlink(missing_ok=True)

    if 'segments' not in stt_result:
        raise HTTPException(status_code=400, detail="STT 결과에 segments가 없습니다")

    segments = stt_result.get('segments', [])
    transcript = format_segments_to_transcript(segments)
    speaker_stats = extract_speaker_statistics(segments)
    
    ai_reports = None
    ai_reports_error = None

    # 4. AI 전체 분석 실행 (병합된 모든 기능 호출)
    try:
        ai_reports = {
            # 기본 5개
            'summary': report_generator.summarize(transcript),
            'meeting_notes': report_generator.generate_meeting_notes(transcript),
            'action_items': report_generator.generate_action_items(transcript),
            'sentiment': report_generator.analyze_sentiment(transcript),
            'follow_up_questions': report_generator.generate_follow_up_questions(transcript),
            
            # 신규 6개 (병합됨)
            'keywords': report_generator.extract_keywords(transcript),
            'topics': report_generator.classify_topics(transcript),
            'by_speaker': report_generator.analyze_by_speaker(transcript), # Git 'e319...' 버전의 상세 분석
            'meeting_type': report_generator.classify_meeting_type(transcript),
            'speaker_summary': report_generator.summarize_by_speaker(transcript), # Local(DB) 버전의 간단 요약
        }
    except Exception as e:
        # AI 분석에 실패하더라도 STT 결과는 반환 (부분적 성공)
        ai_reports_error = f"AI 요약 생성 실패: {str(e)}"

    # 5. DB에 저장
    db_id = None
    db_created_at = None
    try:
        new_record = models.MeetingRecord(
            filename=file.filename,
            transcript=transcript,
            
            # 기본 5개 저장
            summary=ai_reports.get('summary') if ai_reports else None,
            meeting_notes=ai_reports.get('meeting_notes') if ai_reports else None,
            action_items=ai_reports.get('action_items') if ai_reports else None,
            sentiment=ai_reports.get('sentiment') if ai_reports else None,
            follow_up_questions=ai_reports.get('follow_up_questions') if ai_reports else None,
            
            # [수정] 신규 5개도 DB에 저장
            keywords=ai_reports.get('keywords') if ai_reports else None,
            speaker_summary=ai_reports.get('speaker_summary') if ai_reports else None,
            topics=ai_reports.get('topics') if ai_reports else None,
            speaker_analysis=ai_reports.get('by_speaker') if ai_reports else None, # by_speaker 결과를 speaker_analysis 컬럼에 저장
            meeting_type=ai_reports.get('meeting_type') if ai_reports else None,
        )
        db.add(new_record)
        db.commit()
        db.refresh(new_record)
        db_id = new_record.id
        db_created_at = new_record.created_at
    except Exception as e:
        db.rollback()
        # DB 저장은 실패하더라도, 분석 결과는 사용자에게 반환
        ai_reports_error = f"{ai_reports_error or ''} | DB 저장 오류: {e}"


    # 6. 통합 결과 반환
    response_data = {
        "db_id": db_id, # DB 저장 성공 시 ID, 실패 시 None
        "filename": file.filename,
        "created_at": db_created_at, # DB 저장 성공 시 시간, 실패 시 None
        "transcript": transcript,
        "speaker_statistics": speaker_stats,
        "ai_reports": ai_reports,
        "ai_reports_error": ai_reports_error,
        "stt_raw_result": stt_result # 원본 STT 결과도 포함
    }

    return JSONResponse(content=response_data)


@app.get("/history", response_model=List[Dict[str, Any]])
async def get_history(db: Session = Depends(get_db), skip: int = 0, limit: int = 100):
    """
    [신규] DB에 저장된 회의록 내역 조회
    """
    try:
        records = db.query(models.MeetingRecord).order_by(
            models.MeetingRecord.created_at.desc()
        ).offset(skip).limit(limit).all()
        
        # SQLAlchemy 모델 객체를 딕셔너리로 변환
        return [r.to_dict() for r in records]
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB 조회 오류: {e}")

# ----------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)