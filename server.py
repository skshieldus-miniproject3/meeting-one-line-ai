"""CLOVA Speech REST API 서버 (DB 연동 제거 버전)"""

import os
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

from fastapi import (
    FastAPI, UploadFile, File, HTTPException, BackgroundTasks,
    Depends, Form  # Depends는 DB 제거로 실제 사용 안 함
)
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

# --- [DB 연동 제거] ---
# from sqlalchemy.orm import Session
# from src.core import database, models
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
# ai_analyzer는 DB와 무관하므로 그대로 사용
from src.core.ai_analyzer import ReportGenerator, ReportGeneratorError

# .env 로드
load_dotenv()

# --- [DB 연동 제거] ---
# models.Base.metadata.create_all(bind=database.engine) # DB 테이블 생성 제거
# -----------------------

# FastAPI 앱 생성
app = FastAPI(
    title="CLOVA Speech STT API",
    description="NAVER Cloud Platform CLOVA Speech API 서버 (DB 연동 제거)",
    version="1.3.0" # 버전 업데이트
)

# 글로벌 설정
INVOKE_URL = os.getenv('CLOVA_SPEECH_INVOKE_URL')
SECRET_KEY = os.getenv('CLOVA_SPEECH_SECRET_KEY')

if not INVOKE_URL or not SECRET_KEY:
    raise ValueError("CLOVA_SPEECH_INVOKE_URL과 CLOVA_SPEECH_SECRET_KEY를 .env에 설정해주세요")

# 클라이언트 인스턴스
# [참고] stt_client.py는 'enko' 기본값을 사용 (kwak 버전 기준)
client = ClovaSpeechClient(INVOKE_URL, SECRET_KEY)

# AI 보고서 생성기 (OpenAI 키가 있는 경우에만)
# [참고] ai_analyzer.py는 병합된 10개 AI 기능 포함 (kwak 버전 기준)
try:
    report_generator = ReportGenerator()
except Exception:
    report_generator = None

# 작업 상태 저장소 (실제 환경에서는 Redis 등 사용)
job_store: Dict[str, Dict[str, Any]] = {}


# --- [DB 연동 제거] ---
# def get_db(): ... 함수 제거
# -----------------------


class URLRequest(BaseModel):
    """URL 방식 요청 모델"""
    url: HttpUrl
    language: str = "enko" # kwak 버전 기본값
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
    language: str = "enko" # kwak 버전 기본값
    include_ai_summary: bool = True
    meeting_title: Optional[str] = None
    speaker_count_min: int = 2
    speaker_count_max: int = 10


class TranscriptFormatRequest(BaseModel):
    """대화록 포맷팅 요청 모델"""
    segments: list
    format_type: str = "basic"
    include_timestamps: bool = False
    include_confidence: bool = False
    speaker_names: Optional[Dict[str, str]] = None


class SummaryRequest(BaseModel):
    """요약 요청 모델"""
    transcript: str
    summary_type: str = "summary" # 병합된 모든 AI 타입 포함


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
        "version": "1.3.0", # 버전 업데이트
        "endpoints": [
            "/stt/url",
            "/stt/file",
            "/stt/status/{job_id}",
            "/meeting/transcribe",
            "/transcript/format",
            "/transcript/summarize",
            "/transcript/statistics",
            "/upload_and_analyze", # [수정] DB 저장 기능 없음
            # "/history" # [삭제] DB 조회 엔드포인트 제거
        ]
    }


@app.get("/health")
async def health():
    """헬스체크"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# --- [기존 엔드포인트들 (stt/url, stt/file 등)은 DB와 무관하므로 변경 없음] ---
# ... (stt/url, stt/file, stt/status/{job_id} 코드 생략 - 이전 버전과 동일) ...

@app.post("/stt/url")
async def transcribe_url(request: URLRequest, background_tasks: BackgroundTasks):
    """URL 방식 음성 인식 (kwak 버전 기본값 enko 적용)"""
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
        result = client.request_by_url(str(request.url), **options)

        if request.completion == 'sync':
            return JSONResponse(content=result)
        else:
            job_id = result if isinstance(result, str) else str(uuid.uuid4())
            job_store[job_id] = {
                'status': 'processing',
                'created_at': datetime.now().isoformat(),
                'type': 'url', 'url': str(request.url), 'options': options, 'result': None
            }
            background_tasks.add_task(poll_result, job_id)
            return JobResponse(job_id=job_id, status='processing', created_at=job_store[job_id]['created_at'])

    except ClovaSpeechError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {e}")


@app.post("/stt/file")
async def transcribe_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    language: str = Form("enko"), # kwak 버전 기본값
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
    """파일 업로드 방식 음성 인식 (kwak 버전 기본값 enko 적용)"""
    temp_file = None # finally 블록에서 사용하기 위해 초기화
    try:
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / f"{uuid.uuid4()}_{file.filename}"

        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)

        options = {
            'language': language, 'completion': completion, 'wordAlignment': word_alignment,
            'fullText': full_text, 'enable_diarization': enable_diarization,
            'enable_noise_filtering': enable_noise_filtering, 'enable_sed': enable_sed
        }
        if enable_diarization:
            options['diarization'] = {'enable': True, 'speakerCountMin': speaker_count_min, 'speakerCountMax': speaker_count_max}
        if callback: options['callback'] = callback
        if userdata: options['userdata'] = userdata

        result = client.request_by_file(temp_file, **options)

        if completion == 'sync':
            temp_file.unlink(missing_ok=True)
            return JSONResponse(content=result)
        else:
            job_id = result if isinstance(result, str) else str(uuid.uuid4())
            job_store[job_id] = {
                'status': 'processing', 'created_at': datetime.now().isoformat(),
                'type': 'file', 'filename': file.filename, 'temp_file': str(temp_file),
                'options': options, 'result': None
            }
            background_tasks.add_task(poll_result, job_id)
            # temp_file은 poll_result에서 삭제하므로 여기서는 삭제 안 함
            return JobResponse(job_id=job_id, status='processing', created_at=job_store[job_id]['created_at'])

    except ClovaSpeechError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {e}")
    finally:
        # 비동기 모드가 아니고 임시 파일이 생성된 경우 여기서 삭제
        if completion == 'sync' and temp_file and temp_file.exists():
             temp_file.unlink(missing_ok=True)


@app.get("/stt/status/{job_id}")
async def get_job_status(job_id: str):
    """작업 상태 조회"""
    if job_id not in job_store:
        raise HTTPException(status_code=404, detail="작업을 찾을 수 없습니다")
    job_info = job_store[job_id]
    if job_info['status'] == 'completed' and job_info['result']:
        return JSONResponse(content=job_info['result'])
    elif job_info['status'] == 'failed':
        return JobResponse(job_id=job_id, status='failed', created_at=job_info['created_at'], message=job_info.get('error'))
    else:
        return JobResponse(job_id=job_id, status=job_info['status'], created_at=job_info['created_at'])


async def poll_result(job_id: str):
    """백그라운드에서 비동기 작업 결과 폴링 (DB 무관)"""
    try:
        result = client.wait_for_completion(job_id, poll_interval=2, max_wait=300)
        job_store[job_id]['status'] = 'completed'
        job_store[job_id]['result'] = result
    except Exception as e:
        job_store[job_id]['status'] = 'failed'
        job_store[job_id]['error'] = str(e)
    finally:
        # 비동기 작업 완료 후 임시 파일 정리
        if job_store[job_id]['type'] == 'file':
            temp_file_path = job_store[job_id].get('temp_file')
            if temp_file_path and Path(temp_file_path).exists():
                Path(temp_file_path).unlink(missing_ok=True)


@app.on_event("startup")
async def startup():
    """서버 시작 시 실행 (DB 초기화 제거)"""
    # --- [DB 연동 제거] ---
    # database.init_db() 제거
    # -----------------------
    print("CLOVA Speech STT API server started")
    print(f"Invoke URL: {INVOKE_URL}")


@app.on_event("shutdown")
async def shutdown():
    """서버 종료 시 정리 (DB 무관)"""
    print("Server shutting down...")
    temp_dir = Path("temp")
    if temp_dir.exists():
        for temp_file in temp_dir.glob("*"):
            temp_file.unlink(missing_ok=True)
        try:
            temp_dir.rmdir()
        except OSError: pass
    print("Cleanup completed")


@app.post("/meeting/transcribe")
async def transcribe_meeting(request: MeetingRequest, background_tasks: BackgroundTasks):
    """회의 오디오 전사 및 AI 요약 (URL 방식, DB 무관)"""
    try:
        options = {
            'language': request.language, 'completion': 'sync', 'wordAlignment': True,
            'fullText': True, 'enable_diarization': True, 'enable_noise_filtering': True,
            'diarization': {'enable': True, 'speakerCountMin': request.speaker_count_min, 'speakerCountMax': request.speaker_count_max}
        }
        stt_result = client.request_by_url(str(request.url), **options)

        if 'segments' not in stt_result:
            raise HTTPException(status_code=400, detail="STT 결과에 segments가 없습니다")

        segments = stt_result['segments']
        transcript = format_segments_to_transcript(segments)
        detailed_transcript = format_segments_to_detailed_transcript(segments, include_timestamps=True, include_confidence=True)
        speaker_stats = extract_speaker_statistics(segments)
        response_data = {
            'stt_result': stt_result, 'transcript': transcript, 'detailed_transcript': detailed_transcript,
            'speaker_statistics': speaker_stats, 'meeting_header': generate_meeting_summary_header(segments, request.meeting_title)
        }

        if request.include_ai_summary and report_generator:
            try:
                # 병합된 모든 AI 기능 호출
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
    """STT segments 포맷팅 (DB 무관)"""
    try:
        segments = request.segments
        if request.format_type == "basic":
            result = format_segments_to_transcript(segments)
        elif request.format_type == "detailed":
            result = format_segments_to_detailed_transcript(segments, include_timestamps=request.include_timestamps, include_confidence=request.include_confidence)
        elif request.format_type == "with_speakers" and request.speaker_names:
            from src.core.formatter import format_transcript_with_speakers
            result = format_transcript_with_speakers(segments, request.speaker_names)
        else:
            raise HTTPException(status_code=400, detail="지원하지 않는 format_type입니다")
        return JSONResponse(content={"formatted_transcript": result})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"포맷팅 오류: {e}")


@app.post("/transcript/summarize")
async def summarize_transcript(request: SummaryRequest):
    """대화록 AI 요약 및 분석 (DB 무관, 병합된 모든 AI 타입 지원)"""
    if not report_generator:
        raise HTTPException(status_code=503, detail="AI 요약 기능을 사용할 수 없습니다. OPENAI_API_KEY를 설정해주세요.")

    try:
        transcript = request.transcript
        summary_type = request.summary_type
        result = None

        if summary_type == "summary": result = report_generator.summarize(transcript)
        elif summary_type == "meeting_notes": result = report_generator.generate_meeting_notes(transcript)
        elif summary_type == "action_items": result = report_generator.generate_action_items(transcript)
        elif summary_type == "sentiment": result = report_generator.analyze_sentiment(transcript)
        elif summary_type == "follow_up": result = report_generator.generate_follow_up_questions(transcript)
        elif summary_type == "keywords": result = report_generator.extract_keywords(transcript)
        elif summary_type == "topics": result = report_generator.classify_topics(transcript)
        elif summary_type == "by_speaker": result = report_generator.analyze_by_speaker(transcript)
        elif summary_type == "meeting_type": result = report_generator.classify_meeting_type(transcript)
        elif summary_type == "speaker_summary": result = report_generator.summarize_by_speaker(transcript)
        else: raise HTTPException(status_code=400, detail="지원하지 않는 summary_type입니다")

        return JSONResponse(content={"summary_type": summary_type, "result": result})

    except ReportGeneratorError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"요약 생성 오류: {e}")


@app.post("/transcript/statistics")
async def get_transcript_statistics(segments: list):
    """대화록 통계 정보 추출 (DB 무관)"""
    try:
        stats = extract_speaker_statistics(segments)
        header = generate_meeting_summary_header(segments)
        return JSONResponse(content={"speaker_statistics": stats, "meeting_header": header})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 추출 오류: {e}")


# --- [수정된 통합 엔드포인트 (DB 저장 제거)] ---

@app.post("/upload_and_analyze")
async def upload_and_analyze(
    # [DB 연동 제거] db: Session = Depends(get_db), 제거
    file: UploadFile = File(...),
    language: str = Form("enko"), # kwak 버전 기본값
    speaker_count_min: int = Form(2),
    speaker_count_max: int = Form(10)
):
    """
    [수정] 파일 업로드 + STT + 전체 AI 분석 (DB 저장 X) 통합 엔드포인트
    """
    if not report_generator:
        raise HTTPException(status_code=503, detail="AI 요약 기능을 사용할 수 없습니다. OPENAI_API_KEY를 설정해주세요.")

    temp_file = None # finally 블록에서 사용
    try:
        # 1. 파일 임시 저장
        temp_dir = Path("temp")
        temp_dir.mkdir(exist_ok=True)
        temp_file = temp_dir / f"{uuid.uuid4()}_{file.filename}"

        with open(temp_file, "wb") as f:
            content = await file.read()
            f.write(content)

        # 2. STT 옵션 구성 및 실행
        options = {
            'language': language, 'completion': 'sync', 'wordAlignment': True, 'fullText': True,
            'enable_diarization': True, 'enable_noise_filtering': True,
            'diarization': {'enable': True, 'speakerCountMin': speaker_count_min, 'speakerCountMax': speaker_count_max}
        }
        stt_result = client.request_by_file(temp_file, **options)

    except ClovaSpeechError as e:
        raise HTTPException(status_code=400, detail=f"STT 오류: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"파일 처리 또는 STT 오류: {e}")
    finally:
        # 임시 파일 삭제
        if temp_file and temp_file.exists():
            temp_file.unlink(missing_ok=True)

    if 'segments' not in stt_result:
        raise HTTPException(status_code=400, detail="STT 결과에 segments가 없습니다")

    segments = stt_result.get('segments', [])
    transcript = format_segments_to_transcript(segments)
    speaker_stats = extract_speaker_statistics(segments)

    ai_reports = None
    ai_reports_error = None

    # 3. AI 전체 분석 실행
    try:
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
            'speaker_summary': report_generator.summarize_by_speaker(transcript),
        }
    except Exception as e:
        ai_reports_error = f"AI 요약 생성 실패: {str(e)}"

    # --- [DB 연동 제거] ---
    # 4. DB 저장 로직 제거
    # try: ... db.add()... 제거
    # -----------------------

    # 5. 통합 결과 반환 (DB ID, 생성 시간 제거)
    response_data = {
        # "db_id": None, # 제거
        "filename": file.filename,
        # "created_at": None, # 제거
        "transcript": transcript,
        "speaker_statistics": speaker_stats,
        "ai_reports": ai_reports,
        "ai_reports_error": ai_reports_error,
        "stt_raw_result": stt_result
    }

    return JSONResponse(content=response_data)

# --- [DB 연동 제거] ---
# @app.get("/history", ...) 엔드포인트 제거
# -----------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)