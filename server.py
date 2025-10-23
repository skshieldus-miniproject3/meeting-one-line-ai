"""CLOVA Speech REST API 서버 (DB 연동 제거 버전)
[수정됨] App 서버 연동을 위한 /ai/analyze 엔드포인트 및 콜백 로직 추가
[수정됨] /meetings 회의록 목록 조회 엔드포인트 추가
"""

import os
import uuid
import asyncio
import json      # 임베딩 파일 처리용
import requests  # [신규] 콜백 전송을 위해 추가
import math      # [신규] 페이징 계산을 위해 추가
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
from src.core.embedding_manager import EmbeddingManager

# .env 로드
load_dotenv()

# --- [DB 연동 제거] ---
# models.Base.metadata.create_all(bind=database.engine) # DB 테이블 생성 제거
# -----------------------

# FastAPI 앱 생성
app = FastAPI(
    title="CLOVA Speech STT API",
    description="NAVER Cloud Platform CLOVA Speech API 서버 (DB 연동 제거)",
    version="1.6.0" # [수정] 버전 업데이트 (회의록 목록)
)

# 글로벌 설정
INVOKE_URL = os.getenv('CLOVA_SPEECH_INVOKE_URL')
SECRET_KEY = os.getenv('CLOVA_SPEECH_SECRET_KEY')
# [신규] App 서버 콜백 호스트
APP_SERVER_CALLBACK_HOST = os.getenv('APP_SERVER_CALLBACK_HOST')

if not INVOKE_URL or not SECRET_KEY:
    raise ValueError("CLOVA_SPEECH_INVOKE_URL과 CLOVA_SPEECH_SECRET_KEY를 .env에 설정해주세요")

# 클라이언트 인스턴스
client = ClovaSpeechClient(INVOKE_URL, SECRET_KEY)

# AI 보고서 생성기 (OpenAI 키가 있는 경우에만)
try:
    report_generator = ReportGenerator()
except Exception:
    report_generator = None

# 임베딩 관리자
embedding_manager = EmbeddingManager(embeddings_dir="embeddings")

# 작업 상태 저장소 (실제 환경에서는 Redis 등 사용)
job_store: Dict[str, Dict[str, Any]] = {}


# --- [DB 연동 제거] ---
# def get_db(): ... 함수 제거
# -----------------------


class URLRequest(BaseModel):
    """URL 방식 요청 모델"""
    url: HttpUrl
    language: str = "enko"
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
    language: str = "enko"
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
    summary_type: str = "summary"


class SemanticSearchRequest(BaseModel):
    """의미 검색 요청 모델"""
    query: str
    top_k: int = 5


class UpdateEmbeddingRequest(BaseModel):
    """임베딩 업데이트 요청 모델"""
    summary: str  # 수정된 요약문
    title: Optional[str] = None  # 수정된 제목 (선택)


class JobResponse(BaseModel):
    """작업 응답 모델"""
    job_id: str
    status: str
    created_at: str
    message: Optional[str] = None

# ========================================
# App 서버 연동을 위한 Pydantic 모델
# ========================================

class AiAnalyzeRequest(BaseModel):
    """AI 분석 요청 모델 (App Server -> AI Server)"""
    meetingId: str
    filePath: str # 예: /data/uploads/meeting_123.wav

class AiAnalyzeResponse(BaseModel):
    """AI 분석 요청 즉시 응답 모델"""
    status: str

# ========================================
# [신규] 3.4 회의록 목록 조회를 위한 Pydantic 모델
# ========================================

class MeetingListItem(BaseModel):
    """회의록 목록의 개별 항목 모델"""
    meetingId: str
    title: str
    status: str
    summary: Optional[str] = None
    createdAt: str

class MeetingListResponse(BaseModel):
    """회의록 목록 페이징 응답 모델"""
    content: List[MeetingListItem]
    page: int
    size: int
    totalPages: int

# ========================================

@app.get("/")
async def root():
    """루트 엔드포인트"""
    return {
        "service": "CLOVA Speech STT API",
        "version": "1.6.0", # [수정] 버전 업데이트
        "endpoints": [
            "/stt/url",
            "/stt/file",
            "/stt/status/{job_id}",
            "/meeting/transcribe",
            "/transcript/format",
            "/transcript/summarize",
            "/transcript/statistics",
            "/upload_and_analyze",
            "/search/semantic",  # 의미 기반 검색
            "/embeddings/{meeting_id}",  # 임베딩 업데이트 (PUT)
            "/ai/analyze", # App 서버 연동 엔드포인트
            "/meetings"   # [신규] 회의록 목록 조회
        ]
    }


@app.get("/health")
async def health():
    """헬스체크"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


# ========================================
# App 서버 연동을 위한 헬퍼 함수 및 백그라운드 작업
# ========================================

def format_clova_to_app_speakers(segments: list) -> list:
    """
    CLOVA STT segments를 App 서버 콜백('speakers') 형식으로 변환합니다.
    - 화자별로 그룹화
    - 타임스탬프를 밀리초(ms)에서 초(sec)로 변환
    """
    speakers_dict = {}
    
    if not segments:
        return []
        
    for segment in segments:
        try:
            # "S1", "S2" ... 형식으로 speakerId 생성
            speaker_label = segment.get("speaker", {}).get("label", "Unknown")
            speaker_id = f"S{speaker_label}" 
            
            if speaker_id not in speakers_dict:
                speakers_dict[speaker_id] = {"speakerId": speaker_id, "segments": []}
            
            # 밀리초 -> 초 변환 (소수점)
            start_sec = segment.get("start", 0) / 1000.0
            end_sec = segment.get("end", 0) / 1000.0
            
            speakers_dict[speaker_id]["segments"].append({
                "start": round(start_sec, 2), # 소수점 2자리
                "end": round(end_sec, 2),   # 소수점 2자리
                "text": segment.get("text", "").strip()
            })
        except Exception as e:
            # 세그먼트 하나가 실패해도 전체가 중단되지 않도록 로깅만
            print(f"세그먼트 포맷팅 중 오류: {e} (세그먼트: {segment})")

    # 각 화자의 segments를 시간순으로 정렬
    for speaker_data in speakers_dict.values():
        speaker_data["segments"].sort(key=lambda x: x["start"])

    # 딕셔너리의 값들(value)을 리스트로 변환하여 반환
    return list(speakers_dict.values())


async def background_analysis_task(meeting_id: str, file_path: str):
    """
    백그라운드에서 STT, 화자 분리, 요약, 키워드 추출을 수행하고
    App 서버로 새로운 형식의 콜백을 전송하는 함수
    """
    print(f"[Task {meeting_id}] AI 분석 작업 시작: {file_path}")
    
    # App 서버로 전송할 콜백 데이터
    callback_data = {
        "status": "failed", # 기본값 'failed'
        "summary": None,
        "keywords": [],
        "speakers": [],
        "error": None
    }
    
    try:
        # 0. 파일 경로 유효성 검사 (AI 서버 로컬 경로)
        local_path = Path(file_path)
        if not local_path.exists():
            raise FileNotFoundError(f"AI 서버에서 해당 파일을 찾을 수 없습니다: {file_path}")

        # 1. STT + 화자 분리 (CLOVA Speech 사용)
        print(f"[Task {meeting_id}] 1. STT(Clova) 및 화자 분리 시작...")
        stt_options = {
            'language': 'enko',  # 한영 혼합
            'completion': 'sync', # 동기 처리 (결과를 바로 받아야 함)
            'wordAlignment': True,
            'fullText': True,
            'enable_diarization': True, # 화자 분리 활성화
            'diarization': {
                'enable': True,
                'speakerCountMin': 2,
                'speakerCountMax': 10
            }
        }
        
        # 'client'는 server.py의 전역 ClovaSpeechClient 인스턴스
        stt_result = client.request_by_file(local_path, **stt_options)
        
        if 'segments' not in stt_result or not stt_result['segments']:
            raise ValueError("STT 실패: Clova 결과에 'segments'가 없습니다.")
        
        segments = stt_result.get('segments', [])
        print(f"[Task {meeting_id}] 2. STT 완료 (세그먼트 {len(segments)}개)")
        
        # 2. 대화록(flat text) 변환 (AI 분석용)
        transcript = format_segments_to_transcript(segments)
        
        # 3. AI 분석 (요약 + 키워드)
        # 'report_generator'는 server.py의 전역 ReportGenerator 인스턴스
        if report_generator and transcript:
            print(f"[Task {meeting_id}] 3. AI 요약(OpenAI) 시작...")
            # 3a. 요약
            callback_data["summary"] = report_generator.summarize(transcript)
            
            print(f"[Task {meeting_id}] 4. AI 키워드 추출(OpenAI) 시작...")
            # 3b. 키워드
            # extract_keywords는 텍스트를 반환
            keyword_text = report_generator.extract_keywords(transcript) 
            
            # AI가 반환한 텍스트("- 키워드1\n- 키워드2")를 파싱하여 리스트로 변환
            raw_keywords = [line.strip().lstrip('-•* ').strip() for line in keyword_text.split('\n') if line.strip().lstrip('-•* ').strip()]

            # 키워드 정제
            # 1. 카테고리명 제거 (끝에 ':' 있는 것)
            # 2. 긴 문장 제거 (30자 이상 또는 마침표/쉼표 2개 이상 포함)
            # 3. 중복 제거 (순서 유지)
            # 4. 상위 10개만 선택
            filtered_keywords = [
                kw for kw in raw_keywords
                if not kw.endswith(':')
                and len(kw) <= 30
                and (kw.count('.') + kw.count(',')) < 2  # 설명문 제거
            ]
            # 중복 제거 (순서 유지)
            unique_keywords = list(dict.fromkeys(filtered_keywords))
            # 상위 10개만
            callback_data["keywords"] = unique_keywords[:10]

            # 임베딩 생성 및 저장
            try:
                print(f"[Task {meeting_id}] 임베딩 생성 및 저장 중...")
                embedding = report_generator.generate_embedding(callback_data["summary"])
                embedding_manager.save_meeting_embedding(
                    meeting_id=meeting_id,
                    title=Path(file_path).stem,
                    summary=callback_data["summary"],
                    embedding=embedding
                )
                print(f"[Task {meeting_id}] ✅ 임베딩 저장 완료")
            except Exception as e:
                print(f"[Task {meeting_id}] ⚠️ 임베딩 저장 실패: {e}")

        else:
            print(f"[Task {meeting_id}] 3. AI 요약기(OpenAI)가 없거나 대화록이 비어 요약을 건너뜁니다.")

        # 4. 화자 데이터 포맷팅 (App 서버 요구사항)
        print(f"[Task {meeting_id}] 5. 화자 데이터 포맷팅...")
        callback_data["speakers"] = format_clova_to_app_speakers(segments)
        
        callback_data["status"] = "completed"
        # 성공 시 error 필드는 None이므로 제거 (json에서 제외)
        callback_data.pop("error") 

    except Exception as e:
        print(f"[Task {meeting_id}] ❌ 분석 중 오류 발생: {e}")
        callback_data["error"] = str(e)
        callback_data["status"] = "failed" # 명시적으로 failed 설정
    
    # 5. App 서버로 콜백 전송
    if not APP_SERVER_CALLBACK_HOST:
        print(f"[Task {meeting_id}] ⚠️ .env에 APP_SERVER_CALLBACK_HOST가 설정되지 않아 콜백을 보낼 수 없습니다.")
        return
        
    # 요청 사양에 명시된 콜백 URL
    callback_url = f"{APP_SERVER_CALLBACK_HOST}/api/meetings/{meeting_id}/callback"
    
    try:
        # status가 'failed'일 때도 에러 정보를 콜백으로 전송
        print(f"[Task {meeting_id}] 6. App 서버로 콜백 전송: {callback_url}")
        response = requests.post(callback_url, json=callback_data, timeout=15)
        response.raise_for_status() # 4xx, 5xx 에러 발생 시 예외
        print(f"[Task {meeting_id}] 7. 콜백 전송 성공 (App 서버 응답: {response.status_code})")
    except requests.exceptions.RequestException as e:
        print(f"[Task {meeting_id}] ❌ 콜백 전송 실패: {e} (App 서버가 응답하지 않거나 에러 발생)")

# ========================================
# (기존 엔드포인트 - 변경 없음)
# ========================================

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
    language: str = Form("enko"),
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
    temp_file = None
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
            return JobResponse(job_id=job_id, status='processing', created_at=job_store[job_id]['created_at'])

    except ClovaSpeechError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"내부 서버 오류: {e}")
    finally:
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
        if job_store[job_id]['type'] == 'file':
            temp_file_path = job_store[job_id].get('temp_file')
            if temp_file_path and Path(temp_file_path).exists():
                Path(temp_file_path).unlink(missing_ok=True)


@app.on_event("startup")
async def startup():
    """서버 시작 시 실행 (DB 초기화 제거)"""
    print("CLOVA Speech STT API server started")
    print(f"Invoke URL: {INVOKE_URL}")
    if APP_SERVER_CALLBACK_HOST:
        print(f"[CALLBACK] App Server Host: {APP_SERVER_CALLBACK_HOST}")
    else:
        print("[CALLBACK] ⚠️ APP_SERVER_CALLBACK_HOST가 .env에 설정되지 않았습니다. /ai/analyze 콜백이 작동하지 않습니다.")


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
        elif summary_type == "engagement_score": result = report_generator.calculate_engagement_score(transcript)
        elif summary_type == "improvement_suggestions": result = report_generator.generate_improvement_suggestions(transcript)
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


@app.post("/upload_and_analyze")
async def upload_and_analyze(
    file: UploadFile = File(...),
    language: str = Form("enko"),
    speaker_count_min: int = Form(2),
    speaker_count_max: int = Form(10)
):
    """
    [수정] 파일 업로드 + STT + 전체 AI 분석 (DB 저장 X) 통합 엔드포인트
    """
    if not report_generator:
        raise HTTPException(status_code=503, detail="AI 요약 기능을 사용할 수 없습니다. OPENAI_API_KEY를 설정해주세요.")

    temp_file = None
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

    # 5. 통합 결과 반환 (DB ID, 생성 시간 제거)
    response_data = {
        "filename": file.filename,
        "transcript": transcript,
        "speaker_statistics": speaker_stats,
        "ai_reports": ai_reports,
        "ai_reports_error": ai_reports_error,
        "stt_raw_result": stt_result
    }

    return JSONResponse(content=response_data)


# ========================================
# 임베딩 검색 API (기존)
# ========================================

@app.post("/search/semantic")
async def semantic_search(request: SemanticSearchRequest):
    """
    의미 기반 회의록 검색
    """
    if not report_generator:
        raise HTTPException(status_code=500, detail="OpenAI API 키가 설정되지 않았습니다")

    try:
        # 1. 검색 쿼리를 임베딩으로 변환
        query_embedding = report_generator.generate_embedding(request.query)

        # 2. 유사한 회의록 검색
        results = embedding_manager.search_similar_meetings(
            query_embedding=query_embedding,
            top_k=request.top_k
        )

        return {
            "query": request.query,
            "total_results": len(results),
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")


@app.put("/embeddings/{meeting_id}")
async def update_meeting_embedding(
    meeting_id: str,
    request: UpdateEmbeddingRequest
):
    """
    회의록 수정 시 임베딩을 업데이트합니다.

    백엔드에서 회의록(요약 또는 제목)을 수정한 후 호출합니다.
    """
    if not report_generator:
        raise HTTPException(status_code=500, detail="OpenAI API 키가 설정되지 않았습니다")

    try:
        print(f"[UPDATE] 임베딩 업데이트 요청: {meeting_id}")

        # 기존 임베딩 파일 로드 (제목 등 메타데이터 유지용)
        existing_file = embedding_manager.embeddings_dir / f"meeting_{meeting_id}.json"
        title = request.title

        if not title and existing_file.exists():
            # 제목이 없으면 기존 파일에서 가져오기
            with open(existing_file, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
                title = existing_data.get("title", meeting_id)

        # 새 임베딩 생성
        embedding = report_generator.generate_embedding(request.summary)

        # 임베딩 저장 (덮어쓰기)
        embedding_manager.save_meeting_embedding(
            meeting_id=meeting_id,
            title=title or meeting_id,
            summary=request.summary,
            embedding=embedding
        )

        print(f"[UPDATE] ✅ 임베딩 업데이트 완료: {meeting_id}")

        return {
            "status": "success",
            "message": f"임베딩이 업데이트되었습니다: {meeting_id}"
        }

    except Exception as e:
        print(f"[UPDATE] ❌ 임베딩 업데이트 실패: {e}")
        raise HTTPException(status_code=500, detail=f"임베딩 업데이트 실패: {str(e)}")


# ========================================
# App 서버 연동 엔드포인트
# ========================================

@app.post("/ai/analyze", 
          response_model=AiAnalyzeResponse,
          status_code=200) # 요청 사양에 명시된 200 OK
async def request_ai_analysis(
    request: AiAnalyzeRequest,
    background_tasks: BackgroundTasks
):
    """
    App 서버로부터 AI 분석(STT + 화자분리 + 요약)을 요청받습니다.
    
    즉시 'processing' 상태를 응답하고, 백그라운드에서 실제 작업을 수행합니다.
    작업 완료 시 App 서버의 '/api/meetings/{id}/callback'으로 결과를 POST합니다.
    """
    print(f"✅ AI 분석 요청 수신: {request.meetingId} (파일: {request.filePath})")
    
    # (보안 참고) 실제 운영 시, filePath가 허용된 디렉토리(예: /data/uploads/)
    # 내에 있는지 엄격히 검사해야 합니다.
    # 예: if not request.filePath.startswith("/data/uploads/"):
    #        raise HTTPException(status_code=400, detail="Invalid filePath")
    
    # 백그라운드 작업 등록
    background_tasks.add_task(
        background_analysis_task, # 위에서 정의한 백그라운드 함수
        request.meetingId,
        request.filePath
    )
    
    # 요청 사양대로 즉시 응답 반환
    return AiAnalyzeResponse(status="processing")


# ========================================
# [신규] 3.4 회의록 목록 검색 조회
# ========================================

@app.get("/meetings", response_model=MeetingListResponse)
async def get_meeting_list(
    page: int = 1,
    size: int = 10,
    keyword: Optional[str] = None,
    title: Optional[str] = None,
    summary: Optional[str] = None,
    status: Optional[str] = None
):
    """
    [신규] 저장된 회의록 목록을 페이지 단위로 조회합니다.
    
    데이터는 `embeddings/` 디렉토리에 저장된 JSON 파일들을 기반으로 합니다.
    
    - `keyword`: 제목과 요약에서 동시 검색
    - `title`: 제목에서 검색
    - `summary`: 요약에서 검색
    - `status`: 상태에서 검색 (현재 "completed"만 조회 가능)
    """
    try:
        # 1. 모든 임베딩 데이터(회의록 정보) 로드
        # 'embedding_manager'는 server.py의 전역 인스턴스
        all_meetings_data = embedding_manager.load_all_embeddings()
        
        enriched_meetings = []
        for meeting_data in all_meetings_data:
            meeting_id = meeting_data.get("meeting_id")
            if not meeting_id:
                continue
            
            # 2. 'createdAt' (파일 수정시간), 'status' 필드 동적 추가
            file_path = embedding_manager.embeddings_dir / f"meeting_{meeting_id}.json"
            created_at_iso = datetime.now().isoformat() # 기본값
            
            if file_path.exists():
                try:
                    # 파일의 마지막 수정 시간을 'createdAt'으로 사용
                    mtime = file_path.stat().st_mtime
                    created_at_iso = datetime.fromtimestamp(mtime).isoformat()
                except Exception as e:
                    print(f"파일 시간 읽기 오류: {meeting_id} - {e}")
            
            enriched_meetings.append({
                "meetingId": meeting_id,
                "title": meeting_data.get("title", ""),
                "summary": meeting_data.get("summary", ""),
                "status": "completed", # 임베딩이 저장된 것은 'completed'로 간주
                "createdAt": created_at_iso
            })
        
        # 3. 필터링 로직
        filtered_meetings = enriched_meetings

        # 3a. 'keyword' (제목 + 요약 동시 검색)
        if keyword:
            kw = keyword.lower()
            filtered_meetings = [
                m for m in filtered_meetings 
                if kw in m['title'].lower() or kw in m['summary'].lower()
            ]
        
        # 3b. 'title' (제목 검색)
        if title:
            filtered_meetings = [
                m for m in filtered_meetings 
                if title.lower() in m['title'].lower()
            ]
        
        # 3c. 'summary' (요약 검색)
        if summary:
            filtered_meetings = [
                m for m in filtered_meetings 
                if summary.lower() in m['summary'].lower()
            ]
        
        # 3d. 'status' (상태 검색)
        if status:
            filtered_meetings = [
                m for m in filtered_meetings 
                if status.lower() == m['status'].lower()
            ]

        # 4. 정렬 (최신순)
        filtered_meetings.sort(key=lambda m: m['createdAt'], reverse=True)

        # 5. 페이징
        total_items = len(filtered_meetings)
        total_pages = math.ceil(total_items / size)
        
        # 페이지 번호 보정
        if page < 1:
            page = 1
        
        start_index = (page - 1) * size
        end_index = start_index + size
        
        paginated_content = filtered_meetings[start_index:end_index]

        # 6. 응답 모델로 반환
        return MeetingListResponse(
            content=paginated_content,
            page=page,
            size=size,
            totalPages=total_pages
        )

    except Exception as e:
        print(f"❌ /meetings 엔드포인트 오류: {e}")
        raise HTTPException(status_code=500, detail=f"회의록 목록 조회 중 오류 발생: {e}")


# ========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)