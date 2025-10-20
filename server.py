"""CLOVA Speech REST API 서버"""

import os
import uuid
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, HttpUrl
from dotenv import load_dotenv

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
    description="NAVER Cloud Platform CLOVA Speech API 서버",
    version="1.0.0"
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
    summary_type: str = "summary"  # summary, meeting_notes, action_items, sentiment, follow_up


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
        "version": "1.0.0",
        "endpoints": [
            "/stt/url",
            "/stt/file",
            "/stt/status/{job_id}",
            "/meeting/transcribe",
            "/transcript/format",
            "/transcript/summarize",
            "/transcript/statistics"
        ]
    }


@app.get("/health")
async def health():
    """헬스체크"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


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
    language: str = "ko-KR",
    completion: str = "sync",
    word_alignment: bool = True,
    full_text: bool = True,
    callback: Optional[str] = None,
    userdata: Optional[str] = None,
    enable_diarization: bool = True,
    enable_noise_filtering: bool = True,
    enable_sed: bool = False,
    speaker_count_min: int = 2,
    speaker_count_max: int = 10
):
    """파일 업로드 방식 음성 인식"""
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
        temp_dir.rmdir()

    print("Cleanup completed")


@app.post("/meeting/transcribe")
async def transcribe_meeting(request: MeetingRequest, background_tasks: BackgroundTasks):
    """회의 오디오 전사 및 AI 요약 (통합 엔드포인트)"""
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
                ai_reports = {
                    'summary': report_generator.summarize(transcript),
                    'meeting_notes': report_generator.generate_meeting_notes(transcript),
                    'action_items': report_generator.generate_action_items(transcript),
                    'sentiment': report_generator.analyze_sentiment(transcript),
                    'follow_up_questions': report_generator.generate_follow_up_questions(transcript)
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
    """STT segments를 다양한 형태의 대화록으로 포맷팅"""
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
            from ncp_clova_speech.formatter import format_transcript_with_speakers
            result = format_transcript_with_speakers(segments, request.speaker_names)
        else:
            raise HTTPException(status_code=400, detail="지원하지 않는 format_type입니다")

        return JSONResponse(content={"formatted_transcript": result})

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"포맷팅 오류: {e}")


@app.post("/transcript/summarize")
async def summarize_transcript(request: SummaryRequest):
    """대화록 AI 요약 및 분석"""
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
    """대화록 통계 정보 추출"""
    try:
        stats = extract_speaker_statistics(segments)
        header = generate_meeting_summary_header(segments)

        return JSONResponse(content={
            "speaker_statistics": stats,
            "meeting_header": header
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"통계 추출 오류: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)