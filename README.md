# Meeting Transcriber

AI 기반 회의 전사 및 분석 애플리케이션

## ✨ 주요 기능

- 🎯 **고정밀 음성 인식**: NAVER Cloud Platform CLOVA Speech 기반
- 👥 **화자 분리**: 최대 10명까지 화자 자동 인식
- 🧠 **AI 분석**: OpenAI GPT를 활용한 자동 요약 및 회의록 생성
- 📊 **통계 분석**: 화자별 발화 시간 및 참여도 분석
- 🌐 **REST API**: FastAPI 기반 웹 서비스
- 📝 **다양한 출력**: 텍스트, JSON, Markdown 형태로 결과 제공

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python -m venv .venv
.venv\Scripts\Activate.ps1  # Windows
# source .venv/bin/activate  # macOS/Linux

# 의존성 설치
pip install -r requirements.txt
pip install -e .
```

### 2. API 키 설정

`.env` 파일을 생성하고 API 키를 설정하세요:

```bash
# CLOVA Speech 설정 (필수)
CLOVA_SPEECH_INVOKE_URL=https://clovaspeech-gw.ncloud.com/external/v1/YOUR_PROJECT_ID/YOUR_ENDPOINT_ID
CLOVA_SPEECH_SECRET_KEY=your_secret_key_here

# OpenAI 설정 (AI 분석 기능용, 선택사항)
OPENAI_API_KEY=your_openai_api_key_here

# 기본 설정
DEFAULT_LANGUAGE=ko-KR
```

### 3. 오디오 파일 준비

오디오 파일을 `audio/` 디렉토리에 넣어주세요:

```bash
audio/
├── meeting1.wav
├── interview.mp3
└── conference.m4a
```

## 💻 사용법

### 🎵 기본 전사

```bash
# 기본 전사 (화자 분리 + 노이즈 필터링 자동 활성화)
python transcribe.py audio/meeting.wav

# 화자 분리 비활성화
python transcribe.py audio/meeting.wav --disable-diarization

# 음향 이벤트 탐지 활성화
python transcribe.py audio/meeting.wav --enable-sed

# 화자 수 제한
python transcribe.py audio/meeting.wav --speaker-min 2 --speaker-max 5
```

### 🧠 AI 분석

```bash
# 전체 AI 분석 (요약 + 회의록 + 액션아이템 + 분위기분석 + 후속질문)
python analyze.py audio/meeting.wav --full-analysis

# 특정 분석만 실행
python analyze.py audio/meeting.wav --summary --action-items

# 이미 전사된 텍스트 파일 분석
python analyze.py outputs/meeting_transcript.txt --meeting-notes

# 오디오 파일 + AI 분석 한번에
python analyze.py audio/meeting.wav --summary --sentiment
```

### 🌐 API 서버

```bash
# API 서버 실행
python server.py

# 또는 uvicorn 직접 실행
uvicorn server:app --host 0.0.0.0 --port 8000 --reload
```

API 서버가 실행되면 `http://localhost:8000`에서 사용할 수 있습니다.

**주요 엔드포인트:**
- `POST /meeting/transcribe` - 통합 회의 전사 및 AI 분석
- `POST /transcript/format` - 대화록 포맷팅
- `POST /transcript/summarize` - AI 요약 생성
- `POST /stt/file` - 기본 STT (파일 업로드)
- `POST /stt/url` - 기본 STT (URL 방식)

## 📁 프로젝트 구조

```
meeting-transcriber/
├── transcribe.py           # 🎯 메인 전사 도구
├── analyze.py              # 🧠 AI 분석 도구
├── server.py               # 🌐 API 서버
│
├── audio/                  # 사용자 오디오 파일
│   ├── meeting.wav
│   └── *.mp3, *.m4a
│
├── outputs/                # 전사 및 분석 결과
│   ├── *_transcript.txt
│   ├── *_result.json
│   └── *_analysis_report.md
│
├── src/                    # 핵심 라이브러리
│   ├── core/
│   │   ├── stt_client.py   # STT 클라이언트
│   │   ├── formatter.py    # 대화록 포맷팅
│   │   └── ai_analyzer.py  # AI 분석기
│   └── ncp_clova_speech/   # 호환성 래퍼
│
├── tools/                  # 개발자 도구 (레거시)
│   ├── transcribe_file.py
│   ├── transcribe_url.py
│   └── meeting_api_example.py
│
├── tests/                  # 테스트
├── .env                    # 환경변수 (사용자 생성)
├── .env.example            # 환경변수 템플릿
├── requirements.txt        # 의존성
└── setup.py               # 패키지 설정
```

## 🎯 사용 시나리오

### 시나리오 1: 빠른 회의 전사
```bash
# 회의 오디오를 텍스트로 변환
python transcribe.py audio/weekly_meeting.wav

# 결과: outputs/weekly_meeting_transcript.txt
```

### 시나리오 2: 완전 자동 분석
```bash
# 전사 + AI 분석 한번에
python analyze.py audio/board_meeting.wav --full-analysis

# 결과:
# - outputs/board_meeting_transcript.txt (대화록)
# - outputs/board_meeting_analysis_report.md (종합 분석 보고서)
```

### 시나리오 3: API 통합
```bash
# API 서버 실행
python server.py

# 다른 터미널에서 테스트
curl -X POST "http://localhost:8000/meeting/transcribe" \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com/meeting.wav", "include_ai_summary": true}'
```

## ⚙️ 고급 설정

### CLOVA Speech 옵션

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--language` | `ko-KR` | 언어 설정 |
| `--disable-diarization` | `False` | 화자 분리 비활성화 |
| `--disable-noise-filtering` | `False` | 노이즈 필터링 비활성화 |
| `--enable-sed` | `False` | 음향 이벤트 탐지 활성화 |
| `--speaker-min` | `2` | 최소 화자 수 |
| `--speaker-max` | `10` | 최대 화자 수 |

### AI 분석 옵션

| 옵션 | 설명 |
|------|------|
| `--summary` | 핵심 요약 생성 |
| `--meeting-notes` | 공식 회의록 생성 |
| `--action-items` | 액션 아이템 추출 |
| `--sentiment` | 회의 분위기 분석 |
| `--follow-up` | 후속 질문 생성 |
| `--full-analysis` | 모든 분석 실행 |

## 📊 출력 형태

### 기본 전사 결과
```
outputs/
├── meeting_transcript.txt      # 읽기 쉬운 대화록
├── meeting_result.json         # 상세 STT 결과 (타임스탬프, 신뢰도 등)
└── meeting_stats.json          # 화자별 통계
```

### AI 분석 결과
```
outputs/
├── meeting_analysis_report.md  # 종합 분석 보고서 (Markdown)
└── meeting_analysis.json       # 구조화된 분석 데이터 (JSON)
```

## 🧪 테스트

```bash
# 단위 테스트
pytest tests/

# 기본 기능 테스트 (샘플 오디오 필요)
python transcribe.py audio/sample.wav
python analyze.py audio/sample.wav --summary

# API 서버 테스트
python server.py &
curl http://localhost:8000/health
```

## 🔒 보안 및 주의사항

- ❌ `.env` 파일을 git에 커밋하지 마세요
- ❌ API 키를 코드에 하드코딩하지 마세요
- ✅ 프로덕션에서는 환경변수나 시크릿 관리 도구 사용
- ✅ 음성 파일은 개인정보보호 규정 준수

## 📞 지원

**문제 해결 순서:**
1. `.env` 파일 설정 확인
2. API 키 유효성 확인
3. 오디오 파일 형식 확인 (WAV, MP3, M4A, FLAC 지원)
4. 네트워크 연결 확인

**성능 최적화:**
- 16kHz, 16-bit, Mono 형태의 WAV 파일 권장
- 5분 이하 오디오 파일로 나누어 처리 권장
- 대용량 파일은 비동기 모드 사용

## 📄 라이선스

이 프로젝트는 MIT 라이선스하에 제공됩니다.