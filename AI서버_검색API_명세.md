# AI 서버 검색 API 명세서

백엔드 팀원에게 전달용

---

## 1. AI 서버 정보

### 개발 환경
- **URL**: `http://localhost:8000`
- **프로토콜**: HTTP
- **포트**: 8000

### 프로덕션 환경 (배포 시)
- **URL**: `http://{AI서버IP}:8000`
- 또는 도메인 사용 시: `http://ai-server.your-domain.com`

---

## 2. 검색 API 명세

### 엔드포인트
```
POST /search/semantic
```

### 요청 헤더
```
Content-Type: application/json
```

### 요청 바디
```json
{
  "query": "검색할 텍스트",
  "top_k": 5
}
```

**파라미터:**
- `query` (string, required): 검색 쿼리
  - 예: "예산 관련 회의", "마케팅 전략", "개발 일정"
- `top_k` (integer, optional): 반환할 결과 개수
  - 기본값: 5
  - 범위: 1~100

### 응답 (성공 시)

**상태 코드:** `200 OK`

**응답 바디:**
```json
{
  "query": "예산 관련 회의",
  "total_results": 3,
  "results": [
    {
      "meeting_id": "uuid-string",
      "title": "2024년 Q1 예산 회의",
      "summary": "마케팅 예산 20% 증액, 신제품 출시...",
      "similarity": 0.89
    },
    {
      "meeting_id": "uuid-string",
      "title": "재무 계획 회의",
      "summary": "연간 예산 배분에 대한 논의...",
      "similarity": 0.76
    }
  ]
}
```

**응답 필드:**
- `query` (string): 검색한 쿼리
- `total_results` (integer): 검색된 결과 개수
- `results` (array): 검색 결과 배열
  - `meeting_id` (string): 회의 고유 ID (UUID)
  - `title` (string): 회의 제목
  - `summary` (string): 회의 요약문
  - `similarity` (float): 유사도 점수 (0~1, 높을수록 유사)

### 응답 (실패 시)

**상태 코드:** `500 Internal Server Error`

**응답 바디:**
```json
{
  "detail": "검색 실패: {오류 메시지}"
}
```

---

## 3. 요청 예시

### cURL
```bash
curl -X POST http://localhost:8000/search/semantic \
  -H "Content-Type: application/json" \
  -d '{
    "query": "예산 관련 회의",
    "top_k": 5
  }'
```

### Java (RestTemplate)
```java
RestTemplate restTemplate = new RestTemplate();

String aiServerUrl = "http://localhost:8000/search/semantic";

// 요청 바디 구성
Map<String, Object> requestBody = new HashMap<>();
requestBody.put("query", "예산 관련 회의");
requestBody.put("top_k", 5);

// 응답 타입 정의
class AiSearchResponse {
    public String query;
    public int total_results;
    public List<AiSearchResult> results;
}

class AiSearchResult {
    public String meeting_id;
    public String title;
    public String summary;
    public Double similarity;
}

// API 호출
ResponseEntity<AiSearchResponse> response = restTemplate.postForEntity(
    aiServerUrl,
    requestBody,
    AiSearchResponse.class
);

AiSearchResponse body = response.getBody();
List<AiSearchResult> results = body.results;
```

### Java (WebClient - 비동기)
```java
WebClient webClient = WebClient.create("http://localhost:8000");

Mono<AiSearchResponse> response = webClient
    .post()
    .uri("/search/semantic")
    .contentType(MediaType.APPLICATION_JSON)
    .bodyValue(Map.of(
        "query", "예산 관련 회의",
        "top_k", 5
    ))
    .retrieve()
    .bodyToMono(AiSearchResponse.class);

AiSearchResponse result = response.block();
```

---

## 4. 통합 시나리오

### Backend에서 해야 할 일

**기존 `GET /api/meetings` 엔드포인트 확장:**

1. **파라미터 확인**
   - `semantic=true`이고 `query` 파라미터가 있으면 의미 기반 검색
   - 그 외에는 기존 LIKE 검색

2. **AI 서버로 검색 요청 전송** (semantic=true일 때)
   - POST /search/semantic 호출
   - 사용자 검색 쿼리 전달
   - `top_k`는 페이지 크기(size)로 설정

3. **AI 서버 응답에서 meeting_id 추출**
   - results 배열의 각 항목에서 meeting_id 가져오기

4. **DB에서 해당 회의들 조회**
   - meeting_id로 검색
   - **중요**: 현재 로그인한 사용자의 회의만 필터링

5. **similarity 점수와 함께 페이지네이션 응답 구성**
   - AI 서버 응답의 similarity와 DB 정보 병합
   - 정렬은 similarity 순서 유지
   - 기존과 동일한 페이지네이션 형태 반환

### 예시 플로우

```
사용자: "예산" 검색 + AI 검색 모드
    ↓
Frontend: GET /api/meetings?semantic=true&query=예산&page=1&size=10
    ↓
Backend: semantic=true 확인
    ↓
Backend: POST http://localhost:8000/search/semantic
         { "query": "예산", "top_k": 10 }
    ↓
AI 서버 응답:
{
  "results": [
    { "meeting_id": "uuid1", "similarity": 0.89, "title": "...", "summary": "..." },
    { "meeting_id": "uuid2", "similarity": 0.76, "title": "...", "summary": "..." }
  ]
}
    ↓
Backend: SELECT * FROM meetings
         WHERE id IN ('uuid1', 'uuid2')
         AND user_id = {현재사용자}
    ↓
Backend 응답 (페이지네이션 형태): {
  "content": [
    {
      "meetingId": "uuid1",
      "title": "...",
      "summary": "...",
      "status": "completed",
      "similarity": 0.89,
      "createdAt": "..."
    }
  ],
  "page": 1,
  "size": 10,
  "totalPages": 1
}
```

---

## 5. 주의사항

### 타임아웃 설정
- AI 서버는 OpenAI API를 호출하므로 응답 시간이 2~5초 걸릴 수 있음
- RestTemplate/WebClient 타임아웃 **최소 10초** 권장

```java
SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
factory.setConnectTimeout(10000);  // 10초
factory.setReadTimeout(10000);     // 10초
RestTemplate restTemplate = new RestTemplate(factory);
```

### 에러 처리
- AI 서버 다운 시: Connection refused
- OpenAI API 키 오류 시: 500 에러 ("OpenAI API 키가 설정되지 않았습니다")
- 임베딩 파일 없을 시: 빈 배열 반환 (`"results": []`)

### 보안
- 현재 AI 서버는 인증 없음 (내부 네트워크 전용)
- 프로덕션 배포 시:
  - VPC 내부 통신 권장
  - 또는 API Key 인증 추가 필요

---

## 6. 환경 설정

### application.yml
```yaml
ai:
  server:
    url: http://localhost:8000  # 개발
    # url: http://ai-server:8000  # 프로덕션
    timeout: 10000  # 10초
```

### RestTemplate Bean 설정
```java
@Configuration
public class RestTemplateConfig {

    @Value("${ai.server.timeout:10000}")
    private int timeout;

    @Bean
    public RestTemplate restTemplate() {
        SimpleClientHttpRequestFactory factory = new SimpleClientHttpRequestFactory();
        factory.setConnectTimeout(timeout);
        factory.setReadTimeout(timeout);
        return new RestTemplate(factory);
    }
}
```

---

## 7. 테스트

### AI 서버 상태 확인
```bash
curl http://localhost:8000/health
```

**응답:**
```json
{
  "status": "healthy",
  "timestamp": "2025-10-22T12:00:00"
}
```

### 검색 테스트
```bash
curl -X POST http://localhost:8000/search/semantic \
  -H "Content-Type: application/json" \
  -d '{"query": "테스트", "top_k": 3}'
```

### 임베딩 파일 확인
AI 서버에서:
```bash
cd C:/project/meeting-one-line/meeting-one-line-ai/stt-ncp
ls embeddings/
# meeting_*.json 파일들이 있어야 함
```

---

## 8. FAQ

**Q: 검색 결과가 없으면?**
- A: `"results": []` 빈 배열 반환. 에러 아님.

**Q: meeting_id가 DB에 없으면?**
- A: Backend에서 필터링 후 제외. 최종 결과에서 누락됨.

**Q: AI 서버가 다운되면?**
- A: Connection refused 에러. Backend에서 catch 후 적절한 에러 메시지 반환.

**Q: 유사도 점수가 0에 가까우면?**
- A: 관련성 낮음. 필요시 threshold (예: 0.5 이상만) 설정 가능.

---

## 9. 연락처

질문이나 문제 발생 시:
- AI 서버 담당: [본인 이름/연락처]
- 임베딩 관련 이슈: [본인 이름/연락처]
