# 임베딩 검색 API 설계 (Backend 수정 최소화)

## 목표
- **DB 스키마 수정 없음**
- **Backend 코드 수정 최소화** (새 엔드포인트 1개만 추가)
- **AI 서버 기존 검색 기능 활용**

---

## 1. 현재 구조

### 기존 검색 API (이미 있음)
**엔드포인트:** `GET /api/meetings`

**파라미터:**
- `page`: 페이지 번호 (기본 1)
- `size`: 페이지 크기 (기본 10)
- `keyword`: 제목 또는 요약에서 텍스트 검색 (LIKE 쿼리)
- `title`: 제목 검색
- `summary`: 요약 검색
- `status`: 상태 필터

**응답:**
```json
{
  "content": [
    {
      "meetingId": "uuid",
      "title": "회의 제목",
      "status": "completed",
      "summary": "요약문",
      "createdAt": "2025-10-20T15:00:00"
    }
  ],
  "page": 1,
  "size": 10,
  "totalPages": 5
}
```

---

## 2. 의미 기반 검색 기능 추가 (기존 엔드포인트 확장)

### Backend 기존 엔드포인트 확장

**엔드포인트:** `GET /api/meetings` (기존 엔드포인트 그대로)

**기존 파라미터 (그대로):**
- `page`: 페이지 번호 (기본 1)
- `size`: 페이지 크기 (기본 10)
- `keyword`: 제목 또는 요약에서 텍스트 검색 (LIKE 쿼리)
- `title`: 제목 검색
- `summary`: 요약 검색
- `status`: 상태 필터

**새로 추가할 파라미터:**
- `semantic` (optional): 의미 기반 검색 활성화 (true/false, 기본: false)
- `query` (optional): 의미 기반 검색 쿼리 (semantic=true일 때 필수)

**요청 예시 (의미 기반 검색):**
```
GET /api/meetings?semantic=true&query=예산 관련 회의&page=1&size=10
Authorization: Bearer {token}
```

**응답 (페이지네이션 형태 - 기존과 동일):**
```json
{
  "content": [
    {
      "meetingId": "uuid1",
      "title": "2024년 Q1 예산 회의",
      "summary": "마케팅 예산 20% 증액...",
      "status": "completed",
      "similarity": 0.89,
      "createdAt": "2025-10-20T15:00:00"
    },
    {
      "meetingId": "uuid2",
      "title": "재무 계획 회의",
      "summary": "연간 예산 배분...",
      "status": "completed",
      "similarity": 0.76,
      "createdAt": "2025-10-18T10:00:00"
    }
  ],
  "page": 1,
  "size": 10,
  "totalPages": 1
}
```

**참고:**
- `semantic=false` 또는 파라미터 없을 때: 기존 LIKE 검색
- `semantic=true`일 때: AI 서버 임베딩 검색, `similarity` 필드 포함

---

## 3. Backend 구현 (최소 수정)

### MeetingController.java 수정 (기존 엔드포인트에 파라미터 추가)

```java
@GetMapping  // 기존 GET /meetings 엔드포인트
@Operation(
    summary = "회의록 목록 조회",
    description = "페이지 단위 조회, LIKE 검색, 의미 기반 검색 지원"
)
public ResponseEntity<Page<MeetingDto>> getMeetings(
    @AuthenticationPrincipal UUID userId,
    @RequestParam(defaultValue = "1") int page,
    @RequestParam(defaultValue = "10") int size,
    @RequestParam(required = false) String keyword,
    @RequestParam(required = false) String title,
    @RequestParam(required = false) String summary,
    @RequestParam(required = false) String status,
    // 새로 추가되는 파라미터
    @RequestParam(defaultValue = "false") boolean semantic,
    @RequestParam(required = false) String query
) {
    Page<MeetingDto> meetings;

    if (semantic && query != null) {
        // 의미 기반 검색
        meetings = meetingService.semanticSearch(userId, query, page, size);
    } else {
        // 기존 LIKE 검색
        meetings = meetingService.getMeetings(userId, page, size, keyword, title, summary, status);
    }

    return ResponseEntity.ok(meetings);
}
```

### MeetingService.java에 추가할 메서드

```java
public Page<MeetingDto> semanticSearch(UUID userId, String query, int page, int size) {
    // 1. AI 서버 호출
    String aiServerUrl = env.getProperty("ai.server.url") + "/search/semantic";

    Map<String, Object> requestBody = new HashMap<>();
    requestBody.put("query", query);
    requestBody.put("top_k", size);  // 페이지 크기만큼 요청

    // RestTemplate으로 AI 서버 호출
    ResponseEntity<AiSearchResponse> aiResponse = restTemplate.postForEntity(
        aiServerUrl,
        requestBody,
        AiSearchResponse.class
    );

    // 2. AI 서버 응답에서 meeting_id 추출
    List<String> meetingIds = aiResponse.getBody().getResults().stream()
        .map(result -> result.getMeetingId())
        .collect(Collectors.toList());

    if (meetingIds.isEmpty()) {
        return Page.empty(PageRequest.of(page - 1, size));
    }

    // 3. DB에서 해당 회의들 조회 (본인 것만)
    List<MeetingEntity> meetings = meetingRepository.findAllByIdInAndUserId(meetingIds, userId);

    // 4. similarity 점수를 Map으로 저장
    Map<String, Double> similarityMap = aiResponse.getBody().getResults().stream()
        .collect(Collectors.toMap(
            AiSearchResult::getMeetingId,
            AiSearchResult::getSimilarity
        ));

    // 5. MeetingDto로 변환하면서 similarity 추가
    List<MeetingDto> content = meetings.stream()
        .map(meeting -> {
            MeetingDto dto = MeetingDto.from(meeting);
            dto.setSimilarity(similarityMap.get(meeting.getId().toString()));
            return dto;
        })
        .sorted((a, b) -> Double.compare(b.getSimilarity(), a.getSimilarity()))  // similarity 내림차순
        .collect(Collectors.toList());

    // 6. Page 객체로 반환 (페이지네이션 형태 유지)
    return new PageImpl<>(content, PageRequest.of(page - 1, size), content.size());
}
```

### MeetingDto.java 수정 (similarity 필드 추가)

```java
@Getter
@Setter
public class MeetingDto {
    private UUID meetingId;
    private String title;
    private String status;
    private String summary;
    private LocalDateTime createdAt;

    // 의미 기반 검색 시에만 사용
    private Double similarity;  // 새로 추가

    public static MeetingDto from(MeetingEntity entity) {
        MeetingDto dto = new MeetingDto();
        dto.setMeetingId(entity.getId());
        dto.setTitle(entity.getTitle());
        dto.setStatus(entity.getStatus());
        dto.setSummary(entity.getSummary());
        dto.setCreatedAt(entity.getCreatedAt());
        return dto;
    }
}
```

### application.yml에 AI 서버 URL 추가

```yaml
ai:
  server:
    url: http://localhost:8000  # 또는 AI 서버 실제 URL
```

---

## 4. AI 서버 (이미 구현됨 ✅)

**엔드포인트:** `POST /search/semantic`

**현재 코드 (server.py):**
```python
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

        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"검색 실패: {str(e)}")
```

**임베딩 파일:** `embeddings/meeting_{id}.json`
- 회의록 분석 시 자동 생성 (analyze.py)
- 또는 수동 저장 (search.py)

---

## 5. Frontend 구현

### API 호출 함수 (lib/api.ts) - 수정 없음!

기존 `GET /meetings` 호출 함수를 그대로 사용하되, 파라미터만 추가:

```typescript
// 기존 함수 활용 (변경 없음)
async getMeetings(params: {
  page?: number,
  size?: number,
  keyword?: string,
  // 새로 추가되는 파라미터
  semantic?: boolean,
  query?: string
}) {
  const queryString = new URLSearchParams(params as any).toString()
  return this.get(`/meetings?${queryString}`)
}
```

### Hook (hooks/useMeetings.ts) - 기존 hook 활용

```typescript
// 기존 useMeetings hook에 의미 기반 검색 옵션만 추가
export function useMeetings() {
  const [meetings, setMeetings] = useState([])
  const [isLoading, setIsLoading] = useState(false)

  // 기존 함수
  const fetchMeetings = async (params: { page, size, keyword?, ... }) => {
    const response = await apiClient.getMeetings(params)
    setMeetings(response.data.content)
  }

  // 의미 기반 검색 (새 함수)
  const semanticSearch = async (query: string, page = 1, size = 10) => {
    const response = await apiClient.getMeetings({
      semantic: true,
      query,
      page,
      size
    })
    setMeetings(response.data.content)
  }

  return { meetings, isLoading, fetchMeetings, semanticSearch }
}
```

### UI 컴포넌트 (app/meetings/page.tsx)

```typescript
const { meetings, semanticSearch, fetchMeetings } = useMeetings()
const [searchMode, setSearchMode] = useState<'keyword' | 'semantic'>('keyword')

const handleSearch = (e) => {
  e.preventDefault()

  if (searchMode === 'semantic') {
    // 의미 기반 검색
    semanticSearch(searchQuery)
  } else {
    // 기존 키워드 검색
    fetchMeetings({ keyword: searchQuery })
  }
}

// 검색 모드 전환 버튼
<div>
  <button onClick={() => setSearchMode('keyword')}>키워드 검색</button>
  <button onClick={() => setSearchMode('semantic')}>AI 검색</button>
</div>

// 검색창 (기존과 동일)
<input
  type="text"
  placeholder={
    searchMode === 'semantic'
      ? "의미 검색 (예: 예산 관련 회의)"
      : "키워드 검색"
  }
  value={searchQuery}
  onChange={(e) => setSearchQuery(e.target.value)}
/>
<button onClick={handleSearch}>검색</button>

// 결과 표시 (기존 코드 그대로, similarity만 추가 표시)
{meetings.map(meeting => (
  <div key={meeting.meetingId}>
    <h3>{meeting.title}</h3>
    <p>{meeting.summary}</p>
    {meeting.similarity && (
      <span>관련도: {(meeting.similarity * 100).toFixed(1)}%</span>
    )}
  </div>
))}
```

---

## 6. 데이터 플로우

```
사용자: "예산 관련 회의" 입력 + AI 검색 모드 선택
    ↓
Frontend: GET /api/meetings?semantic=true&query=예산 관련 회의&page=1&size=10
    ↓
Backend (MeetingController):
    ↓
    semantic=true 확인
    ↓
    1. AI 서버 호출: POST http://ai-server:8000/search/semantic
       Request: { "query": "예산 관련 회의", "top_k": 10 }
    ↓
    AI 서버:
       1. 쿼리 → 임베딩 변환 (OpenAI API)
       2. embeddings/*.json 파일들과 코사인 유사도 계산
       3. 상위 10개 반환
    ↓
    AI 서버 응답: {
      "results": [
        { "meeting_id": "uuid1", "title": "...", "similarity": 0.89 },
        { "meeting_id": "uuid2", "title": "...", "similarity": 0.76 }
      ]
    }
    ↓
    2. meeting_id로 DB 조회 (본인 회의만)
    3. similarity 점수 병합
    4. 페이지네이션 형태로 구성
    ↓
Backend 응답: {
  "content": [
    {
      "meetingId": "uuid1",
      "title": "2024년 Q1 예산 회의",
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
    ↓
Frontend: 기존 회의록 목록과 동일한 UI에 표시 (similarity 점수 추가)
```

---

## 7. 환경 변수 설정

### AI 서버 (.env)

이미 설정되어 있음:
```
OPENAI_API_KEY=sk-...
```

### Backend (application.yml)

```yaml
ai:
  server:
    url: http://localhost:8000  # 개발 환경
    # url: http://ai-server.prod:8000  # 프로덕션
```

---

## 8. 필요한 작업 요약

### Backend 팀원 작업

1. **MeetingController.java** - 기존 GET /meetings에 파라미터 2개 추가 (약 5줄)
   - `semantic` (boolean)
   - `query` (String)
2. **MeetingService.java** - semanticSearch 메서드 추가 (약 45줄)
3. **MeetingDto.java** - similarity 필드 1개 추가 (1줄)
4. **application.yml** - AI 서버 URL 설정 (3줄)
5. **RestTemplate Bean 설정** (이미 있으면 스킵)

**총 코드:** 약 55줄 (엔드포인트 추가 없이 기존 확장!)

### Frontend 팀원 작업

1. **hooks/useMeetings.ts** - semanticSearch 함수 추가 (약 10줄)
2. **app/meetings/page.tsx** - 검색 모드 전환 버튼 추가 (약 15줄)
3. **lib/api.ts** - 수정 없음 (기존 getMeetings 활용)

**총 코드:** 약 25줄 (기존 UI/Hook 활용!)

### AI 서버 작업

✅ **이미 완료됨!** 추가 작업 없음.

---

## 9. 장점

✅ **DB 스키마 수정 불필요** - embedding 컬럼 추가 안 함
✅ **Backend 수정 최소화** - 새 엔드포인트 추가 없이 기존 확장
✅ **API 일관성 유지** - 기존 `/meetings` 엔드포인트 그대로 사용
✅ **페이지네이션 호환** - 기존 응답 형태 그대로 유지
✅ **Frontend 코드 재사용** - 기존 useMeetings hook 그대로 활용
✅ **임베딩 저장소 유지** - AI 서버 파일 시스템 그대로 사용
✅ **확장 용이** - 나중에 벡터 DB로 전환 쉬움
✅ **권한 검증** - Backend에서 본인 회의만 조회
✅ **선택적 사용** - semantic=false면 기존 LIKE 검색 그대로

---

## 10. 테스트 시나리오

1. **임베딩 파일 존재 확인**
   ```bash
   ls embeddings/
   # meeting_test_001.json, meeting_test_002.json 등
   ```

2. **AI 서버 검색 테스트**
   ```bash
   curl -X POST http://localhost:8000/search/semantic \
     -H "Content-Type: application/json" \
     -d '{"query": "예산 관련 회의", "top_k": 5}'
   ```

3. **Backend API 테스트**
   ```bash
   # 의미 기반 검색
   curl "http://localhost:8080/api/meetings?semantic=true&query=예산 관련 회의&page=1&size=10" \
     -H "Authorization: Bearer {token}"

   # 기존 키워드 검색 (여전히 작동)
   curl "http://localhost:8080/api/meetings?keyword=예산&page=1&size=10" \
     -H "Authorization: Bearer {token}"
   ```

4. **Frontend 통합 테스트**
   - 검색창에 "예산" 입력
   - 결과 목록 표시 확인
   - 유사도 점수 확인

---

## 11. 향후 확장

### 단계 1 (현재)
- AI 서버 파일 시스템 사용
- 소규모 데이터 (< 1000건)

### 단계 2 (데이터 증가 시)
- PostgreSQL + pgvector
- 빠른 벡터 검색

### 단계 3 (대규모 시)
- Pinecone / Weaviate
- 분산 벡터 검색
