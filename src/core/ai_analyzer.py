"""OpenAI를 이용한 회의록 요약 및 문서 생성"""

import os
import logging
from typing import Optional
from openai import OpenAI


class ReportGeneratorError(Exception):
    """Report Generator 오류"""
    pass


class ReportGenerator:
    """OpenAI LLM을 이용한 요약 및 문서 생성기"""

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o-mini"):
        """
        ReportGenerator 초기화

        Args:
            api_key: OpenAI API 키 (None이면 환경변수에서 로드)
            model: 사용할 OpenAI 모델 (기본: gpt-4o-mini)
        """
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ReportGeneratorError("OPENAI_API_KEY가 설정되지 않았습니다. .env 파일을 확인해주세요.")

        self.model = model
        self.client = OpenAI(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)

    def _call_llm(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        """
        LLM 호출 공통 함수

        Args:
            system_prompt: 시스템 프롬프트
            user_prompt: 사용자 프롬프트
            temperature: 창의성 수준 (0.0-1.0)

        Returns:
            LLM 응답 텍스트

        Raises:
            ReportGeneratorError: API 호출 실패 시
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=temperature,
                max_tokens=4000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            self.logger.error(f"OpenAI API 호출 오류: {e}")
            raise ReportGeneratorError(f"LLM 호출 중 오류가 발생했습니다: {e}")

    # --- [기존 5개 기능] ---

    def summarize(self, transcript: str) -> str:
        """
        대화록 텍스트를 핵심 요약합니다. (기존 기능)
        """
        self.logger.info("대화록 요약 생성 중...")

        system_prompt = """당신은 회의록을 전문적으로 요약하는 어시스턴트입니다.
        다음 지침을 따라 요약해주세요:
        1. 핵심 내용을 3-5개의 글머리 기호로 정리
        2. 중요한 결정사항이나 합의사항 강조
        3. 구체적인 수치나 날짜가 있다면 포함
        4. 간결하고 명확한 표현 사용"""

        user_prompt = f"""
다음은 화자별로 구분된 회의록 텍스트입니다.
이 대화의 핵심 내용을 3-5줄의 글머리 기호(•)로 요약해 주세요.

--- [회의록] ---
{transcript}
----------------

요약:
"""
        return self._call_llm(system_prompt, user_prompt)

    def generate_meeting_notes(self, transcript: str) -> str:
        """
        대화록을 바탕으로 공식 회의록 문서를 생성합니다. (기존 기능)
        """
        self.logger.info("공식 회의록 문서 생성 중...")

        system_prompt = """당신은 회의 내용을 바탕으로 공식적인 회의록(Meeting Minutes)을 작성하는 전문 비서입니다.
        다음 형식을 반드시 준수하여 문서를 생성해 주세요:

        ## 회의록

        ### 1. 주요 안건
        - (대화에서 논의된 핵심 주제들을 나열)

        ### 2. 결정 사항
        - (대화를 통해 합의되거나 결정된 내용들)

        ### 3. 향후 조치 사항 (Action Items)
        - (특정 담당자에게 할당된 업무나 향후 계획)

        ### 4. 기타 사항
        - (추가적인 논의 사항이나 참고 내용)

        전문적이고 공식적인 문체를 사용하며, 구체적인 내용을 포함해주세요."""

        user_prompt = f"""
다음 회의록 텍스트를 바탕으로 공식 회의록 문서를 작성해 주세요.

--- [회의록] ---
{transcript}
----------------

공식 회의록:
"""
        return self._call_llm(system_prompt, user_prompt)

    def generate_action_items(self, transcript: str) -> str:
        """
        대화록에서 액션 아이템을 추출합니다. (기존 기능)
        """
        self.logger.info("액션 아이템 추출 중...")

        system_prompt = """당신은 회의 내용에서 실행 가능한 액션 아이템을 추출하는 전문가입니다.
        다음 기준에 따라 액션 아이템을 식별해주세요:
        1. 구체적인 행동이 필요한 항목
        2. 담당자가 명시되거나 추정 가능한 항목
        3. 기한이 있거나 우선순위가 높은 항목"""

        user_prompt = f"""
다음 회의록에서 액션 아이템을 추출해주세요.
각 항목에 대해 [담당자] 작업내용 형태로 정리해주세요.

--- [회의록] ---
{transcript}
----------------

액션 아이템:
"""
        return self._call_llm(system_prompt, user_prompt)

    def analyze_sentiment(self, transcript: str) -> str:
        """
        회의 분위기와 참석자들의 감정을 분석합니다. (기존 기능)
        """
        self.logger.info("회의 분위기 분석 중...")

        system_prompt = """당신은 회의 분위기와 참석자들의 감정을 분석하는 전문가입니다.
        대화의 톤, 의견 충돌, 합의 정도, 전반적인 분위기를 객관적으로 분석해주세요."""

        user_prompt = f"""
다음 회의록의 분위기와 참석자들의 감정을 분석해주세요.

--- [회의록] ---
{transcript}
----------------

분석 결과:
"""
        return self._call_llm(system_prompt, user_prompt, temperature=0.1)

    def generate_follow_up_questions(self, transcript: str) -> str:
        """
        회의 내용을 바탕으로 후속 질문을 생성합니다. (기존 기능)
        """
        self.logger.info("후속 질문 생성 중...")

        system_prompt = """당신은 회의 내용을 분석하여 후속 논의가 필요한 질문들을 생성하는 전문가입니다.
        미해결 이슈, 추가 검토가 필요한 사항, 명확화가 필요한 내용을 중심으로 질문을 만들어주세요."""

        user_prompt = f"""
다음 회의록을 분석하여 후속 회의나 개별 논의에서 다뤄야 할 질문들을 생성해주세요.

--- [회의록] ---
{transcript}
----------------

후속 질문:
"""
        return self._call_llm(system_prompt, user_prompt)

    # --- [신규 기능 6개 (Git + Local Merge)] ---

    def extract_keywords(self, transcript: str) -> str:
        """
        대화록에서 핵심 키워드를 추출합니다. (Git 'e319...' 버전 기준)
        """
        self.logger.info("핵심 키워드 추출 중...")

        system_prompt = """당신은 텍스트에서 핵심 키워드를 추출하는 전문가입니다.
        다음 기준에 따라 키워드를 추출해주세요:
        1. 가장 중요하고 반복적으로 언급되는 개념
        2. 고유명사 (회사명, 제품명, 인명, 지명 등)
        3. 핵심 주제어
        4. 중요 수치나 날짜
        5. 10-15개 정도의 키워드로 제한"""

        user_prompt = f"""
다음 회의록에서 핵심 키워드를 추출해주세요.
카테고리별로 구분하여 정리해주세요:

**주요 주제어**:
- (핵심 주제와 관련된 키워드)

**고유명사**:
- (회사명, 제품명, 인명, 지명)

**중요 수치/날짜**:
- (의미 있는 숫자, 날짜, 기간)

--- [회의록] ---
{transcript}
----------------

키워드 추출:
"""
        return self._call_llm(system_prompt, user_prompt)

    def classify_topics(self, transcript: str) -> str:
        """
        대화록의 주제를 분류하고 카테고리화합니다. (Git 'e319...' 버전 신규)
        """
        self.logger.info("주제 분류 중...")

        system_prompt = """당신은 회의 내용을 주제별로 분류하는 전문가입니다.
        대화록을 읽고 다음 작업을 수행해주세요:
        1. 논의된 주요 주제들을 식별
        2. 각 주제에 대한 중요도 평가
        3. 주제 간 연관관계 파악
        4. 각 주제별 논의 비중 추정"""

        user_prompt = f"""
다음 회의록의 주제를 분류하고 분석해주세요.

다음 형식으로 작성해주세요:

**주요 주제 분류**:
1. [주제명] (중요도: 높음/중간/낮음)
   - 논의 내용 요약
   - 전체 대화에서 차지하는 비중: XX%

**주제 간 연관관계**:
- (주제들이 어떻게 연결되는지 설명)

**우선순위 순서**:
1. (가장 중요한 주제부터 나열)

--- [회의록] ---
{transcript}
----------------

주제 분류:
"""
        return self._call_llm(system_prompt, user_prompt)

    def analyze_by_speaker(self, transcript: str) -> str:
        """
        발언자별로 내용을 요약하고 분석합니다. (Git 'e319...' 버전 신규)
        """
        self.logger.info("발언자별 분석 중...")

        system_prompt = """당신은 회의 참석자들의 발언을 분석하는 전문가입니다.
        각 화자별로 다음을 분석해주세요:
        1. 주요 발언 내용 요약
        2. 전문 분야 또는 역할 추정
        3. 의견의 방향성 (찬성/반대/중립)
        4. 주도성 정도 (적극적/보통/소극적)
        5. 핵심 기여 사항"""

        user_prompt = f"""
다음 회의록에서 각 화자별로 발언을 분석해주세요.

각 화자마다 다음 형식으로 작성해주세요:

**화자 [번호]**:
- **주요 발언 내용**: (핵심 내용 2-3줄 요약)
- **추정 역할**: (기술팀, 관리자, 마케팅 등)
- **의견 방향성**: (긍정적/부정적/중립적/혼합)
- **참여도**: (적극적/보통/소극적)
- **핵심 기여**: (이 사람이 회의에 기여한 핵심 내용)

--- [회의록] ---
{transcript}
----------------

발언자별 분석:
"""
        return self._call_llm(system_prompt, user_prompt, temperature=0.2)

    def classify_meeting_type(self, transcript: str) -> str:
        """
        회의 유형을 분류합니다. (Git 'e319...' 버전 신규)
        """
        self.logger.info("회의 유형 분류 중...")

        system_prompt = """당신은 회의 유형을 분석하고 분류하는 전문가입니다.
        다음 카테고리 중에서 가장 적합한 유형을 선택하고 그 근거를 설명해주세요:

        **회의 유형 카테고리**:
        1. 프로젝트 킥오프 - 새로운 프로젝트 시작
        2. 정기 진행 회의 - 주간/월간 정기 미팅
        3. 브레인스토밍 - 아이디어 발산 세션
        4. 의사결정 회의 - 중요한 결정을 내리는 회의
        5. 문제 해결 회의 - 특정 문제 해결 논의
        6. 보고/공유 회의 - 정보 전달 및 공유
        7. 계획 수립 회의 - 전략/계획 수립
        8. 검토/피드백 회의 - 결과물 검토
        9. 교육/학습 세션 - 지식 전달
        10. 기타 - 위 카테고리에 해당하지 않는 경우"""

        user_prompt = f"""
다음 회의록을 분석하여 회의 유형을 분류해주세요.

다음 형식으로 작성해주세요:

**회의 유형**: [선택한 카테고리]

**주 카테고리 신뢰도**: XX%

**부 카테고리** (해당되는 경우):
- [추가 카테고리] (XX%)

**분류 근거**:
- (이 유형으로 분류한 이유를 구체적으로 설명)

**회의 특징**:
- 참여자 구성: (경영진/실무자/혼합)
- 진행 방식: (구조적/자유로운/혼합)
- 의사결정 여부: (있음/없음/보류)

--- [회의록] ---
{transcript}
----------------

회의 유형 분류:
"""
        return self._call_llm(system_prompt, user_prompt, temperature=0.1)

    def summarize_by_speaker(self, transcript: str) -> str:
        """
        대화록을 바탕으로 화자별 주요 발언을 요약합니다. (로컬 DB 버전 신규)
        (analyze_by_speaker보다 간단한 요약본)
        """
        self.logger.info("화자별 간단 요약 생성 중...")

        system_prompt = """당신은 회의록을 바탕으로 각 화자의 주요 입장이나 발언을 요약하는 전문가입니다.
        [화자 N] 형태로 구분된 대화록을 보고, 각 화자별로 핵심 주장을 1~2줄로 요약해주세요.
        모든 화자를 포함할 필요는 없으며, 중요 발언을 한 화자 중심으로 정리해주세요.

        [출력 형식]
        - [화자 1]: (화자 1의 핵심 발언 요약)
        - [화자 2]: (화자 2의 핵심 발언 요약)
        """

        user_prompt = f"""
다음 회의록을 화자별로 요약해주세요.

--- [회의록] ---
{transcript}
----------------

화자별 요약:
"""
        return self._call_llm(system_prompt, user_prompt)