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

    def summarize(self, transcript: str) -> str:
        """
        대화록 텍스트를 핵심 요약합니다.

        Args:
            transcript: 대화록 텍스트 (화자별로 구분된 형태)

        Returns:
            요약된 텍스트

        Raises:
            ReportGeneratorError: 요약 생성 실패 시
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
        대화록을 바탕으로 공식 회의록 문서를 생성합니다.

        Args:
            transcript: 대화록 텍스트 (화자별로 구분된 형태)

        Returns:
            공식 회의록 문서

        Raises:
            ReportGeneratorError: 회의록 생성 실패 시
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
        대화록에서 액션 아이템을 추출합니다.

        Args:
            transcript: 대화록 텍스트

        Returns:
            액션 아이템 목록

        Raises:
            ReportGeneratorError: 액션 아이템 추출 실패 시
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
        회의 분위기와 참석자들의 감정을 분석합니다.

        Args:
            transcript: 대화록 텍스트

        Returns:
            감정 분석 결과

        Raises:
            ReportGeneratorError: 감정 분석 실패 시
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
        회의 내용을 바탕으로 후속 질문을 생성합니다.

        Args:
            transcript: 대화록 텍스트

        Returns:
            후속 질문 목록

        Raises:
            ReportGeneratorError: 질문 생성 실패 시
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