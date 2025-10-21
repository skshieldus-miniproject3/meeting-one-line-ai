"""임베딩 저장 및 검색 관리"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class EmbeddingManager:
    """임베딩 저장 및 검색을 관리하는 클래스"""

    def __init__(self, embeddings_dir: str = "embeddings"):
        """
        EmbeddingManager 초기화

        Args:
            embeddings_dir: 임베딩 파일을 저장할 디렉토리
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.embeddings_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)

    def save_meeting_embedding(
        self,
        meeting_id: str,
        title: str,
        summary: str,
        embedding: List[float],
        segments: Optional[List[Dict]] = None
    ) -> None:
        """
        회의록 임베딩을 저장합니다.

        Args:
            meeting_id: 회의 ID
            title: 회의 제목
            summary: 회의 요약
            embedding: 임베딩 벡터
            segments: 세그먼트별 임베딩 (선택사항)
        """
        data = {
            "meeting_id": meeting_id,
            "title": title,
            "summary": summary,
            "embedding": embedding,
            "segments": segments or []
        }

        file_path = self.embeddings_dir / f"meeting_{meeting_id}.json"
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"임베딩 저장 완료: {file_path}")

    def load_all_embeddings(self) -> List[Dict]:
        """
        저장된 모든 회의록 임베딩을 로드합니다.

        Returns:
            회의록 임베딩 데이터 리스트
        """
        embeddings = []

        for file_path in self.embeddings_dir.glob("meeting_*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    embeddings.append(data)
            except Exception as e:
                self.logger.error(f"임베딩 로드 실패 ({file_path}): {e}")

        self.logger.info(f"임베딩 {len(embeddings)}개 로드 완료")
        return embeddings

    def search_similar_meetings(
        self,
        query_embedding: List[float],
        top_k: int = 5
    ) -> List[Dict]:
        """
        쿼리 임베딩과 유사한 회의록을 검색합니다.

        Args:
            query_embedding: 검색 쿼리의 임베딩 벡터
            top_k: 반환할 결과 개수

        Returns:
            유사도 높은 순으로 정렬된 회의록 정보 리스트
            [
                {
                    "meeting_id": "...",
                    "title": "...",
                    "summary": "...",
                    "similarity": 0.89
                }
            ]
        """
        all_embeddings = self.load_all_embeddings()

        if not all_embeddings:
            self.logger.warning("저장된 임베딩이 없습니다")
            return []

        results = []

        for meeting_data in all_embeddings:
            meeting_embedding = meeting_data.get("embedding")
            if not meeting_embedding:
                continue

            # 코사인 유사도 계산
            similarity = cosine_similarity(
                [query_embedding],
                [meeting_embedding]
            )[0][0]

            results.append({
                "meeting_id": meeting_data["meeting_id"],
                "title": meeting_data["title"],
                "summary": meeting_data["summary"],
                "similarity": float(similarity)
            })

        # 유사도 높은 순으로 정렬
        results.sort(key=lambda x: x["similarity"], reverse=True)

        return results[:top_k]

    def delete_meeting_embedding(self, meeting_id: str) -> bool:
        """
        회의록 임베딩을 삭제합니다.

        Args:
            meeting_id: 삭제할 회의 ID

        Returns:
            삭제 성공 여부
        """
        file_path = self.embeddings_dir / f"meeting_{meeting_id}.json"

        if file_path.exists():
            file_path.unlink()
            self.logger.info(f"임베딩 삭제 완료: {meeting_id}")
            return True
        else:
            self.logger.warning(f"임베딩 파일 없음: {meeting_id}")
            return False

    def get_stats(self) -> Dict:
        """
        저장된 임베딩 통계 정보를 반환합니다.

        Returns:
            통계 정보 딕셔너리
        """
        all_embeddings = self.load_all_embeddings()

        return {
            "total_meetings": len(all_embeddings),
            "storage_path": str(self.embeddings_dir.absolute()),
            "meetings": [
                {
                    "meeting_id": e["meeting_id"],
                    "title": e["title"]
                }
                for e in all_embeddings
            ]
        }
