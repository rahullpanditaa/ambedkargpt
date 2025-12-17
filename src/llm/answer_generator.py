"""
LLM-based Answer Generation for SemRAG.

This module implements Step 4 of the SemRAG assignment:
LLM integration and response generation.

It takes the output of Global Graph RAG (Equation 5),
constructs a grounded prompt using retrieved community
summaries and chunk-level evidence, and generates a final
answer using a local LLM via Ollama.

This completes the end-to-end SemRAG pipeline:
Query → Retrieval → Prompt → LLM Answer
"""

import json
from pathlib import Path
import ollama

from src.retrieval.global_search import GlobalGraphRAG
from src.utils.constants import (
    PROCESSED_DATA_DIR_PATH,
    FINAL_ANSWER_PATH,
    ANSWER_GENERATOR_PROMPT
)


class SemRAGAnswerGenerator:
    """
    End-to-end SemRAG answer generator.

    This class orchestrates:
    - Global Graph RAG retrieval
    - Prompt construction
    - LLM-based answer generation

    It represents the final online stage of the SemRAG system.
    """

    def __init__(self, llm_model: str = "mistral", model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the answer generator.

        Args:
            llm_model (str): Ollama model name
                             (e.g., mistral, llama3:8b)
            embedding_model (str): SentenceTransformer model
        """

        self.llm_model = llm_model
        self.global_rag = GlobalGraphRAG(model_name=model_name)

        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)

    def _build_prompt(self, user_query: str, retrieved_items: list[dict]) -> str:
        """
        Construct an LLM prompt using retrieved communities and chunks.

        The prompt is explicitly grounded:
        - No external knowledge allowed
        - The LLM must rely only on retrieved evidence

        Args:
            user_query (str): User question
            retrieved_items (list[dict]): Output from GlobalGraphRAG

        Returns:
            str: Fully formatted prompt
        """

        # unique community summaries
        seen_communities = set()
        community_summaries = []

        for item in retrieved_items:
            cid = item["community_id"]
            if cid not in seen_communities:
                seen_communities.add(cid)
                community_summaries.append(
                    f"- Community {cid}: {item.get('summary', '')}"
                )

        # chunk texts
        chunk_texts = []
        for item in retrieved_items:
            chunk_texts.append(item["chunk_text"])

        community_block = "\n".join(community_summaries)
        chunks_block = "\n\n---\n\n".join(chunk_texts)

        prompt = ANSWER_GENERATOR_PROMPT.replace("{USER_QUERY}", user_query)
        prompt = prompt.replace("{COMMUNITY_BLOCK}", community_block)
        prompt = prompt.replace("{CHUNKS_BLOCK}", chunks_block)

        return prompt

    def generate_answer(self, user_query: str) -> dict:
        """
        Generate a final answer for a user query using SemRAG.

        Pipeline:
        1. Run Global Graph RAG
        2. Build grounded prompt
        3. Invoke local LLM via Ollama
        4. Save and return answer

        Args:
            user_query (str): User question

        Returns:
            dict: Answer object containing:
                - query
                - answer
                - retrieved evidence
        """

        # retrieval (eq 5 -> eq 4)
        retrieved_items = self.global_rag.global_search(user_query)

        if not retrieved_items:
            return {
                "query": user_query,
                "answer": "No relevant information was found.",
                "evidence": []
            }

        # build prompt
        prompt = self._build_prompt(
            user_query=user_query,
            retrieved_items=retrieved_items
        )

        # call llm
        response = ollama.generate(
            model=self.llm_model,
            prompt=prompt
        )

        answer_text = response.response.strip()

        output = {
            "query": user_query,
            "answer": answer_text,
            "evidence": retrieved_items
        }

        with open(FINAL_ANSWER_PATH, "w") as f:
            json.dump(output, f, indent=2)

        return output


def answer_query_command(query: str):
    """
    CLI-style helper for answering a single query.

    Args:
        query (str): User question
    """

    generator = SemRAGAnswerGenerator()
    result = generator.generate_answer(query)

    print("\n=== QUERY ===")
    print(result["query"])

    print("\n=== ANSWER ===")
    print(result["answer"])
