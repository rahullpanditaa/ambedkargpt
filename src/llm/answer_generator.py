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
import ollama
from pathlib import Path

from src.utils.constants import (
    PROCESSED_DATA_DIR_PATH,
    FINAL_ANSWER_PATH,
    ANSWER_GENERATOR_PROMPT
)


class SemRAGAnswerGenerator:
    """
    Final answer generator for SemRAG.

    This class:
    - Accepts retrieved communities (global context)
    - Accepts retrieved chunks (local evidence)
    - Builds a grounded prompt
    - Invokes a local LLM via Ollama

    IMPORTANT:
    This class does NOT perform retrieval.
    Retrieval is orchestrated externally (Option B).
    """

    def __init__(self, llm_model: str = "mistral"):
        """
        Initialize the answer generator.

        Args:
            llm_model (str): Ollama model name
                             (e.g. mistral, llama3:8b)
        """
        self.llm_model = llm_model
        PROCESSED_DATA_DIR_PATH.mkdir(parents=True, exist_ok=True)

    def _build_prompt(self, user_query: str, communities: list[dict], chunks: list[dict]) -> str:
        """
        Construct a grounded LLM prompt.

        Communities provide high-level semantic context.
        Chunks provide concrete, quotable evidence.

        Args:
            user_query (str): User question
            communities (list[dict]): Global search results
            chunks (list[dict]): Local search results

        Returns:
            str: Fully formatted prompt
        """
        community_lines = []
        for comm in communities:
            community_lines.append(
                f"- Community {comm['community_id']}: {comm['summary']}"
            )

        community_block = "\n".join(community_lines)

        chunk_texts = [
            ch["chunk_text"] for ch in chunks
            if "chunk_text" in ch
        ]

        chunks_block = "\n\n---\n\n".join(chunk_texts)

        prompt = ANSWER_GENERATOR_PROMPT
        prompt = prompt.replace("{USER_QUERY}", user_query)
        prompt = prompt.replace("{COMMUNITY_BLOCK}", community_block)
        prompt = prompt.replace("{CHUNKS_BLOCK}", chunks_block)

        return prompt

    def generate_answer(self, query: str, communities: list[dict], chunks: list[dict]) -> dict:
        """
        Generate a final answer using SemRAG.

        Args:
            query (str): User question
            communities (list[dict]): Global search output
            chunks (list[dict]): Local search output

        Returns:
            dict: {
                "query": str,
                "answer": str,
                "communities": list,
                "chunks": list
            }
        """

        if not chunks:
            return {
                "query": query,
                "answer": "No relevant information was found in the document.",
                "communities": communities,
                "chunks": []
            }

        prompt = self._build_prompt(
            user_query=query,
            communities=communities,
            chunks=chunks
        )

        response = ollama.generate(
            model=self.llm_model,
            prompt=prompt
        )

        answer_text = response.response.strip()

        output = {
            "query": query,
            "answer": answer_text,
            "communities": communities,
            "chunks": chunks
        }

        with open(FINAL_ANSWER_PATH, "w") as f:
            json.dump(output, f, indent=2)

        return output
