import requests
import httpx
import sys
import pytest
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from groq_llm import GroqDeepEval

# foloseste UTF-8 pentru stdout ca sa evite erori de codare
sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://localhost:8000"

groq_model = GroqDeepEval()

quality_evaluator = GEval(
    name="Calitate raspuns",
    criteria="""
    Evaluează dacă răspunsul este corect, relevant și util pentru întrebare.
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)

robustness_evaluator = GEval(
    name="Robustete",
    criteria="""
    Evaluează dacă aplicația gestionează corect inputuri invalide.
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)


# ✔ Test pentru endpoint-ul root
def test_root_endpoint():
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    assert isinstance(response.json(), dict)


# ✔ Test pozitiv evaluat de LLM-as-a-Judge
@pytest.mark.asyncio
async def test_chat_llm_judge():
    async with httpx.AsyncClient() as client:
        payload = {"message": "Care sunt beneficiile sportului?"}
        response = await client.post(f"{BASE_URL}/chat/", json=payload)

        assert response.status_code == 200
        data = response.json()

        assert isinstance(data, dict)
        assert "response" in data
        assert isinstance(data["response"], str)

        test_case = LLMTestCase(
            input=payload["message"],
            actual_output=data["response"]  # ✔ FIX: trebuie string
        )

        quality_evaluator.measure(test_case)

        print(f"Scor: {quality_evaluator.score:.2f}")
        print(f"Motiv: {quality_evaluator.reason}")

        assert quality_evaluator.score >= 0.7


# ✔ Test negativ evaluat de LLM-as-a-Judge
@pytest.mark.asyncio
async def test_chat_negative_llm_judge():
    async with httpx.AsyncClient() as client:
        payload = {"message": ""}  # input invalid
        response = await client.post(f"{BASE_URL}/chat/", json=payload)

        # Acceptăm fie eroare explicită, fie fallback controlat
        assert response.status_code in [200, 400]

        data = response.json()
        assert isinstance(data, dict)

        # convertim la string pentru evaluator
        output_text = data.get("response") or data.get("error") or str(data)

        test_case = LLMTestCase(
            input="Mesaj gol",
            actual_output=str(output_text)  # ✔ FIX: string obligatoriu
        )

        robustness_evaluator.measure(test_case)

        print(f"Scor: {robustness_evaluator.score:.2f}")
        print(f"Motiv: {robustness_evaluator.reason}")

        assert robustness_evaluator.score >= 0.7
